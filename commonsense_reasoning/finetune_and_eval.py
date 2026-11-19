import json
import os
import sys
from typing import List

import datasets
import fire
import lm_eval
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from peft import LoraConfig, get_peft_model
from prompter_utils import Prompter, generate_lm_eval_prompts
from scalora import LoraScaling, ScalingTrainer


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    # training hyperparams
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    dtype: str = "bf16",
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    # hyperparams of ScaLoRA
    scale_lora: bool = False,
    scalar_scale_only: bool = False,
    scale_warmup: int = 10,
    scale_interval: int = 1, 
    gamma: float = 1.,
    # llm hyperparams
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    # other
    save_strategy: str = "no",
    lr_scheduler_type: str = "cosine",
    do_eval: bool = True,
    activate_profiling: bool = False,
):
    eval_task = data_path
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Fine-Tuning with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"adapter_name: {adapter_name}\n"
            f"seed: {seed}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"dtype: {dtype}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"scale_lora: {scale_lora}\n"
            f"scalar_scale_only: {scalar_scale_only}\n"
            f"scale_warmup: {scale_warmup}\n"
            f"scale_interval: {scale_interval}\n"
            f"gamma: {gamma}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"eval_task: {eval_task}\n"
            f"save_strategy: {save_strategy}\n"
            f"lr_scheduler_type: {lr_scheduler_type}\n"
            f"do_eval: {do_eval}\n"
            f"activate_profiling: {activate_profiling}\n"
        )
        cli_run_args = {
            "epoch": num_epochs,
            "dataset": data_path,
            "model_name": base_model,
            "adapter": adapter_name,
            "lr": learning_rate,
            "seed": seed,
            "rank": lora_r,
            "dtype": dtype,
            "wandb_name": wandb_run_name,
            "eval_task": eval_task,
            "save_strategy": save_strategy,
            "lr_scheduler_type": lr_scheduler_type,
            "do_eval": do_eval,
            "activate_profiling": activate_profiling,
        }
        with open(os.path.join(output_dir, "cli_run_args.json"), "w") as file:
            json.dump(cli_run_args, file, indent=4)
    assert base_model, "Please specify a --base_model"
    assert dtype in ["bf16", "fp16"], f"Invalid dtype: {dtype}"
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device_map = "auto" if world_size > 1 else {"": 0}
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    transformers.set_seed(seed)
    if "llama" in base_model.lower():
        print(f"Using torch dtype: {torch_dtype}")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        gradient_checkpointing = False
        if "Llama-2-7b-hf" in base_model:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = prepare_model_for_int8_training(model)
    else:
        raise ValueError(f"Invalid base model: {base_model}")

    def tokenize(prompt, add_eos_token=True):
        if "prompt" in prompt.keys():
            prompt = prompt["prompt"]
        else:
            prompt = prompter.generate_prompt(prompt)
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=(adapter_name.lower() == "dora"),
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        prompter = Prompter(verbose=False)
        data = load_dataset("json", data_files=data_path)
    else:
        # dataset based on lm eval harness
        prompts = generate_lm_eval_prompts(task_name=data_path)
        data = datasets.Dataset.from_list(prompts)

    model.print_trainable_parameters()

    if val_set_size > 0:
        if isinstance(data, datasets.dataset_dict.DatasetDict):
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
        else:
            train_val = data.train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
        train_data = train_val["train"].shuffle().map(tokenize)
        val_data = train_val["test"].shuffle().map(tokenize)
    else:
        if isinstance(data, datasets.dataset_dict.DatasetDict):
            train_data = data["train"].shuffle().map(tokenize)
        else:
            train_data = data.shuffle().map(tokenize)
        val_data = None

    use_bf16 = dtype == "bf16"
    if adapter_name.lower() == "lora" and scale_lora:
        lora_scaling = LoraScaling(model, 
                                   gamma=gamma, 
                                   warmup_steps=scale_warmup,
                                   interval=scale_interval,
                                   scalar_only=scalar_scale_only)
    else:
        lora_scaling = None
    # trainer = transformers.Trainer(
    trainer = ScalingTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type,
            # warmup_steps=100,
            warmup_ratio=0.03,
            gradient_checkpointing=gradient_checkpointing,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=0.02,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy=save_strategy,
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            ddp_find_unused_parameters=True if gradient_checkpointing else False,
            output_dir=output_dir,
            save_total_limit=1,
            seed=seed,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        lora_scaling=lora_scaling
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    if activate_profiling:
        from transformers.trainer_callback import TrainerCallback

        class ProfCallback(TrainerCallback):
            def __init__(self, prof):
                self.prof = prof

            def on_step_end(self, args, state, control, **kwargs):
                self.prof.step()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=5, wait=5, warmup=5, active=3, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(output_dir, "profiling")
            ),
            profile_memory=True,
            with_flops=False,
            with_stack=False,
            record_shapes=False,
        ) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train()
    else:
        trainer.train()

    if scale_lora:
        trainer.model = model.merge_and_unload()
        state_dict = trainer.model.state_dict()
        if "llama" in base_model.lower() and 'lm_head.0.weight' in state_dict:
            state_dict['lm_head.weight'] = state_dict.pop('lm_head.0.weight')
        model_args_str = f"pretrained={output_dir},trust_remote_code=True"
    else:
        state_dict = trainer.model.state_dict()
        model_args_str = f"pretrained={base_model},peft={output_dir},trust_remote_code=True"
    trainer._save(output_dir, state_dict=state_dict)
    torch.distributed.barrier()
    
    if do_eval:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args_str,
            tasks=eval_task,
            batch_size=micro_batch_size,
            num_fewshot=None,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            log_samples=False,
            write_out=True,
        )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            # save eval results to json
            dumped = json.dumps(
                results,
                indent=2,
                default=lm_eval.utils.handle_non_serializable,
                ensure_ascii=False,
            )
            with open(
                os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8"
            ) as file:
                file.write(dumped)
            print(lm_eval.utils.make_table(results))


# from previous version of peft
def prepare_model_for_int8_training(
    model, output_embedding_layer_name="lm_head", use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]
):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

        if loaded_in_8bit:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
                param.data = param.data.to(torch.float32)

    if loaded_in_8bit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
            in fp32

            """

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


if __name__ == "__main__":
    fire.Fire(train)
