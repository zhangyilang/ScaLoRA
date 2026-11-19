# ScaLoRA
This repository provides codes for reproducing the results in our ICML 2026 paper [ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](https://proceedings.mlr.press/v306/zhang26hi.html).

## Setup
Install the ScaLoRA package in editable mode, together with the dependencies of the experiments:
```bash
pip install -e ".[experiments]"
```
The core package alone (`pip install -e .`) only depends on `torch`, `peft`, and `transformers`.

To run the tests (CPU is enough), which also check the implementation against the one used in the paper:
```bash
pip install -e ".[test]"
pytest
```

## General usage
ScaLoRA combines with Hugging Face's `Trainer` by replacing it with `ScalingTrainer`. Set `scalar_only=True` for the scalar variant.
```python
from scalora import LoraScaling, ScalingTrainer

lora_scaling = LoraScaling(model, warmup_steps=10, interval=1)
trainer = ScalingTrainer(*args, **kwargs, lora_scaling=lora_scaling)
```
Alternatively, it can be directly hooked into a `torch.optim.Optimizer` via
```python
lora_scaling.hook_to_optimizer(optimizer)
```

#### Arguments of `LoraScaling`

| Argument | Default | Description |
| --- | --- | --- |
| `model` | — | A `PeftModel`/`LoraModel`, or any module containing PEFT LoRA layers. |
| `gamma` | `1.0` | Multiplier of the squared optimal scales, corresponding to $1/(L\eta)$ in the paper. |
| `warmup_steps` | `0` | Number of optimizer steps run as vanilla LoRA before scaling starts. |
| `interval` | `1` | Scale every `interval` optimizer steps after warmup (ScaLoRA-I). |
| `scalar_only` | `False` | Use a single scalar per LoRA factor instead of column-wise scales. |

> [!NOTE]
> PEFT initializes `lora_B = 0`, for which the scaling is skipped (Assumption 3.4) until both gradients are nonzero. A short warmup (e.g., 10 steps as in our scripts) is recommended.

#### Supported features
- **LoRA layers:** Linear (including GPT-2 style Conv1D) and Embedding. DoRA, LoRA with bias, quantized base layers, and convolutional LoRA layers are not supported.
- **Optimizers:** Adam/AdamW (including AMSGrad) and SGD, whose moments are rescaled together with the LoRA factors.
- **Precision:** fp32, bf16, and fp16 weights; the scaling factors are always computed in at least fp32 (and solved in fp64).
- **Efficiency:** the computation is batched over all LoRA layers and never synchronizes with the host.
- **Training:** single-GPU and DDP training (as in our scripts). FSDP and DeepSpeed are not supported. Since ScaLoRA modifies the frozen base weights, which are not part of the adapter-only checkpoints, resuming from an intermediate checkpoint is not supported; save the merged model (`model.merge_and_unload()`) instead, as in our scripts.

## Experiments
### GLUE
To perform experiments on the GLUE benchmark, one can leverage the scripts in `glue/scripts/`. For instance, the following commands run the test on the mnli dataset.
```bash
cd glue
bash scripts/mnli.sh
```
### Commonsense reasoning
The scripts take the dataset as an argument, which is one of `boolq`, `piqa`, `social_iqa`, `hellaswag`, `winogrande`, `arc_easy`, `arc_challenge`, and `openbookqa`.
```bash
cd commonsense_reasoning
bash llama2_7b.sh boolq
```

### Overhead
`benchmarks/throughput.py` measures the fine-tuning throughput and peak GPU memory of LoRA and ScaLoRA on one GPU, e.g.,
```bash
python benchmarks/throughput.py --model llama3-8b --method scalora --interval 10
```
Throughput relative to LoRA on one H100 (`benchmarks/results_h100.jsonl`), for the implementation used in the paper and the current one, which batches the computation over all layers. Peak GPU memory is the same as LoRA's.

| Setup | Paper implementation | Current implementation |
| --- | --- | --- |
| DeBERTaV3-base (r=4), ScaLoRA | 84.1% | 96.6% |
| DeBERTaV3-base (r=4), scalar ScaLoRA | 87.2% | 97.2% |
| LLaMA3-8B (r=8), ScaLoRA | 62.7% | 93.8% |
| LLaMA3-8B (r=8), ScaLoRA-I (I=10) | 94.7% | 99.7% |
| LLaMA3-8B (r=8), scalar ScaLoRA-I (I=10) | 96.6% | 100.0% |

## Credits
Our implementation builds upon the following GitHub repositories.
- https://github.com/AGI-Edgerunners/LLM-Adapters
- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/kcc-lion/polar

## Citation
If you find this work useful, please consider citing:
> Y. Zhang, X. Yang, Y. Cai, and G. B. Giannakis, “ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning,” in *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, 2026.

```tex
@inproceedings{ScaLoRA,
 title = {{S}ca{L}o{RA}: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning},
 author = {Zhang, Yilang and Yang, Xiaodong and Cai, Yiwei and Giannakis, Georgios B.},
 booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
 pages = {159735--159760},
 publisher = {PMLR},
 series = {Proceedings of Machine Learning Research},
 volume = {306},
 year = {2026}
}
```
