# Frozen copy of the ScaLoRA implementation used for the paper results (LoraScaling and its helpers).
# Only used by the tests to check the current implementation against it. Do not modify.

import torch
from torch import nn
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTuner
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import LoraLayer, Linear, Embedding


def get_base_tuner(model: PeftModel) -> LoraModel:
    # PeftModel.base_model is a BaseTuner (e.g., LoraModel)
    if isinstance(model, BaseTuner):
        base_tuner = model
    elif isinstance(model, PeftModel):
        base_tuner = model.base_model
    else:
        raise ValueError(f"Expected model to be PeftModel or LoraModel, but got {type(model)} instead")
    
    return base_tuner


def get_lora_weights(layer: LoraLayer, adapter: str) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(layer, Linear):
        lora_A = layer.lora_A[adapter].weight
        lora_B = layer.lora_B[adapter].weight
    elif isinstance(layer, Embedding):
        lora_A = layer.lora_embedding_A[adapter]
        lora_B = layer.lora_embedding_B[adapter]

    return lora_A, lora_B


class LoraScaling:
    def __init__(self, model: nn.Module, gamma: float = 1., warmup_steps: int = 0, interval: int = 1, scalar_only: bool = False) -> None:
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.interval = interval
        self.scalar_only = scalar_only

        self.lora_layers = []
        for module in get_base_tuner(model).modules():
            if not isinstance(module, LoraLayer):
                continue
            if not isinstance(module, (Linear, Embedding)):
                raise ValueError(f"Unsupported LoraLayer type: {type(module)}")
            self.lora_layers.append(module)
    
    @torch.no_grad()
    def scale_lora_adapters(self, optimizer) -> None:
        if self.skip_steps > 0:
            self.skip_steps -= 1
            return
        else:
            self.skip_steps = self.interval - 1
        
        for layer in self.lora_layers:
            for adapter in layer.active_adapters:
                lora_A, lora_B = get_lora_weights(layer, adapter)   # (A: r x m, B: n x r)
                gram_A = lora_A @ lora_A.t()
                gram_B = lora_B.t() @ lora_B
                GB_sq = lora_A.grad ** 2                            # G represents the gradient nabla l
                AG_sq = lora_B.grad ** 2
                AGB_sq = (lora_A @ lora_A.grad.t()) ** 2
                
                if not self.scalar_only:
                    blk_A = gram_A * (lora_B.grad.t() @ lora_B.grad)
                    blk_B = gram_B * (lora_A.grad @ lora_A.grad.t())
                    blk_mat = torch.cat((torch.cat((blk_A, AGB_sq), dim=1), torch.cat((AGB_sq.t(), blk_B), dim=1)))
                    blk_vec = self.gamma * torch.cat((AG_sq.sum(dim=0), GB_sq.sum(dim=1)))
                    scales_sq = torch.linalg.lstsq(blk_mat, blk_vec).solution

                    if (scales_sq >=0).all():
                        r = lora_A.size(0)
                        scale_A_sq, scale_B_sq = scales_sq[:r], scales_sq[r:]
                        scale_A, scale_B = scale_A_sq.sqrt(), scale_B_sq.sqrt()
                        delta_weight = layer.get_delta_weight(adapter)
                        lora_A.data *= scale_A.unsqueeze(1)
                        lora_B.data *= scale_B
                        lora_A.grad *= scale_B.unsqueeze(1)
                        lora_B.grad *= scale_A
                        if optimizer.state:
                            optimizer.state[lora_A]['exp_avg'] *= scale_B.unsqueeze(1)
                            optimizer.state[lora_B]['exp_avg'] *= scale_A
                            optimizer.state[lora_A]['exp_avg_sq'] *= scale_B_sq.unsqueeze(1)
                            optimizer.state[lora_B]['exp_avg_sq'] *= scale_A_sq
                        layer.weight.data += delta_weight - layer.get_delta_weight(adapter)
                        continue

                GB_normsq = GB_sq.sum()
                AG_normsq = AG_sq.sum()
                AGB_normsq = AGB_sq.sum()
                GBB_normsq = (lora_A.grad * (gram_B @ lora_A.grad)).sum()
                AAG_normsq = ((lora_B.grad @ gram_A) * lora_B.grad).sum()

                numerator_A = AG_normsq * GBB_normsq - GB_normsq * AGB_normsq
                denominator = AAG_normsq * GBB_normsq - AGB_normsq ** 2

                if numerator_A < 0 or (denominator == 0 and GBB_normsq > 0):        # scale_A = 0, scale_B^2 > 0
                    scale_A_sq, scale_A = 0, 0
                    scale_B_sq = self.gamma * GB_normsq / GBB_normsq
                    scale_B = scale_B_sq.sqrt()
                else:
                    numerator_B = GB_normsq * AAG_normsq - AG_normsq * AGB_normsq
                    if numerator_B < 0 or (denominator == 0 and AAG_normsq > 0):    # scale_A^2 > 0, scale_B = 0
                        scale_A_sq = self.gamma * AG_normsq / AAG_normsq
                        scale_A = scale_A_sq.sqrt()
                        scale_B_sq, scale_B = 0, 0
                    else:   # scale_A^2 > 0, scale_B^2 > 0
                        scale_A_sq = self.gamma * numerator_A / denominator
                        scale_B_sq = self.gamma * numerator_B / denominator
                        scale_A, scale_B = scale_A_sq.sqrt(), scale_B_sq.sqrt()
                
                # scale lora adapters
                delta_weight = layer.get_delta_weight(adapter)
                lora_A.data *= scale_A
                lora_B.data *= scale_B
                lora_A.grad *= scale_B
                lora_B.grad *= scale_A
                if optimizer.state:
                    optimizer.state[lora_A]['exp_avg'] *= scale_B
                    optimizer.state[lora_B]['exp_avg'] *= scale_A
                    optimizer.state[lora_A]['exp_avg_sq'] *= scale_B_sq
                    optimizer.state[lora_B]['exp_avg_sq'] *= scale_A_sq
                layer.weight.data += (1 - scale_A * scale_B) * delta_weight

    def hook_to_optimizer(self, optimizer) -> None:
        self.skip_steps = self.warmup_steps

        def step_pre_hook(optimizer, args, kwargs) -> None:
            self.scale_lora_adapters(optimizer)
    
        optimizer.register_step_pre_hook(step_pre_hook)
