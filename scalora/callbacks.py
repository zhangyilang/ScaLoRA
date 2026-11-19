import torch
import numpy as np
from transformers import TrainerCallback
from peft.tuners.lora.layer import LoraLayer, Linear, Embedding

from .utils import get_base_tuner


class LoraRankCallback(TrainerCallback):
    def __init__(self, model: torch.nn.Module, interval: int = 10, pretrained_change=False):
        """
        Callback to log the rank of LoRA update.
        
        Args:
            interval (int): The number of steps between logging the rank.
        """
        self.interval = interval
        self.counter = 0

        self.lora_layers = list()
        for module in get_base_tuner(model).modules():
            if not isinstance(module, LoraLayer):
                continue
            if not isinstance(module, (Linear, Embedding)):
                raise ValueError(f"Unsupported LoraLayer type: {type(module)}")
            if pretrained_change:
                pretrained_weight = module.get_base_layer().weight.data.detach().clone()
            else:
                pretrained_weight = None
            self.lora_layers.append([module, pretrained_weight])

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        self.counter += 1
        if self.counter % self.interval == 0:
            ranks = list()
            for layer, pretrained_weight in self.lora_layers:
                weight_change = sum(layer.get_delta_weight(adapter) for adapter in layer.active_adapters)
                if pretrained_weight is not None:
                    weight_change += layer.get_base_layer().weight.data - pretrained_weight
                ranks.append(torch.linalg.matrix_rank(weight_change).item())

            print(f'Rank: {np.mean(ranks)} +/- {np.std(ranks)}')


class FullftRankCallback(TrainerCallback):
    def __init__(self, model: torch.nn.Module, target_modules: list[str], interval: int = 10):
        """
        Callback to log the rank of full finetuning update.
        
        Args:
            interval (int): The number of steps between logging the rank.
        """
        self.interval = interval
        self.counter = 0

        self.target_layers = list()
        for name, module in model.named_modules():
            # re match
            if any(name.endswith(target) for target in target_modules):
                self.target_layers.append([module, module.weight.data.detach().clone()])

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        self.counter += 1
        if self.counter % self.interval == 0:
            ranks = list()
            for layer, pretrained_weight in self.target_layers:
                ranks.append(torch.linalg.matrix_rank(layer.weight.data - pretrained_weight).item())

            print(f'Rank: {np.mean(ranks)} +/- {np.std(ranks)}')


class FullftGradRankCallback(TrainerCallback):
    def __init__(self, model: torch.nn.Module, target_modules: list[str], interval: int = 10):
        """
        Callback to log the grad rank (among layers) of full ft.
        
        Args:
            interval (int): The number of steps between logging the rank.
        """
        self.interval = interval
        self.counter = 0

        self.target_layers = list()
        for name, module in model.named_modules():
            if any(name.endswith(target) for target in target_modules):
                self.target_layers.append(module)

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        self.counter += 1
        if self.counter % self.interval == 0:
            ranks = list()
            for layer in self.target_layers:
                if hasattr(layer, 'weight'):
                    ranks.append(torch.linalg.matrix_rank(layer.weight.grad).item())

            print(f'Rank: {np.mean(ranks)} +/- {np.std(ranks)}')
