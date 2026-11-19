import warnings
from collections import defaultdict
from typing import Optional

import torch
from torch import nn
from peft.tuners.lora.layer import LoraLayer, Linear, Embedding
from transformers import Trainer

from .utils import get_base_tuner, get_lora_weights

# Optimizer-state entries that transform like the gradient (first moments) or like its square (second moments)
_FIRST_MOMENT_KEYS = ("exp_avg", "momentum_buffer")
_SECOND_MOMENT_KEYS = ("exp_avg_sq", "max_exp_avg_sq")


def _moments(optimizer: torch.optim.Optimizer, param: torch.Tensor) -> list:
    '''(tensor, order) for each first (order 1) and second (order 2) moment of `param` in the optimizer state.'''
    moments = []
    for key, value in optimizer.state.get(param, {}).items():
        if key == "step" or not torch.is_tensor(value):
            continue
        if key in _FIRST_MOMENT_KEYS:
            moments.append((value, 1))
        elif key in _SECOND_MOMENT_KEYS:
            moments.append((value, 2))
        else:
            raise RuntimeError(f"ScaLoRA cannot rescale optimizer state '{key}' of {type(optimizer).__name__}; "
                               "supported optimizers are Adam/AdamW (including AMSGrad) and SGD")
    return moments


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    # Gram matrices are computed in at least fp32
    return torch.promote_types(dtype, torch.float32)


def optimal_scales_sq(gram_P: torch.Tensor, gram_Q: torch.Tensor, gamma: float = 1.,
                      scalar_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    '''Squared optimal scales of a batch of LoRA layers.

    With A = lora_A (r x m), B = lora_B (n x r), and their gradients gA, gB, the inputs are the Gram matrices
    gram_P = P P^T and gram_Q = Q^T Q of P = [A; gA] (2r x m) and Q = [gB, B] (n x 2r), stacked along dim 0.

    The column-wise optimum solves ((P P^T) ⊙ (Q^T Q)) v = [diag(gB^T gB); diag(gA gA^T)] (Theorem 3.7). It is
    used if v >= 0; otherwise, or if `scalar_only`, the scalar optimum (Theorem 3.5) is used, whose 2 x 2
    system consists of the block sums of the same matrix. Layers with a zero gradient (Assumption 3.4) or a
    non-finite solution are left unscaled.

    Returns:
        (scale_A^2, scale_B^2), each of shape (batch, r).
    '''
    r = gram_P.shape[-1] // 2
    out_dtype = gram_P.dtype
    # the 2r x 2r systems are tiny and can be ill-conditioned, so solve them in fp64 (MPS has no fp64)
    dtype = torch.float32 if gram_P.device.type == "mps" else torch.float64
    gram_P, gram_Q = gram_P.to(dtype), gram_Q.to(dtype)
    mat = gram_P * gram_Q                                                       # (L, 2r, 2r)
    vec = torch.cat((gram_Q.diagonal(dim1=1, dim2=2)[:, :r],
                     gram_P.diagonal(dim1=1, dim2=2)[:, r:]), dim=1)            # (L, 2r)

    # scalar scales: 2 x 2 non-negative least squares in closed form
    blk_mat = mat.view(-1, 2, r, 2, r).sum((2, 4))
    blk_vec = vec.view(-1, 2, r).sum(-1)                                        # (|gB|^2, |gA|^2)
    aa, ab, bb = blk_mat[:, 0, 0], blk_mat[:, 0, 1], blk_mat[:, 1, 1]
    ca, cb = blk_vec[:, 0], blk_vec[:, 1]
    num_A = ca * bb - cb * ab
    num_B = cb * aa - ca * ab
    det = aa * bb - ab * ab
    zero_A = (num_A < 0) | ((det == 0) & (bb > 0))
    zero_B = ~zero_A & ((num_B < 0) | ((det == 0) & (aa > 0)))
    sq_A = torch.where(zero_A, 0., torch.where(zero_B, ca / aa, num_A / det))
    sq_B = torch.where(zero_A, cb / bb, torch.where(zero_B, 0., num_B / det))
    sq = torch.stack((sq_A, sq_B), dim=1).repeat_interleave(r, dim=1)           # (L, 2r)

    # column-wise scales
    if not scalar_only:
        sol, info = torch.linalg.solve_ex(mat, vec.unsqueeze(-1))
        sol = sol.squeeze(-1)
        use_sol = (info == 0) & (sol >= 0).all(dim=1) & sol.isfinite().all(dim=1)
        sq = torch.where(use_sol.unsqueeze(1), sol, sq)

    sq = gamma * sq
    valid = (blk_vec > 0).all(dim=1, keepdim=True) & (sq >= 0).all(dim=1, keepdim=True) \
        & sq.isfinite().all(dim=1, keepdim=True)
    sq = torch.where(valid, sq, 1.).to(out_dtype)
    return sq[:, :r], sq[:, r:]


def _merge_residual(layer: LoraLayer, adapter: str, lora_A: torch.Tensor, lora_B: torch.Tensor,
                    residual: torch.Tensor) -> None:
    '''W0 += s * B diag(residual) A in place, without materializing a full-size temporary.'''
    weight = layer.get_base_layer().weight
    B = (lora_B * residual).to(weight.dtype)
    A = lora_A.to(weight.dtype)
    if isinstance(layer, Embedding) or layer.fan_in_fan_out:
        weight.addmm_(A.mT, B.mT, alpha=layer.scaling[adapter])
    else:
        weight.addmm_(B, A, alpha=layer.scaling[adapter])


class LoraScaling:
    '''ScaLoRA: optimal scaling of the LoRA factors right before every optimizer step.

    For a LoRA layer W = W0 + s * B A, the rows of A and the columns of B are rescaled as A <- diag(a) A and
    B <- B diag(b), and s * B diag(1 - a * b) A is merged into W0 so that W stays unchanged. The gradients and
    the optimizer moments are rescaled accordingly, so the optimizer does not need to be restarted.

    The computation is batched over all layers sharing a rank (one batched linear solve, and one batched
    Gram matrix per factor shape) and never synchronizes with the host.

    Args:
        model: a `PeftModel`/`LoraModel`, or any module containing PEFT LoRA layers on `nn.Linear`, Conv1D,
            or `nn.Embedding` base layers.
        gamma: multiplier of the squared scales; it corresponds to 1 / (L * eta) in the paper.
        warmup_steps: optimizer steps run as vanilla LoRA before scaling starts. PEFT initializes
            `lora_B = 0`, for which the scaling is skipped (Assumption 3.4), so a short warmup is recommended.
        interval: scale every `interval` optimizer steps after warmup.
        scalar_only: use a single scalar per factor instead of one scale per rank-1 component.
    '''

    def __init__(self, model: nn.Module, gamma: float = 1., warmup_steps: int = 0, interval: int = 1,
                 scalar_only: bool = False) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if interval < 1:
            raise ValueError(f"interval must be a positive integer, got {interval}")

        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.interval = interval
        self.scalar_only = scalar_only
        self.skip_steps = warmup_steps
        self._hook_handle = None
        self._batches = None
        self._batch_signature = None

        self.lora_weights = []  # (layer, adapter, A: r x m, B: n x r)
        for module in get_base_tuner(model).modules():
            if not isinstance(module, LoraLayer):
                continue
            if not isinstance(module, (Linear, Embedding)):
                raise ValueError(f"Unsupported LoraLayer type: {type(module)}; "
                                 "only Linear (including Conv1D) and Embedding layers are supported")
            for adapter in module.active_adapters:
                weights = get_lora_weights(module, adapter)
                if weights is None:
                    continue
                if getattr(module, "use_dora", {}).get(adapter, False):
                    raise ValueError("ScaLoRA does not support DoRA adapters")
                if getattr(module, "lora_bias", {}).get(adapter, False):
                    raise ValueError("ScaLoRA does not support LoRA adapters with bias (lora_bias=True)")
                self.lora_weights.append((module, adapter, *weights))

        if not self.lora_weights:
            warnings.warn("No LoRA layers found in the model; LoraScaling will have no effect.")
        elif warmup_steps == 0 and any(not A.any() or not B.any() for _, _, A, B in self.lora_weights):
            warnings.warn("Some LoRA factors are all zeros (PEFT initializes lora_B = 0). Scaling is skipped for "
                          "those layers until both gradients are nonzero; consider warmup_steps > 0.")

    def _skip(self) -> bool:
        '''Advance the warmup/interval schedule by one step; return True if this step is skipped.'''
        if self.skip_steps > 0:
            self.skip_steps -= 1
            return True
        self.skip_steps = self.interval - 1
        return False

    def _get_batches(self) -> list:
        '''Group layers sharing rank, device, and dtypes; within a group, index the layers by factor shape.

        Cached, and rebuilt only if a device or dtype changes (e.g., the model is moved after creating
        LoraScaling). Index tensors live on the device to avoid host syncs.
        '''
        signature = [(A.device, A.dtype, B.dtype) for _, _, A, B in self.lora_weights]
        if signature == self._batch_signature:
            return self._batches

        groups = defaultdict(list)
        for entry in self.lora_weights:
            A, B = entry[2], entry[3]
            groups[(A.shape[0], A.device, A.dtype, B.dtype)].append(entry)

        def by_shape(entries, factor, device):
            shapes = defaultdict(list)
            for pos, entry in enumerate(entries):
                shapes[tuple(entry[factor].shape)].append(pos)
            return [(positions, torch.tensor(positions, device=device)) for positions in shapes.values()]

        self._batches = [(entries, by_shape(entries, 2, device), by_shape(entries, 3, device))
                         for (_, device, _, _), entries in groups.items()]
        self._batch_signature = signature
        return self._batches

    def _batched_scales_sq(self, entries: list, by_A: list, by_B: list) -> tuple[torch.Tensor, torch.Tensor]:
        '''Gram matrices with one batched matmul per factor shape, then `optimal_scales_sq` for all layers.'''
        A0, B0 = entries[0][2], entries[0][3]
        L, r, device = len(entries), A0.shape[0], A0.device
        dtype = _compute_dtype(torch.promote_types(A0.dtype, B0.dtype))

        def grad(param):    # a missing gradient leaves the layer unscaled (Assumption 3.4)
            return param.grad if param.grad is not None else torch.zeros_like(param)

        gram_P = torch.empty(L, 2 * r, 2 * r, dtype=dtype, device=device)
        for positions, index in by_A:
            As = [entries[p][2] for p in positions]
            P = torch.cat((torch.stack(As), torch.stack([grad(A) for A in As])), dim=1).to(dtype)  # (l, 2r, m)
            gram_P.index_copy_(0, index, P @ P.mT)

        gram_Q = torch.empty(L, 2 * r, 2 * r, dtype=dtype, device=device)
        for positions, index in by_B:
            Bs = [entries[p][3] for p in positions]
            Q = torch.cat((torch.stack([grad(B) for B in Bs]), torch.stack(Bs)), dim=2).to(dtype)  # (l, n, 2r)
            gram_Q.index_copy_(0, index, Q.mT @ Q)

        return optimal_scales_sq(gram_P, gram_Q, self.gamma, self.scalar_only)

    @torch.no_grad()
    def scale_lora_adapters(self, optimizer: torch.optim.Optimizer) -> None:
        if self._skip():
            return

        for entries, by_A, by_B in self._get_batches():
            active = [A.grad is not None and B.grad is not None for _, _, A, B in entries]
            if not any(active):
                continue

            sq_A, sq_B = self._batched_scales_sq(entries, by_A, by_B)
            scale_A, scale_B = sq_A.sqrt(), sq_B.sqrt()
            residual = 1 - scale_A * scale_B
            # tensors shaped like A (r x m) are scaled along rows, those shaped like B (n x r) along columns
            rows_A, rows_B, rows_B_sq = scale_A.unsqueeze(-1), scale_B.unsqueeze(-1), sq_B.unsqueeze(-1)

            tensors, scales = [], []
            for (layer, adapter, A, B), act, a, b, a_sq, a_row, b_row, b_sq_row, res in zip(
                    entries, active, scale_A, scale_B, sq_A, rows_A, rows_B, rows_B_sq, residual):
                if not act:
                    continue
                _merge_residual(layer, adapter, A, B, res)   # uses A and B before rescaling
                tensors += [A, B, A.grad, B.grad]
                scales += [a_row, b, b_row, a]
                for value, order in _moments(optimizer, A):
                    tensors.append(value)
                    scales.append(b_row if order == 1 else b_sq_row)
                for value, order in _moments(optimizer, B):
                    tensors.append(value)
                    scales.append(a if order == 1 else a_sq)
            torch._foreach_mul_(tensors, scales)

    def hook_to_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        '''Run `scale_lora_adapters` before every `optimizer.step()`; calling it again replaces the previous hook.'''
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self.skip_steps = self.warmup_steps

        def scaling_hook(optimizer, args, kwargs) -> None:
            self.scale_lora_adapters(optimizer)
        self._hook_handle = optimizer.register_step_pre_hook(scaling_hook)


class ScalingTrainer(Trainer):
    '''`transformers.Trainer` that applies `lora_scaling` (a `LoraScaling`) before every optimizer step.

    ScaLoRA modifies the frozen base weights, which are not part of the adapter-only checkpoints saved by the
    Trainer, so resuming from an intermediate checkpoint is not supported. Save the merged model instead
    (`model.merge_and_unload()`).
    '''

    def __init__(self, *args, lora_scaling: Optional[LoraScaling] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_scaling = lora_scaling

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if self.lora_scaling is not None:
            self.lora_scaling.hook_to_optimizer(optimizer)

        return optimizer
