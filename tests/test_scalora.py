"""Tests for ScaLoRA; run with `pytest` (CPU is enough)."""
import pytest
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers.pytorch_utils import Conv1D

from scalora import LoraScaling
from scalora.scalora import optimal_scales_sq
from scalora.utils import get_lora_weights
from reference_scalora import LoraScaling as ReferenceLoraScaling

pytestmark = pytest.mark.filterwarnings("ignore:fan_in_fan_out")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 16)
        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24, 16)
        self.conv = Conv1D(12, 16)      # GPT-2 style layer, i.e., fan_in_fan_out=True

    def forward(self, x, labels=None):
        h = self.emb(x)
        out = self.conv(h + self.fc2(torch.tanh(self.fc1(h))))
        return out if labels is None else {"loss": ((out - labels) ** 2).mean(), "logits": out}


TARGETS = ("emb", "fc1", "fc2", "conv")


def make_model(targets=TARGETS, seed=0, dtype=None, **lora_kwargs):
    torch.manual_seed(seed)
    net = Net() if dtype is None else Net().to(dtype)
    return get_peft_model(net, LoraConfig(r=4, lora_alpha=8, target_modules=list(targets), **lora_kwargs))


def make_data(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 50, (32,), generator=g), torch.randn(32, 12, generator=g)) for _ in range(n)]


def loss_fn(model, x, y):
    out = model(x)
    dtype = torch.promote_types(out.dtype, torch.float32)
    return ((out.to(dtype) - y.to(dtype)) ** 2).mean()


def train(model, scaling, make_optimizer=None, steps=10):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params) if make_optimizer else torch.optim.AdamW(params, lr=1e-2, weight_decay=0.01)
    scaling.hook_to_optimizer(optimizer)
    for x, y in make_data(steps):
        optimizer.zero_grad()
        loss_fn(model, x, y).backward()
        optimizer.step()
    return optimizer


def all_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def all_finite(model):
    return all(torch.isfinite(p).all() for p in model.parameters())


def effective_weights(model):
    from peft.tuners.lora.layer import LoraLayer
    return [m.get_base_layer().weight.double() + sum(m.get_delta_weight(a).double() for a in m.active_adapters)
            for m in model.modules() if isinstance(m, LoraLayer)]


def pretrain(model, steps=3):
    '''A few plain AdamW steps so that lora_B != 0 and the optimizer has state.'''
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    for x, y in make_data(steps, seed=5):
        optimizer.zero_grad()
        loss_fn(model, x, y).backward()
        optimizer.step()
    return optimizer


# ---------- same results as the paper implementation ----------

@pytest.mark.parametrize("scalar_only", [False, True])
@pytest.mark.parametrize("warmup", [2, 5])
@pytest.mark.parametrize("interval", [1, 3])
def test_matches_reference_fp64(scalar_only, warmup, interval):
    # fp64, where the reference's least-squares solve is accurate; two ranks to exercise several batches
    kwargs = dict(scalar_only=scalar_only, warmup_steps=warmup, interval=interval, gamma=0.8)
    ref_model = make_model(dtype=torch.float64, rank_pattern={"fc2": 2})
    model = make_model(dtype=torch.float64, rank_pattern={"fc2": 2})
    train(ref_model, ReferenceLoraScaling(ref_model, **kwargs))
    scaling = LoraScaling(model, **kwargs)
    train(model, scaling)
    assert len(scaling._get_batches()) == 2
    expected, actual = all_params(ref_model), all_params(model)
    assert expected.keys() == actual.keys()
    assert all(torch.allclose(expected[k], actual[k], rtol=1e-9, atol=1e-12) for k in expected)


@pytest.mark.parametrize("interval", [1, 3])
def test_scalar_matches_reference_fp32(interval):
    kwargs = dict(scalar_only=True, warmup_steps=2, interval=interval)
    ref_model, model = make_model(), make_model()
    train(ref_model, ReferenceLoraScaling(ref_model, **kwargs))
    train(model, LoraScaling(model, **kwargs))
    expected, actual = all_params(ref_model), all_params(model)
    assert all(torch.allclose(expected[k], actual[k], rtol=1e-4, atol=1e-6) for k in expected)


def test_fp32_scales_match_fp64():
    # the 2r x 2r systems can be badly conditioned (e.g., Embedding layers); they are solved in fp64
    model = make_model()
    pretrain(model)
    x, y = make_data(1, seed=7)[0]
    loss_fn(model, x, y).backward()
    scaling = LoraScaling(model)
    for entries, by_A, by_B in scaling._get_batches():
        sq32 = scaling._batched_scales_sq(entries, by_A, by_B)
        for _, _, A, B in entries:
            A.data, A.grad, B.data, B.grad = A.double(), A.grad.double(), B.double(), B.grad.double()
        sq64 = scaling._batched_scales_sq(entries, by_A, by_B)
        for s32, s64 in zip(sq32, sq64):
            assert torch.allclose(s32.double(), s64, rtol=1e-3, atol=1e-6)


# ---------- math ----------

def random_layer(seed, r=4, m=30, n=20, scale_B=1.):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r, m, generator=g, dtype=torch.float64)
    B = torch.randn(n, r, generator=g, dtype=torch.float64) * scale_B
    G = torch.randn(n, 3, generator=g, dtype=torch.float64) @ torch.randn(3, m, generator=g, dtype=torch.float64)
    return A, B, G, B.t() @ G, G @ A.t()     # gradients of l(W0 + B A) w.r.t. A and B


def scales_sq(A, B, gA, gB, **kwargs):
    P, Q = torch.cat((A, gA)), torch.cat((gB, B), dim=1)
    sq_A, sq_B = optimal_scales_sq((P @ P.mT)[None], (Q.mT @ Q)[None], **kwargs)
    return sq_A[0], sq_B[0]


def objective(A, B, G, sq_A, sq_B):
    # || G - G A^T diag(sq_A) A - B diag(sq_B) B^T G ||_F^2, i.e., (7) with gamma = 1 / (L eta) = 1
    return (G - G @ A.t() @ (sq_A[:, None] * A) - (B * sq_B) @ B.t() @ G).square().sum()


@pytest.mark.parametrize("scalar_only", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_scales_minimize_objective(scalar_only, seed):
    A, B, G, gA, gB = random_layer(seed, scale_B=10. ** -(seed % 4))
    sq_A, sq_B = scales_sq(A, B, gA, gB, scalar_only=scalar_only)
    assert (sq_A >= 0).all() and (sq_B >= 0).all()
    best = objective(A, B, G, sq_A, sq_B)
    g = torch.Generator().manual_seed(100 + seed)
    for _ in range(50):
        if scalar_only:     # optimal among uniform scales
            da, db = 1e-3 * torch.randn(2, generator=g, dtype=torch.float64)
            da, db = da.expand(4), db.expand(4)
        else:               # optimal among column-wise scales, or the scalar fallback is used
            da, db = 1e-3 * torch.randn(2, 4, generator=g, dtype=torch.float64)
            if not (sq_A.unique().numel() > 1 or sq_B.unique().numel() > 1):
                continue
        assert objective(A, B, G, (sq_A + da).clamp_min(0), (sq_B + db).clamp_min(0)) >= best * (1 - 1e-10)


def test_gamma_multiplies_squared_scales():
    A, B, G, gA, gB = random_layer(0)
    for scalar_only in (False, True):
        sq1 = scales_sq(A, B, gA, gB, scalar_only=scalar_only)
        sq2 = scales_sq(A, B, gA, gB, scalar_only=scalar_only, gamma=0.3)
        assert all(torch.allclose(0.3 * s1, s2) for s1, s2 in zip(sq1, sq2))


@pytest.mark.parametrize("which", ["B", "G"])
def test_zero_factor_or_gradient_is_skipped(which):
    A, B, G, gA, gB = random_layer(0)
    if which == "B":    # lora_B = 0 at initialization, hence gA = 0
        B, gA = torch.zeros_like(B), torch.zeros_like(gA)
    else:
        gA, gB = torch.zeros_like(gA), torch.zeros_like(gB)
    for scalar_only in (False, True):
        sq_A, sq_B = scales_sq(A, B, gA, gB, scalar_only=scalar_only)
        assert torch.equal(sq_A, torch.ones(4, dtype=torch.float64))
        assert torch.equal(sq_B, torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("scalar_only", [False, True])
def test_scaling_keeps_weights_and_transforms_gradients_and_moments(scalar_only):
    model = make_model(dtype=torch.float64)
    optimizer = pretrain(model)
    x, y = make_data(1, seed=7)[0]
    optimizer.zero_grad()
    loss_fn(model, x, y).backward()
    scaling = LoraScaling(model, scalar_only=scalar_only)
    before = {id(p): (p.detach().clone(), p.grad.clone(), {k: v.clone() for k, v in optimizer.state[p].items()})
              for p in model.parameters() if p.requires_grad}
    weights, out = effective_weights(model), model(x).detach()
    scaling.scale_lora_adapters(optimizer)

    assert all(torch.allclose(w1, w2, atol=1e-12) for w1, w2 in zip(weights, effective_weights(model)))
    assert torch.allclose(model(x).detach(), out, atol=1e-12)

    num_scaled = 0
    for _, _, A, B in scaling.lora_weights:
        A0, gA0, stA0 = before[id(A)]
        B0, gB0, stB0 = before[id(B)]
        a = (A * A0).sum(1) / A0.square().sum(1)                   # A = diag(a) A0
        b = (B * B0).sum(0) / B0.square().sum(0)                   # B = B0 diag(b)
        assert torch.allclose(A, a[:, None] * A0) and torch.allclose(B, B0 * b)
        num_scaled += int(not torch.allclose(a * b, torch.ones_like(a)))
        assert torch.allclose(A.grad, b[:, None] * gA0) and torch.allclose(B.grad, gB0 * a)
        st_A, st_B = optimizer.state[A], optimizer.state[B]
        assert torch.allclose(st_A["exp_avg"], b[:, None] * stA0["exp_avg"])
        assert torch.allclose(st_A["exp_avg_sq"], b[:, None] ** 2 * stA0["exp_avg_sq"])
        assert torch.allclose(st_B["exp_avg"], stB0["exp_avg"] * a)
        assert torch.allclose(st_B["exp_avg_sq"], stB0["exp_avg_sq"] * a ** 2)
    assert num_scaled > 0

    # the rescaled gradients are the gradients at the rescaled factors
    grads = {id(p): p.grad.clone() for p in model.parameters() if p.requires_grad}
    optimizer.zero_grad()
    loss_fn(model, x, y).backward()
    assert all(torch.allclose(grads[id(p)], p.grad, atol=1e-12) for p in model.parameters() if p.requires_grad)


# ---------- layers, precision, and initialization ----------

def test_layers_without_grad_are_skipped():
    model = make_model()
    for name, param in model.named_parameters():
        if "lora" in name and "fc2" in name:
            param.requires_grad_(False)
    frozen = {k: v for k, v in all_params(model).items() if "fc2" in k}
    train(model, LoraScaling(model, warmup_steps=2))
    after = all_params(model)
    assert all_finite(model)
    assert all(torch.equal(frozen[k], after[k]) for k in frozen)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalar_only", [False, True])
def test_low_precision_base_weights(dtype, scalar_only):
    model = make_model(dtype=dtype)     # PEFT keeps the adapters in fp32
    train(model, LoraScaling(model, warmup_steps=2, scalar_only=scalar_only), steps=5)
    assert all_finite(model)
    optimizer = pretrain(model)
    x, y = make_data(1, seed=7)[0]
    loss_fn(model, x, y).backward()
    scaling = LoraScaling(model, scalar_only=scalar_only)
    bases = [layer.get_base_layer().weight.double() for layer, _, _, _ in scaling.lora_weights]
    weights = effective_weights(model)
    scaling.scale_lora_adapters(optimizer)
    # unchanged up to the rounding of the merged base weights (and of the merged residual)
    eps = torch.finfo(dtype).eps
    for (layer, _, _, _), base, w1, w2 in zip(scaling.lora_weights, bases, weights, effective_weights(model)):
        new_base = layer.get_base_layer().weight.double()
        assert (w1 - w2).norm() <= eps * (new_base.norm() + 2 * (new_base - base).norm())


@pytest.mark.parametrize("scalar_only", [False, True])
def test_bf16_adapters(scalar_only):
    model = make_model().to(torch.bfloat16)     # LoRA factors in bf16 as well
    train(model, LoraScaling(model, warmup_steps=2, scalar_only=scalar_only), steps=5)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert all_finite(model)


@pytest.mark.parametrize("scalar_only", [False, True])
def test_zero_init_without_warmup(scalar_only):
    model = make_model()
    with pytest.warns(UserWarning, match="all zeros"):
        scaling = LoraScaling(model, scalar_only=scalar_only, warmup_steps=0)
    lora_B = scaling.lora_weights[0][3]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    scaling.hook_to_optimizer(optimizer)
    snapshots = []
    for x, y in make_data(6):
        optimizer.zero_grad()
        loss_fn(model, x, y).backward()
        optimizer.step()
        snapshots.append(lora_B.detach().clone())
    assert all_finite(model)
    assert snapshots[0].abs().sum() > 0
    assert all(not torch.equal(a, b) for a, b in zip(snapshots, snapshots[1:]))    # B is not frozen


def test_unsupported_layers_raise():
    with pytest.raises(ValueError, match="DoRA"):
        LoraScaling(make_model(targets=("fc1",), use_dora=True))


def test_get_lora_weights_missing_adapter():
    model = make_model(targets=("fc1",))
    layer = model.base_model.model.fc1
    assert get_lora_weights(layer, "default") is not None
    assert get_lora_weights(layer, "other") is None


# ---------- optimizers ----------

@pytest.mark.parametrize("make_optimizer", [lambda p: torch.optim.AdamW(p, lr=1e-2, amsgrad=True),
                                            lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9)])
@pytest.mark.parametrize("scalar_only", [False, True])
def test_amsgrad_and_sgd(make_optimizer, scalar_only):
    model = make_model()
    scaling = LoraScaling(model, warmup_steps=2, scalar_only=scalar_only)
    optimizer = train(model, scaling, make_optimizer)
    assert all_finite(model)
    for _, _, A, B in scaling.lora_weights:
        for param in (A, B):
            state = optimizer.state[param]
            if "max_exp_avg_sq" in state:
                assert (state["max_exp_avg_sq"] >= state["exp_avg_sq"]).all()


def test_unsupported_optimizer_raises():
    model = make_model()
    with pytest.raises(RuntimeError, match="cannot rescale"):
        train(model, LoraScaling(model, warmup_steps=2), lambda p: torch.optim.Adagrad(p, lr=1e-2))


def test_hooking_twice_registers_one_hook():
    once, twice = make_model(), make_model()
    train(once, LoraScaling(once, warmup_steps=2))
    scaling = LoraScaling(twice, warmup_steps=2)
    optimizer = torch.optim.AdamW([p for p in twice.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.01)
    scaling.hook_to_optimizer(optimizer)
    train(twice, scaling, lambda p: optimizer)
    expected, actual = all_params(once), all_params(twice)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


@pytest.mark.parametrize("warmup", range(4))
@pytest.mark.parametrize("interval", range(1, 4))
def test_schedule_matches_reference(warmup, interval):
    ref, new = ReferenceLoraScaling.__new__(ReferenceLoraScaling), LoraScaling.__new__(LoraScaling)
    ref.interval, ref.skip_steps = interval, warmup
    new.interval, new.skip_steps = interval, warmup
    expected = []
    for _ in range(20):
        if ref.skip_steps > 0:      # same logic as the reference's scale_lora_adapters
            ref.skip_steps -= 1
            expected.append(True)
        else:
            ref.skip_steps = ref.interval - 1
            expected.append(False)
    assert [new._skip() for _ in range(20)] == expected
