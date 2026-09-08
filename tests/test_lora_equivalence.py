"""Phase 01 — prove your hand-written LoRA computes the same function as peft's.

This is the phase's credential. It demonstrates you know the library rather than trust
it, and it is a genuinely useful thing to be able to say in an interview.

Two properties are tested:

1. **Forward equivalence** — given identical A and B weights, your LoRA layer and peft's
   produce the same output to float tolerance.
2. **Zero-init identity** — a freshly wrapped peft model is *exactly* the base model,
   because ``lora_B`` starts at zero. This is why step-0 loss must equal base-model loss,
   which is a check worth running on every real training run.

Runs on CPU. No GPU needed::

    pytest tests/test_lora_equivalence.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------------------
# Your hand-written LoRA, transcribed from
# /Users/in04844/Anant_Learning/learning_torch/04_learning_lora_from_scratch.ipynb
#
# If your notebook's version differs, EDIT THIS to match yours -- the whole point of the
# test is to check *your* implementation against the library, not a copy of the library.
# ---------------------------------------------------------------------------------------
class LinearWithLoRA(nn.Module):
    """A frozen linear layer plus a trainable low-rank update.

    out = W @ x  +  (alpha / r) * (B @ A @ x)
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # A: (r, in_features).  B: (out_features, r).
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

        # A gets Kaiming init, B stays zero -- so the product is zero at step 0.
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------
IN_FEATURES, OUT_FEATURES, RANK, ALPHA = 64, 32, 8, 16


@pytest.fixture
def base_linear() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)


def test_zero_init_is_identity(base_linear: nn.Linear) -> None:
    """With lora_B == 0, the adapter must be an exact no-op.

    This is the property that lets you write the single most useful sanity check for any
    real training run: step-0 loss must equal base-model loss. If it does not, your
    adapter is not initialized the way you think it is.
    """
    torch.manual_seed(1)
    x = torch.randn(4, IN_FEATURES)

    wrapped = LinearWithLoRA(base_linear, r=RANK, alpha=ALPHA)

    # Exact equality, not approximate: adding scaling * (x @ A.T @ 0) adds literal zeros.
    assert torch.equal(wrapped(x), base_linear(x)), (
        "A freshly-initialized LoRA layer is not identical to its base layer. "
        "lora_B must be zero-initialized."
    )


def test_forward_matches_peft(base_linear: nn.Linear) -> None:
    """Given identical A and B, your implementation and peft's compute the same function."""
    peft = pytest.importorskip("peft", reason="peft not installed")
    from peft.tuners.lora import LoraLayer  # noqa: F401  (kept for the reader's benefit)

    torch.manual_seed(2)
    x = torch.randn(4, IN_FEATURES)

    # --- your version ---
    mine = LinearWithLoRA(base_linear, r=RANK, alpha=ALPHA)

    # --- peft's version, wrapping a copy of the same base weights ---
    base_copy = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    base_copy.weight.data.copy_(base_linear.weight.data)

    model = nn.Sequential()
    model.add_module("target", base_copy)

    config = peft.LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.0,          # dropout must be off or the comparison is stochastic
        bias="none",
        target_modules=["target"],
    )
    peft_model = peft.get_peft_model(model, config)
    peft_model.eval()              # belt and braces: no dropout, no train-mode behaviour

    # Copy YOUR A and B into peft's layer so both compute the same function.
    peft_layer = peft_model.base_model.model.target
    with torch.no_grad():
        peft_layer.lora_A["default"].weight.copy_(mine.lora_A)
        peft_layer.lora_B["default"].weight.copy_(mine.lora_B)
        # Give B a non-zero value, or this test passes trivially via the zero-init path.
        nonzero_b = torch.randn_like(mine.lora_B) * 0.02
        mine.lora_B.copy_(nonzero_b)
        peft_layer.lora_B["default"].weight.copy_(nonzero_b)

    mine_out = mine(x)
    peft_out = peft_model(x)

    torch.testing.assert_close(
        mine_out, peft_out, rtol=1e-4, atol=1e-5,
        msg=lambda s: (
            f"{s}\n\nYour LoRA and peft's disagree. Most likely causes:\n"
            f"  - a different scaling convention (alpha/r vs alpha/sqrt(r) -- see rsLoRA)\n"
            f"  - A and B transposed relative to peft's layout\n"
            f"  - dropout still active on one side"
        ),
    )


def test_scaling_convention(base_linear: nn.Linear) -> None:
    """The alpha/r scaling is a convention, not a law -- and rsLoRA uses alpha/sqrt(r).

    Phase 06 ablates this. The test just pins down what the current implementation does,
    so a later change to it is visible rather than silent.
    """
    layer = LinearWithLoRA(base_linear, r=8, alpha=16)
    assert layer.scaling == pytest.approx(2.0), "expected alpha/r = 16/8 = 2.0"

    rslora_scaling = 16 / (8**0.5)
    assert rslora_scaling == pytest.approx(5.657, abs=1e-3), (
        "rsLoRA would scale by alpha/sqrt(r) instead -- nearly 3x larger at r=8, and the "
        "gap widens with rank. This is why high-rank LoRA underperforms with alpha/r."
    )


def test_trainable_parameter_count(base_linear: nn.Linear) -> None:
    """Check the NUMBERS.md section 5 formula: r * (d_in + d_out) per module."""
    layer = LinearWithLoRA(base_linear, r=RANK, alpha=ALPHA)
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)

    expected = RANK * (IN_FEATURES + OUT_FEATURES)
    assert trainable == expected, (
        f"expected r*(d_in+d_out) = {RANK}*({IN_FEATURES}+{OUT_FEATURES}) = {expected}, "
        f"got {trainable}"
    )

    # And the base weights must be frozen -- otherwise this is not LoRA, it is a full
    # fine-tune with extra steps.
    assert not any(p.requires_grad for p in layer.base.parameters()), (
        "base layer weights are still trainable"
    )
