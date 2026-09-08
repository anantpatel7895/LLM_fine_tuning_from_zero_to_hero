"""Phase 01 — the missing rung between your GPT-2 loop and TRL's SFTTrainer.

This is a training loop with the same *shape* as the one you hand-wrote for GPT-2 124M,
but running on a real tokenizer and a real pretrained model. It deliberately does NOT use
``Trainer`` or ``SFTTrainer``.

The point is that when you meet ``SFTTrainer`` in Phase 04, you will know exactly which
lines of this file it replaced -- and which of the fifteen small correctnesses it adds
that this file does not have.

Read the "WHAT THIS FILE DOES NOT DO" list at the bottom before Phase 04. That list *is*
the argument for using the library.

Runs on CPU or Apple MPS for a few steps on a 0.6B model, so you can do Phase 01 entirely
on the Mac::

    python -m llm_finetuning_from_zero_to_hero.train.train_min --steps 20
    python -m llm_finetuning_from_zero_to_hero.train.train_min --steps 20 --lora
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# A handful of instruction-style examples, inline, so this file has no data dependency.
# Phase 07 replaces this with a real, versioned, deduplicated dataset.
EXAMPLES: list[dict[str, str]] = [
    {"prompt": "What is gradient checkpointing?",
     "completion": "It trades compute for memory: activations are discarded during the "
                   "forward pass and recomputed during the backward pass."},
    {"prompt": "Why use bf16 instead of fp16?",
     "completion": "bf16 keeps fp32's exponent range, so gradients do not underflow and "
                   "no loss scaling is needed."},
    {"prompt": "What does LoRA rank control?",
     "completion": "The dimensionality of the low-rank update. Higher rank means more "
                   "trainable parameters and more capacity to change the model."},
    {"prompt": "What is a KV cache?",
     "completion": "Cached key and value tensors from previous tokens, so each new token "
                   "does not recompute attention over the whole prefix."},
    {"prompt": "Why does DDP work well for LoRA?",
     "completion": "Only the adapter's gradients are all-reduced, which is a few hundred "
                   "megabytes per step rather than the whole model."},
    {"prompt": "What is catastrophic forgetting?",
     "completion": "Losing capabilities the base model already had while learning a new "
                   "task. It must be measured, not assumed."},
    {"prompt": "What does a loss mask do?",
     "completion": "It sets label positions to -100 so they are ignored by the loss, "
                   "which is how you train on the answer but not the question."},
    {"prompt": "Why is the logits tensor sometimes the largest allocation?",
     "completion": "Because it is batch times sequence times vocabulary, upcast to fp32, "
                   "and vocabulary can be over 128,000."},
]


class TinyPromptCompletion(Dataset):
    """Tokenize prompt+completion, and mask the loss to the completion only.

    This is the simplest possible version of what Phase 03 does properly. Note what is
    happening on the ``labels`` line -- everything belonging to the prompt is set to -100
    so it contributes nothing to the loss. Get that wrong and the model learns to imitate
    the question rather than answer it.
    """

    def __init__(self, tokenizer, examples: list[dict[str, str]], max_length: int = 256):
        self.rows = []
        for ex in examples:
            # Deliberately NOT using apply_chat_template here -- that is Phase 02, and
            # doing it by hand first is the point. This is raw prompt/completion.
            prompt_ids = tokenizer(ex["prompt"] + "\n", add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]

            # The EOS token MUST be in the labels, or the model never learns to stop.
            # This is failure #2 in docs/DATA.md Part 3, and it is the single most common
            # "my fine-tune is broken" report. One line, easy to forget.
            completion_ids = completion_ids + [tokenizer.eos_token_id]

            input_ids = (prompt_ids + completion_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]

            self.rows.append({
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.rows[i]


def collate(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad to the longest sequence in the batch.

    RIGHT padding for training. LEFT padding for generation. Mixing them up is failure #6
    in docs/DATA.md -- generation starts inside the pad region and produces nonsense.
    """
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id)]))
        # Padding positions are masked out of the loss with -100, same mechanism as the
        # prompt masking above.
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100)]))
        attention_mask.append(
            torch.cat([torch.ones(len(b["input_ids"])), torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids).long(),
        "labels": torch.stack(labels).long(),
        "attention_mask": torch.stack(attention_mask).long(),
    }


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():          # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="full fine-tune LR. With --lora, use ~1e-4 (see Phase 04)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora", action="store_true",
                        help="wrap in a peft LoRA adapter instead of full fine-tuning")
    parser.add_argument("--inspect", action="store_true",
                        help="print the token/label/in-loss table for one example and exit")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        # Careful: setting pad_token = eos_token is a real trap when the collator masks by
        # token id rather than position -- it masks genuine EOS tokens out of the loss too
        # (failure #4 in docs/DATA.md). We mask by POSITION above, so this is safe here.
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TinyPromptCompletion(tokenizer, EXAMPLES, max_length=args.max_length)

    # ---- the inspection that Phase 03 turns into a real tool ----
    if args.inspect:
        row = dataset[0]
        print(f"\n{'token':<20} {'id':>8} {'label':>8}  in_loss")
        print("-" * 50)
        for tok_id, label in zip(row["input_ids"].tolist(), row["labels"].tolist()):
            token = tokenizer.decode([tok_id])
            in_loss = "YES" if label != -100 else "."
            print(f"{token!r:<20} {tok_id:>8} {label:>8}  {in_loss}")
        n_in_loss = int((row["labels"] != -100).sum())
        print(f"\n{n_in_loss} of {len(row['labels'])} positions contribute to the loss.")
        print("Confirm: the prompt is excluded, and the final EOS token IS included.")
        return 0

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    # ---- model ----
    # NOTE: `dtype=`, NOT `torch_dtype=`. Renamed in transformers v5.
    # See docs/notes/api-2026.md section 1.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False        # incompatible with gradient checkpointing, and
                                          # pointless during training
    model.to(device)

    if args.lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # All linear layers, not attention-only: 67% of the adaptable parameters live
            # in the MLP. See NUMBERS.md section 5.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # If that printed 0.00%, target_modules matched nothing -- and the loss would
        # still descend, teaching you nothing. Phase 12 failure #5.

    model.train()

    # ---- optimizer and schedule ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"trainable parameters: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # Warmup then cosine decay. Your GPT-2 loop probably had a constant LR; warmup is what
    # protects the first few steps, when gradient estimates are worst, from wrecking the
    # pretrained weights.
    n_optimizer_steps = max(1, args.steps // args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.03 * n_optimizer_steps)),
        num_training_steps=n_optimizer_steps,
    )

    # ---- the loop ----
    print(f"\n{'step':>5} {'loss':>8} {'ppl':>9} {'lr':>10} {'grad_norm':>10} {'s/step':>8}")
    print("-" * 60)

    step = 0
    batches = iter(loader)
    t0 = time.perf_counter()

    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # Dividing by grad_accum is the loss normalization that Trainer handles for
            # you. Omit it and your loss value silently depends on grad_accum --
            # Phase 12 failure #10.
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accumulated_loss += loss.item()

        # Gradient clipping: your defence against the loss spikes in Phase 12 failure #1.
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

        optimizer.step()
        scheduler.step()
        step += 1

        elapsed = time.perf_counter() - t0
        ppl = math.exp(min(accumulated_loss, 20))     # clamp: exp overflows on a bad start
        print(f"{step:>5} {accumulated_loss:>8.4f} {ppl:>9.2f} "
              f"{scheduler.get_last_lr()[0]:>10.2e} {grad_norm:>10.3f} "
              f"{elapsed / step:>8.2f}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\npeak VRAM: {peak:.2f} GiB")
        print("Compare against your prediction from NUMBERS.md sections 1-3, and record "
              "the gap in the phase doc.")

    print("\n" + "=" * 60)
    print("WHAT THIS FILE DOES NOT DO -- and why Phase 04 uses the library instead:")
    print("=" * 60)
    for item in [
        "chat templates (this uses raw prompt+completion)      -> Phase 02",
        "multi-turn loss masking                               -> Phase 03",
        "sequence packing, so short examples waste compute     -> Phase 10",
        "gradient checkpointing, so it OOMs on a real model     -> NUMBERS.md section 2",
        "chunked cross-entropy, so the logits tensor is huge   -> NUMBERS.md section 3",
        "checkpointing / resume (optimizer + RNG + data order) -> Phase 12 failure #11",
        "multi-GPU                                             -> Phase 10",
        "eval, metrics, logging, tracking                      -> Phase 05",
        "quantization (this is full precision)                 -> Phase 04",
    ]:
        print(f"  - {item}")
    print("\nThat list is the argument for using TRL. Not that it is conceptually")
    print("different, but that it does ~15 small things correctly that are each easy")
    print("to get subtly wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
