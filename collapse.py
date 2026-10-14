"""
Recursive model-collapse experiments for the induction-head project.


Main idea
---------
Generation 0 is a normally trained model. It produces synthetic targets for a
fresh set of task examples. A new model is trained on those synthetic targets,
then that model produces the next generation's data, and so on.

For the base task the model generates only the answer label.
For the extended task the model generates:
    answer -> next context symbol -> that symbol's answer

The next symbol is constrained to be one of the symbols already present in the
context, matching the structure of the real task.

Across generations we always evaluate on the SAME real validation/test
splits. This prevents a collapsing model from looking good merely because its
evaluation data collapsed in the same way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict

import numpy as np
import torch

from data import TaskConfig, make_loss_mask, make_splits, to_torch
from model import load_model, save_model, warm_start
from train import TrainConfig, evaluate, train_model


# ---------------------------------------------------------------------------
# Sampling helpers

@torch.no_grad()
def _pick_from_candidates(logits, candidate_ids, temperature=1.0, greedy=False):
    """Pick one token per row from a row-specific candidate set.

    logits:        [B, vocab]
    candidate_ids: [B, K] token ids that are valid for each row

    Returns [B, 1] token ids.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    cand_logits = logits.gather(1, candidate_ids)
    if greedy:
        choice = cand_logits.argmax(dim=1, keepdim=True)
    else:
        probs = torch.softmax(cand_logits / temperature, dim=1)
        choice = torch.multinomial(probs, 1)
    return candidate_ids.gather(1, choice)


@torch.no_grad()
def _pick_label(model, seq, cfg, temperature=1.0, greedy=False):
    """Generate one label token, constrained to the label vocabulary."""
    logits = model(seq)[:, -1, :]
    B = seq.shape[0]
    labels = torch.arange(
        cfg.n_symbols,
        cfg.n_symbols + cfg.n_labels,
        device=seq.device,
        dtype=torch.long,
    ).expand(B, -1)
    return _pick_from_candidates(logits, labels, temperature, greedy)


# ---------------------------------------------------------------------------
# Synthetic data generation

@torch.no_grad()
def generate_synthetic_dataset(
    model,
    cfg: TaskConfig,
    real_data: dict,
    device="cpu",
    batch_size=4096,
    temperature=1.0,
    greedy=False,
    seed=0,
):
    """Generate synthetic scored tokens while keeping the real task context.

    Base task (n_queries=1):
        real prefix:      context | q
        model generates:  a

    Extended task (n_queries=2):
        real prefix:      context | q
        model generates:  a, q2, a2
        q2 is constrained to one of the context symbols.

    Returns a data.py-style dict with ``tokens`` and ``loss_mask``.
    """
    cfg.check()
    model = model.to(device).eval()
    torch.manual_seed(seed)

    real_tokens = real_data["tokens"]
    n = len(real_tokens)
    P = cfg.n_pairs
    prefix_len = 2 * P + 1  # context plus first query q

    out = np.empty((n, cfg.seq_len), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prefix_np = real_tokens[start:end, :prefix_len]
        seq = torch.as_tensor(prefix_np, dtype=torch.long, device=device)

        # 1) Generate the answer to the first query.
        a = _pick_label(model, seq, cfg, temperature, greedy)
        seq = torch.cat([seq, a], dim=1)

        if cfg.n_queries == 2:
            # 2) Generate q2, but only from symbols already in this context.
            #    Context symbols live at positions 0,2,4,...,2P-2.
            ctx_syms = seq[:, 0:2 * P:2]
            logits = model(seq)[:, -1, :]
            q2 = _pick_from_candidates(logits, ctx_syms, temperature, greedy)
            seq = torch.cat([seq, q2], dim=1)

            # 3) Generate q2's label.
            a2 = _pick_label(model, seq, cfg, temperature, greedy)
            seq = torch.cat([seq, a2], dim=1)

        out[start:end] = seq.cpu().numpy()

    return {"tokens": out, "loss_mask": make_loss_mask(cfg, n)}


# ---------------------------------------------------------------------------
# Collapse diagnostics


def _entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def synthetic_metrics(cfg: TaskConfig, real_data: dict, synth_data: dict):
    """Measure how the synthetic targets differ from the real task structure."""
    real = real_data["tokens"]
    syn = synth_data["tokens"]
    P = cfg.n_pairs
    first_answer_pos = 2 * P + 1

    metrics = {
        # Agreement with the real first-query answer.
        "synthetic_first_answer_acc": float(
            np.mean(syn[:, first_answer_pos] == real[:, first_answer_pos])
        )
    }

    if cfg.n_queries == 2:
        q2_pos = 2 * P + 2
        a2_pos = 2 * P + 3

        ctx_syms = syn[:, 0:2 * P:2]            # [N, P]
        ctx_labs = syn[:, 1:2 * P:2]            # [N, P]
        q2 = syn[:, q2_pos]                      # [N]
        a2 = syn[:, a2_pos]                      # [N]

        # Which context slot did the model choose for q2?
        slot_match = (ctx_syms == q2[:, None])
        valid_q2 = slot_match.any(axis=1)
        slot = slot_match.argmax(axis=1)
        slot_counts = np.bincount(slot[valid_q2], minlength=P)
        slot_entropy = _entropy_from_counts(slot_counts)
        max_slot_entropy = math.log(P) if P > 1 else 0.0

        # Correct label for the generated q2, read directly from the context.
        row = np.arange(len(syn))
        correct_a2 = ctx_labs[row, slot]
        a2_acc = np.mean((a2 == correct_a2)[valid_q2]) if valid_q2.any() else 0.0

        # Identity-level q2 distribution over all symbols.
        symbol_counts = np.bincount(q2[valid_q2], minlength=cfg.n_symbols)
        symbol_entropy = _entropy_from_counts(symbol_counts)
        max_symbol_entropy = math.log(cfg.n_symbols) if cfg.n_symbols > 1 else 0.0

        metrics.update({
            "synthetic_q2_valid_frac": float(valid_q2.mean()),
            "synthetic_next_answer_acc": float(a2_acc),
            "q2_slot_counts": slot_counts.tolist(),
            "q2_slot_probs": (slot_counts / max(1, slot_counts.sum())).tolist(),
            "q2_slot_entropy": float(slot_entropy),
            "q2_slot_entropy_norm": float(slot_entropy / max_slot_entropy) if max_slot_entropy else 1.0,
            "q2_symbol_entropy": float(symbol_entropy),
            "q2_symbol_entropy_norm": float(symbol_entropy / max_symbol_entropy) if max_symbol_entropy else 1.0,
            "q2_unique_symbols": int(np.count_nonzero(symbol_counts)),
            "q2_top_symbol_share": float(symbol_counts.max() / max(1, symbol_counts.sum())),
        })

    return metrics


def mix_real_and_synthetic(cfg, real_data, synth_data, real_fraction, seed):
    """Optionally preserve some real rows while replacing the rest by synthetic rows."""
    if not (0.0 <= real_fraction <= 1.0):
        raise ValueError("real_fraction must be between 0 and 1")
    if real_fraction == 0.0:
        return synth_data
    if real_fraction == 1.0:
        return real_data

    rng = np.random.default_rng(seed)
    use_real = rng.random(len(real_data["tokens"])) < real_fraction
    tokens = synth_data["tokens"].copy()
    tokens[use_real] = real_data["tokens"][use_real]
    return {
        "tokens": tokens,
        "loss_mask": make_loss_mask(cfg, len(tokens)),
    }


# ---------------------------------------------------------------------------
# Main collapse loop


def task_config_from_checkpoint(extra):
    if "task_config" not in extra:
        raise ValueError(
            "Checkpoint does not contain task_config metadata. "
            "Use a model saved by train.py."
        )
    return TaskConfig(**extra["task_config"])


def run_collapse(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device: {device}")

    gen0, extra = load_model(args.init_from, device=device)

    # Task config is needed for BOTH base and extended tasks.
    cfg = task_config_from_checkpoint(extra)

    base_model = None

    # Extended generations must be warm-started from the trained base model.
    if args.task == "extended":
        if args.base_init_from is None:
            raise ValueError(
                "Extended collapse requires --base_init_from "
                "so every generation can be warm-started from the base model."
            )

        base_model, _ = load_model(args.base_init_from, device=device)

    expected_queries = 1 if args.task == "base" else 2
    if cfg.n_queries != expected_queries:
        raise ValueError(
            f"Checkpoint has n_queries={cfg.n_queries}, but --task {args.task} "
            f"expects n_queries={expected_queries}."
        )

    print("task config:", asdict(cfg))
    print(f"loading real splits: train={args.n_train}, val={args.n_val}, test={args.n_test}")
    splits = make_splits(
        cfg,
        args.n_train,
        args.n_val,
        args.n_test,
        seed=args.data_seed,
    )
    real_train = splits["train"]
    real_val_t = to_torch(splits["val"])
    real_test_t = to_torch(splits["test"])

    os.makedirs(args.out_dir, exist_ok=True)

    # Hyperparameters for every newly trained generation.
    tcfg = TrainConfig(
        lr=args.lr,
        batch_size=args.train_batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        seed=args.train_seed,
    )

    # Keep architecture the same as generation 0.
    model_kwargs = {
        "d_model": gen0.cfg.d_model,
        "n_layers": gen0.cfg.n_layers,
        "n_heads": gen0.cfg.n_heads,
        "use_mlp": gen0.cfg.use_mlp,
        "use_layernorm": gen0.cfg.use_layernorm,
        "d_mlp": gen0.cfg.d_mlp,
    }

    summary = []

    # Generation 0: evaluate the supplied real-data model.
    gen0_test = evaluate(gen0, cfg, real_test_t)
    row0 = {
        "generation": 0,
        "source": "real",
        **{f"test_{k}": v for k, v in gen0_test.items()},
    }
    summary.append(row0)
    print("gen 0 test:", {k: round(v, 4) for k, v in gen0_test.items()})

    prev_model = gen0

    for generation in range(1, args.generations + 1):
        print(f"\n{'=' * 70}\nGENERATION {generation}\n{'=' * 70}")

        synth = generate_synthetic_dataset(
            prev_model,
            cfg,
            real_train,
            device=device,
            batch_size=args.gen_batch_size,
            temperature=args.temperature,
            greedy=args.greedy,
            seed=args.synthetic_seed + generation,
        )
        sm = synthetic_metrics(cfg, real_train, synth)

        # Constant real/synthetic mixture. 0.0 = fully synthetic training data.
        train_data = mix_real_and_synthetic(
            cfg,
            real_train,
            synth,
            args.real_fraction,
            args.synthetic_seed + 10_000 + generation,
        )

        print(
            f"synthetic first-answer acc: {sm['synthetic_first_answer_acc']:.4f}"
        )
        if cfg.n_queries == 2:
            print(
                "q2 slot probs:",
                [round(x, 3) for x in sm["q2_slot_probs"]],
                "| q2 entropy(norm):",
                round(sm["q2_slot_entropy_norm"], 3),
                "| next-answer acc:",
                round(sm["synthetic_next_answer_acc"], 4),
            )

        train_t = to_torch(train_data)

        # Base task: fresh model each generation.
        # Extended task: warm-start every generation from the SAME
        # trained base-task checkpoint.
        if args.task == "extended":
            init_model = warm_start(base_model, cfg)
        else:
            init_model = None

        model, history = train_model(
            cfg,
            train_t,
            real_val_t,
            model_kwargs=model_kwargs,
            tcfg=tcfg,
            device=device,
            verbose=True,
            model=init_model,
        )

        test = evaluate(model, cfg, real_test_t)
        print("real test:", {k: round(v, 4) for k, v in test.items()})

        gen_dir = os.path.join(args.out_dir, f"gen_{generation}")
        os.makedirs(gen_dir, exist_ok=True)
        save_model(
            model,
            os.path.join(gen_dir, "model.pt"),
            extra={
                "task_config": asdict(cfg),
                "train_config": asdict(tcfg),
                "collapse_generation": generation,
                "real_fraction": args.real_fraction,
                "temperature": args.temperature,
                "greedy": args.greedy,
            },
        )
        with open(os.path.join(gen_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        row = {
            "generation": generation,
            "source": "synthetic",
            "real_fraction": args.real_fraction,
            **sm,
            **{f"test_{k}": v for k, v in test.items()},
        }
        summary.append(row)

        with open(os.path.join(args.out_dir, "collapse_metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)

        prev_model = model

    print(f"\nsaved collapse run to: {args.out_dir}")
    return summary


# ---------------------------------------------------------------------------
# CLI


def main():
    p = argparse.ArgumentParser(description="Run recursive model-collapse experiments")
    p.add_argument("--task", choices=["base", "extended"], required=True)
    p.add_argument("--init_from", required=True, help="Generation-0 checkpoint from train.py")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--generations", type=int, default=5)

    # Recreate the same real-data distribution used by train.py.
    p.add_argument("--n_train", type=int, default=400_000)
    p.add_argument("--n_val", type=int, default=2_000)
    p.add_argument("--n_test", type=int, default=5_000)
    p.add_argument("--data_seed", type=int, default=0)

    # Synthetic generation.
    p.add_argument("--gen_batch_size", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--synthetic_seed", type=int, default=1234)
    p.add_argument(
        "--real_fraction",
        type=float,
        default=0.0,
        help="fraction of real rows retained in each later generation (default: 0)",
    )

    # Training each new generation.
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
    "--base_init_from",
    default=None,
    help="Base-task checkpoint used to warm-start each extended generation",
)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("results", "collapse", args.task)

    run_collapse(args)


if __name__ == "__main__":
    main()
