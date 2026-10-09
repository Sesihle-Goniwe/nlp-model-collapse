"""
Train and tune the induction-head model.

What this file gives the team:
  - train_model(...)  trains one model on any data (real or synthetic) and
                      logs loss, accuracy, perplexity and two circuit scores.
  - evaluate(...)     scores a model on one split.
  - sweep(...)        tries a few settings, picks the best by VALIDATION loss,
                      and only then scores it on the test split.

Circuit scores (measured on the validation split during training):
  prev_token  how much layer 1 looks exactly one step back (0 to 1).
  induction   how much the last layer, at the query, looks at the label that
              follows the matching symbol (0 to 1).
These show WHEN the circuit forms, not just when accuracy goes up.

Examples:
  python train.py                                  # one run, base task
  python train.py --n_queries 2                    # one run, extended task
  python train.py --mode sweep                     # small tuning sweep
  python train.py --n_queries 2 --init_from results/base_single/model.pt
                                                   # extended task, started from the base model
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from data import TaskConfig, make_splits, to_torch
from model import build_model, masked_loss_and_acc, save_model, load_model, warm_start


@dataclass
class TrainConfig:
    lr: float = 3e-3
    batch_size: int = 128
    max_steps: int = 4000
    eval_every: int = 250
    weight_decay: float = 0.0
    seed: int = 0                  # model init and batch order
    stop_at_val_acc: float = None  # e.g. 0.995 to stop early; None = fixed budget


# ---------------------------------------------------------------- evaluation

POS_NAMES = {1: ["answer"], 2: ["answer", "next_symbol", "next_answer"]}


def circuit_scores(task_cfg, x, cache):
    """
    prev_token: layer-1 attention from each context LABEL to the symbol just
                before it (the edge the induction circuit needs).
    induction:  last-layer attention from the query to the label right after
                the matching symbol in the context.
    With several heads, the best head is reported.

    Note: high accuracy with a low induction score means the model found a
    different way (a shortcut) to solve the task. Always check both.
    """
    P = task_cfg.n_pairs
    a_first = cache["attn"][0]                     # [B, H, T, T]
    a_last = cache["attn"][-1]
    t = torch.arange(1, 2 * P, 2, device=x.device)             # label positions
    prev = a_first[:, :, t, t - 1].mean(dim=(0, 2))            # [H]

    q_pos = 2 * P
    query = x[:, q_pos]                                        # [B]
    ctx_syms = x[:, 0:2 * P:2]                                 # [B, P]
    match = (ctx_syms == query[:, None]).float().argmax(dim=1) # which pair
    label_pos = 2 * match + 1                                  # [B]
    b = torch.arange(x.shape[0], device=x.device)
    ind = a_last[b, :, q_pos, label_pos].mean(dim=0)           # [H]
    return prev.max().item(), ind.max().item()


@torch.no_grad()
def evaluate(model, task_cfg, split, chunk=4096, ablate=()):
    """split = (inputs, targets, loss_mask) tensors from data.to_torch."""
    model.eval()
    x, y, m = split
    tot_loss = tot_n = 0.0
    correct_pos = torch.zeros(m.shape[1], device=x.device)
    count_pos = torch.zeros(m.shape[1], device=x.device)
    prev_s = ind_s = 0.0
    for i in range(0, len(x), chunk):
        xb, yb, mb = x[i:i + chunk], y[i:i + chunk], m[i:i + chunk]
        logits, cache = model(xb, return_cache=True, ablate=ablate)
        loss, _, _ = masked_loss_and_acc(logits, yb, mb)
        n = mb.sum().item()
        tot_loss += loss.item() * n
        tot_n += n
        correct_pos += ((logits.argmax(-1) == yb) & mb).sum(0)
        count_pos += mb.sum(0)
        p, s = circuit_scores(task_cfg, xb, cache)
        prev_s += p * len(xb)
        ind_s += s * len(xb)
    keep = count_pos > 0
    per_pos = (correct_pos[keep] / count_pos[keep]).tolist()
    loss = tot_loss / tot_n
    out = {"loss": loss, "ppl": math.exp(loss),
           "acc": correct_pos.sum().item() / count_pos.sum().item(),
           "prev_token": prev_s / len(x), "induction": ind_s / len(x)}
    for name, a in zip(POS_NAMES[task_cfg.n_queries], per_pos):
        out[f"acc_{name}"] = a
    model.train()
    return out


# ---------------------------------------------------------------- training

def train_model(task_cfg, train, val, model_kwargs=None, tcfg=None,
                device="cpu", verbose=True, model=None):
    """
    Train one model.

    train, val: (inputs, targets, loss_mask) tensors. 'train' can be real
                or synthetic data - this is the hook for the collapse loop.
    model:      pass an existing model to keep training it; otherwise a new one.
    Returns (model, history). history is a list of dicts, one per evaluation.
    """
    tcfg = tcfg or TrainConfig()
    model_kwargs = model_kwargs or {}
    torch.manual_seed(tcfg.seed)
    if model is None:
        model = build_model(task_cfg, **model_kwargs)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    gen = torch.Generator().manual_seed(tcfg.seed)

    xt, yt, mt = (t.to(device) for t in train)
    val = tuple(t.to(device) for t in val)
    history, t0 = [], time.time()

    def log(step, tr_loss, tr_acc):
        row = {"step": step, "train_loss": tr_loss, "train_acc": tr_acc,
               "time_s": round(time.time() - t0, 1)}
        row.update({f"val_{k}": v for k, v in evaluate(model, task_cfg, val).items()})
        history.append(row)
        if verbose:
            print(f"step {step:6d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} | "
                  f"prev-token {row['val_prev_token']:.2f} "
                  f"induction {row['val_induction']:.2f} | {row['time_s']:.0f}s",
                  flush=True)
        return row

    # train metrics are averaged over the steps since the last evaluation
    run_loss = run_acc = 0.0
    run_n = 0
    log(0, float("nan"), float("nan"))
    for step in range(1, tcfg.max_steps + 1):
        idx = torch.randint(0, len(xt), (tcfg.batch_size,), generator=gen).to(device)
        logits = model(xt[idx])
        loss, acc, _ = masked_loss_and_acc(logits, yt[idx], mt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss += loss.item()
        run_acc += acc.item()
        run_n += 1

        if step % tcfg.eval_every == 0 or step == tcfg.max_steps:
            row = log(step, run_loss / run_n, run_acc / run_n)
            run_loss = run_acc = 0.0
            run_n = 0
            if tcfg.stop_at_val_acc and row["val_acc"] >= tcfg.stop_at_val_acc:
                if verbose:
                    print(f"stopping: val acc reached {tcfg.stop_at_val_acc}")
                break
    return model, history


# ---------------------------------------------------------------- plotting

C1, C2, INK, GRID = "#2a78d6", "#eb6834", "#3a3a38", "#e4e3dc"


def plot_history(history, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = [r for r in history if r["step"] > 0]
    steps = [r["step"] for r in h]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    panels = [
        ("Loss", [("train", "train_loss", C1), ("validation", "val_loss", C2)]),
        ("Accuracy", [("train", "train_acc", C1), ("validation", "val_acc", C2)]),
        ("Circuit scores (validation)",
         [("prev-token head", "val_prev_token", C1), ("induction head", "val_induction", C2)]),
    ]
    for ax, (name, series) in zip(axes, panels):
        for label, key, col in series:
            ax.plot(steps, [r[key] for r in h], color=col, lw=2, label=label)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training step", color=INK)
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=9)
        ax.legend(frameon=False, fontsize=9)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    if title:
        fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- tuning

def sweep(task_cfg, splits_t, grid, base_tcfg, device="cpu", out_dir=None):
    """
    Try every setting in grid (a list of dicts). Each dict can hold TrainConfig
    fields (lr, batch_size, ...) and model fields (d_model, n_heads, ...).
    The winner is the lowest final VALIDATION loss. The test split is used
    once, for the winner only.
    """
    model_fields = {"d_model", "n_layers", "n_heads", "use_mlp", "use_layernorm", "d_mlp"}
    results, best = [], None
    for i, setting in enumerate(grid):
        mk = {k: v for k, v in setting.items() if k in model_fields}
        tk = {k: v for k, v in setting.items() if k not in model_fields}
        tcfg = TrainConfig(**{**asdict(base_tcfg), **tk})
        print(f"\n=== run {i + 1}/{len(grid)}: {setting}")
        model, hist = train_model(task_cfg, splits_t["train"], splits_t["val"],
                                  mk, tcfg, device, verbose=False)
        last = hist[-1]
        jump = next((r["step"] for r in hist if r["val_acc"] >= 0.9), None)
        row = {**setting, "final_val_loss": last["val_loss"],
               "final_val_acc": last["val_acc"], "step_val_acc_0.9": jump,
               "prev_token": last["val_prev_token"], "induction": last["val_induction"]}
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
        results.append(row)
        if best is None or last["val_loss"] < best[0]:
            best = (last["val_loss"], setting, model, hist)

    _, best_setting, best_model, best_hist = best
    test = evaluate(best_model, task_cfg, splits_t["test"])
    print(f"\nbest setting (by validation loss): {best_setting}")
    print("test:", {k: round(v, 4) for k, v in test.items()})
    if out_dir:
        with open(os.path.join(out_dir, "sweep.json"), "w") as f:
            json.dump({"results": results, "best": best_setting, "test": test}, f, indent=2)
    return best_model, best_setting, best_hist, test, results


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "sweep"], default="single")
    p.add_argument("--n_queries", type=int, default=1, help="1 = base, 2 = extended")
    p.add_argument("--n_symbols", type=int, default=1600)
    p.add_argument("--n_labels", type=int, default=32)
    p.add_argument("--n_pairs", type=int, default=4)
    p.add_argument("--zipf_alpha", type=float, default=0.0)
    p.add_argument("--n_train", type=int, default=400_000)
    p.add_argument("--n_val", type=int, default=2_000)
    p.add_argument("--n_test", type=int, default=5_000)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--init_from", default=None,
                   help="start from a saved model (e.g. the base model for the extended task)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_cfg = TaskConfig(n_symbols=args.n_symbols, n_labels=args.n_labels,
                          n_pairs=args.n_pairs, n_queries=args.n_queries,
                          zipf_alpha=args.zipf_alpha)
    name = "base" if args.n_queries == 1 else "extended"
    out_dir = args.out_dir or os.path.join("results", f"{name}_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)

    print(f"making data ({args.n_train} train sequences)...")
    splits = make_splits(task_cfg, args.n_train, args.n_val, args.n_test, seed=args.data_seed)
    splits_t = {k: to_torch(splits[k]) for k in ("train", "val", "test")}
    tcfg = TrainConfig(lr=args.lr, batch_size=args.batch_size, max_steps=args.max_steps,
                       eval_every=args.eval_every, seed=args.seed)

    if args.mode == "single":
        init = None
        if args.init_from:
            src, _ = load_model(args.init_from)
            init = warm_start(src, task_cfg)
            print(f"warm start from {args.init_from}")
        model, hist = train_model(task_cfg, splits_t["train"], splits_t["val"],
                                  {"d_model": args.d_model}, tcfg, device, model=init)
        # The test split is only for the final model. Don't tune on it.
        test = evaluate(model, task_cfg, splits_t["test"])
        print("test:", {k: round(v, 4) for k, v in test.items()})
    else:
        grid = [{"lr": lr, "d_model": d} for lr in (1e-3, 3e-3, 1e-2) for d in (32, 64)]
        model, best, hist, test, _ = sweep(task_cfg, splits_t, grid, tcfg, device, out_dir)

    save_model(model, os.path.join(out_dir, "model.pt"),
               extra={"task_config": asdict(task_cfg), "train_config": asdict(tcfg)})
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"task_config": asdict(task_cfg), "train_config": asdict(tcfg),
                   "history": hist, "test": test}, f, indent=2)
    fig_path = os.path.join("results", "figures", f"{os.path.basename(out_dir.rstrip('/'))}_curves.png")
    plot_history(hist, fig_path, title=f"{name.capitalize()} task: training curves")
    print(f"saved: {out_dir}/ and {fig_path}")


if __name__ == "__main__":
    main()
