import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch
from data import TaskConfig, make_splits, to_torch
from model import load_model
from train import evaluate


def run_head_ablation_suite(
    gen_paths, output_fig="results/figures/ablation_impact.png"
):
    """Evaluates impact of knocking out (L0, H0) and (L1, H0) across generations."""
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    ablation_results = []

    for gen_name, path in gen_paths.items():
        model, extra = load_model(path)
        task_cfg = TaskConfig(**extra["task_config"])
        splits = make_splits(
            task_cfg, n_train=100, n_val=1000, n_test=2000, seed=0
        )
        test_t = to_torch(splits["test"])

        # 1 Baseline acc w/ no ablation
        base_eval = evaluate(model, task_cfg, test_t, ablate=())
        clean_acc = base_eval["acc"]

        # Ablate Previous Token Head (Layer 0, Head 0)
        l0_eval = evaluate(model, task_cfg, test_t, ablate={(0, 0)})
        l0_acc = l0_eval["acc"]

        # Ablate Induction Head (Layer 1, Head 0)
        l1_eval = evaluate(model, task_cfg, test_t, ablate={(1, 0)})
        l1_acc = l1_eval["acc"]

        row = {
            "generation": gen_name,
            "clean_acc": clean_acc,
            "ablate_L0_prev_token": l0_acc,
            "ablate_L1_induction": l1_acc,
        }
        ablation_results.append(row)
        print(
            f"[{gen_name}] Clean: {clean_acc:.4f} | Ablate L0: {l0_acc:.4f} | Ablate L1: {l1_acc:.4f}"
        )

    #result plotting
    gens = [int(r["generation"]) for r in ablation_results]
    plt.figure(figsize=(8, 4))
    plt.plot(
        gens,
        [r["clean_acc"] for r in ablation_results],
        "o-",
        label="Clean Model",
        color="#2a78d6",
    )
    plt.plot(
        gens,
        [r["ablate_L0_prev_token"] for r in ablation_results],
        "s--",
        label="Ablate L0 (Prev Token)",
        color="#eb6834",
    )
    plt.plot(
        gens,
        [r["ablate_L1_induction"] for r in ablation_results],
        "^--",
        label="Ablate L1 (Induction)",
        color="#3a3a38",
    )

    plt.title("Impact of Head Knockouts Across Collapse Generations")
    plt.xlabel("Generation")
    plt.ylabel("Test Accuracy")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    plt.close()

    with open("results/ablation_summary.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    print(f"Saved ablation curves to {output_fig}")


if __name__ == "__main__":
    # get all available models from Gen 0 to Gen 10
    runs = {"0": "results/base_single/model.pt"}
    for g in range(1, 11):
        p = f"results/collapse/base/gen_{g}/model.pt"
        if os.path.exists(p):
            runs[str(g)] = p

    run_head_ablation_suite(runs)