import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch
from data import TaskConfig, make_splits, to_torch
from model import load_model


def plot_attention_across_generations(
    gen_paths, output_fig="results/figures/attention_collapse.png"
):
    """Generates attention maps comparing Layer 0 and Layer 1 across collapse generations."""
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    n_gens = len(gen_paths)
    fig, axes = plt.subplots(n_gens, 2, figsize=(8, 3 * n_gens))

    if n_gens == 1:
        axes = [axes]

    for idx, (gen_name, path) in enumerate(gen_paths.items()):
        model, extra = load_model(path)
        task_cfg = TaskConfig(**extra["task_config"])
        splits = make_splits(task_cfg, n_train=10, n_val=10, n_test=10, seed=0)
        x, _, _ = to_torch(splits["val"])

        model.eval()
        with torch.no_grad():
            _, cache = model(x[:1], return_cache=True)

        # cache['attn'] shape: [B, H, T, T]
        attn_l0 = cache["attn"][0][0, 0].cpu().numpy()  # Layer 0 Head 0
        attn_l1 = cache["attn"][1][0, 0].cpu().numpy()  # Layer 1 Head 0

        im0 = axes[idx][0].imshow(attn_l0, cmap="Blues", vmin=0, vmax=1)
        axes[idx][0].set_title(f"{gen_name} - L0 H0 (Previous Token)")
        axes[idx][0].set_xlabel("Key Position")
        axes[idx][0].set_ylabel("Query Position")
        plt.colorbar(im0, ax=axes[idx][0])

        im1 = axes[idx][1].imshow(attn_l1, cmap="magma", vmin=0, vmax=1)
        axes[idx][1].set_title(f"{gen_name} - L1 H0 (Induction)")
        axes[idx][1].set_xlabel("Key Position")
        axes[idx][1].set_ylabel("Query Position")
        plt.colorbar(im1, ax=axes[idx][1])

    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    plt.close()
    print(f"Saved attention comparison map to {output_fig}")


if __name__ == "__main__":
    runs = {
        "Gen 0": "results/base_single/model.pt",
    }
    # loop through all gens
    for g in range(1, 11):
        p = f"results/collapse/base/gen_{g}/model.pt"
        if os.path.exists(p):
            runs[f"Gen {g}"] = p

    plot_attention_across_generations(runs)