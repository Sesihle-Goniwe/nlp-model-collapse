import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from model import load_model


def analyze_symbol_embeddings(
    gen_paths, output_fig="results/figures/embedding_pca.png"
):
    """Extracts token embeddings and performs PCA to visualize manifold contraction."""
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    n_gens = len(gen_paths)
    cols = 4
    rows = (n_gens + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows))
    axes = axes.flatten()

    if n_gens == 1:
        axes = [axes]

    for idx, (gen_name, path) in enumerate(gen_paths.items()):
        model, extra = load_model(path)
        cfg = extra["task_config"]
        n_symbols = cfg["n_symbols"]

        # embeddings for symbols only (excluding labels)
        emb = model.tok_emb.weight[:n_symbols].detach().cpu().numpy()

        pca = PCA(n_components=2)
        proj = pca.fit_transform(emb)
        var_explained = np.sum(pca.explained_variance_ratio_)

        ax = axes[idx]
        ax.scatter(
            proj[:, 0], proj[:, 1], alpha=0.5, s=15, c="#2a78d6", edgecolors="none"
        )
        ax.set_title(f"{gen_name}\nExpl. Var (2D): {var_explained:.2%}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    plt.close()
    print(f"Saved embedding projection plot to {output_fig}")


if __name__ == "__main__":
    runs = {"Gen 0": "results/base_single/model.pt"}

    #loop through all gens 
    for g in range(1, 11):
        p = f"results/collapse/base/gen_{g}/model.pt"
        if os.path.exists(p):
            runs[f"Gen {g}"] = p

    analyze_symbol_embeddings(runs)