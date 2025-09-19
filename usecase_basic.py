"""
Minimal end-to-end example:
1) Pick a config with PropensityModelBuilder.tune()
2) Fit a fresh PropensityModel on your chosen split
3) Score p(A|X) on new data with a tiny handler
"""

import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

import auto_propensity as ap
from auto_propensity import PropensityModelBuilder


def main():
    rng = np.random.default_rng(42)

    # --- pretend these come from your pipeline ---
    # Choose 'discrete' or 'continuous' to match your action space
    action_type = 'discrete'  # or 'continuous'

    if action_type == 'discrete':
        n, d_x, K = 2000, 10, 5
        X = rng.normal(size=(n, d_x))
        W = rng.normal(size=(d_x, K))
        logits = X @ W + rng.normal(scale=0.5, size=(n, K))
        A = np.argmax(logits, axis=1)
    else:
        n, d_x = 2000, 10
        X = rng.normal(size=(n, d_x))
        w = rng.normal(size=d_x)
        A = X @ w + rng.normal(scale=0.7, size=n)

    X_train, X_test, A_train, A_test = train_test_split(
        X, A, test_size=0.25, random_state=42, stratify=A if action_type == 'discrete' else None
    )

    # 1) Tune once to select a configuration (by validation log-likelihood)
    builder = PropensityModelBuilder(X_train, A_train, action_type=action_type, test_size=0.25, random_state=42)
    builder.trace_on = True  # record all candidates so we can plot comparisons
    result = builder.tune(verbose=False)
    best_cfg = result["config"]

    # --- plots that show why the picked model is better ---
    outdir = "figs"
    os.makedirs(outdir, exist_ok=True)
    try:
        fig_variants = builder.plot_family_variants()
        path_variants = os.path.join(outdir, f"variants_{action_type}.png")
        fig_variants.savefig(path_variants, dpi=150, bbox_inches="tight")
        plt.close(fig_variants)
        print(f"Saved: {path_variants}")
    except Exception as e:
        print(f"plot_family_variants failed: {e}")

    try:
        fig_best = builder.plot_best_per_family()
        path_best = os.path.join(outdir, f"best_per_family_{action_type}.png")
        fig_best.savefig(path_best, dpi=150, bbox_inches="tight")
        plt.close(fig_best)
        print(f"Saved: {path_best}")
    except Exception as e:
        print(f"plot_best_per_family failed: {e}")

    # 2) Fit a fresh model on whichever subset you prefer (no leakage) using package API
    pm = ap.make_from_config(X_train, A_train, best_cfg, random_state=42)

    # 3) Score p(A|X) on new points (package API)
    p, avg_ll, _ = ap.score_and_ll(pm, X_test, A_test)
    print(f"[{action_type}] avg log-likelihood on test: {avg_ll:.4f}")
    print(f"sample p(A|X): {p[:5]}")


if __name__ == "__main__":
    main()
