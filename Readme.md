# AutoPropensity — Auto-select p(A|X) by Validation Log-Likelihood

**Goal:** Given a dataset with states `X` and actions `A` (either **discrete** or **continuous**), this library
**automatically searches** a small, sensible family of propensity models and selects the configuration that **maximizes
validation log-likelihood**. You can then **rebuild** that model on any subset/split and **score** `p(A|X)` for new points.

> TL;DR — Pass your data and whether `A` is _discrete_ or _continuous_. Get back a config you can reuse anywhere.

---

## Why this is helpful

- **No guesswork:** Stop hand-picking models every time you see new data.
- **Unified API:** Works for both **discrete** and **continuous** action spaces.
- **Leak-free workflow:** Tune once → save config → **rebuild on any split** (train/val/test/production).
- **Lightweight & type-safe:** Small, readable code with few dependencies (scikit-learn, numpy, scipy, matplotlib for plots).

---

## What it searches

By default the tuner explores:

- **Discrete `A`:**
  - Logistic / Multinomial Regression (+ optional StandardScaler, PCA/PLS)
- **Continuous `A`:**
  - Linear-Gaussian (residual sigma estimated)
  - Gaussian Process (with RBF + white noise)
  - Optional KDE for higher-dimensional cases (off by default in 1-D `A` for stability)
  - Optional ANN (MLP) toggle

Dimensionality reduction (PCA/PLS) and scaling are considered in the grid.

**Selection metric:** **validation log-likelihood (LL)**.  
For discrete models, “closer to 0” is better. For continuous densities, “higher” is better.

---

## Quick Start

```python
import auto_propensity as ap
from auto_propensity import PropensityModelBuilder

# X: (n, d_x), A: (n,) for discrete or (n, d_a)/(n,) for continuous
builder = PropensityModelBuilder(X, A, action_type='discrete' or 'continuous', test_size=0.25, random_state=42)
result = builder.tune(verbose=False)
best_cfg = result["config"]

# Rebuild a fresh model on any subset (no leakage)
pm = ap.make_from_config(X_train, A_train, best_cfg, random_state=42)
p, avg_ll, per_ll = ap.score_and_ll(pm, X_test, A_test)
```

---

## Installation

You can install the package from PyPI using:

```bash
pip install auto-propensity
```

Alternatively, install directly from GitHub with:

```bash
pip install git+https://github.com/Saeid-Tayebi/AutoPropensity.git
```

---

## Visualization

You can enable detailed tracing of the tuning process by setting `builder.trace_on = True`. This allows you to visualize and compare candidate models using the provided plotting functions:

- `plot_family_variants()`: Compare variants within each model family.
- `plot_best_per_family()`: Compare the best models across different families.

Example usage:

```python
builder.trace_on = True
result = builder.tune(verbose=False)

import os, matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)

fig1 = builder.plot_family_variants()
fig1.savefig("figs/variants_discrete.png", dpi=150, bbox_inches="tight")
plt.close(fig1)

fig2 = builder.plot_best_per_family()
fig2.savefig("figs/best_per_family_discrete.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
```

---

**Acknowledgment**

Parts of the implementation and code structuring were developed with the assistance of ChatGPT.  
The conceptual design and research ideas, however, were entirely original and not derived from ChatGPT.
