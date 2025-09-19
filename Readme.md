# Propensity Model Builder (Auto-select p(A|X) by Validation Log-Likelihood)

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
from propensity_builder import PropensityModelBuilder
from model_creator_from_config import PropensityModel

# X: (n, d_x), A: (n,) for discrete or (n, d_a)/(n,) for continuous
builder = PropensityModelBuilder(X, A, action_type='discrete' or 'continuous', test_size=0.25, random_state=42)
result = builder.tune(verbose=False)
best_cfg = result["config"]

# Rebuild a fresh model on any subset (no leakage)
pm = PropensityModel.from_config(X_train, A_train, best_cfg, random_state=42)
p = pm.score(X_test, A_test)  # propensities (discrete) or densities (continuous)
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

# Generate and save plots (images saved to 'figs/' directory)
builder.plot_family_variants(save_dir='figs/')
builder.plot_best_per_family(save_dir='figs/')
```

---

**Acknowledgment**

Parts of the implementation and code structuring were developed with the assistance of ChatGPT.  
The conceptual design and research ideas, however, were entirely original and not derived from ChatGPT.
