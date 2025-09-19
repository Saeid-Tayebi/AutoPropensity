# tests/test_propensity.py
import numpy as np
from sklearn.model_selection import train_test_split

import auto_propensity as ap
from auto_propensity import PropensityModelBuilder


def _synthetic_discrete(seed=0):
    rng = np.random.default_rng(seed)
    n, d_x, K = 1200, 8, 4
    X = rng.normal(size=(n, d_x))
    W = rng.normal(size=(d_x, K))
    logits = X @ W + rng.normal(scale=0.4, size=(n, K))
    A = np.argmax(logits, axis=1)
    return X, A


def _synthetic_continuous(seed=0):
    rng = np.random.default_rng(seed)
    n, d_x = 1000, 7
    X = rng.normal(size=(n, d_x))
    w = rng.normal(size=d_x)
    A = X @ w + rng.normal(scale=0.6, size=n)  # linear-Gaussian
    return X, A


def test_discrete_ll_beats_baseline():
    X, A = _synthetic_discrete()
    X_tr, X_te, A_tr, A_te = train_test_split(X, A, test_size=0.25, random_state=0, stratify=A)

    b = PropensityModelBuilder(X_tr, A_tr, action_type='discrete', test_size=0.25, random_state=0)
    res = b.tune(verbose=False)
    cfg = res["config"]

    # fresh fit via package API
    pm = ap.make_from_config(X_tr, A_tr, cfg, random_state=0)
    p, avg_ll, _ = ap.score_and_ll(pm, X_te, A_te)

    # uniform baseline: -log(K)
    K = len(np.unique(A_tr))
    baseline = -np.log(K)

    # Discrete LL should be closer to 0 (i.e., greater than baseline by a margin)
    assert np.isfinite(avg_ll)
    assert avg_ll > baseline + 0.2 * abs(baseline)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_continuous_ll_is_finite_and_reasonable():
    X, A = _synthetic_continuous()
    X_tr, X_te, A_tr, A_te = train_test_split(X, A, test_size=0.25, random_state=0)

    b = PropensityModelBuilder(X_tr, A_tr, action_type='continuous', test_size=0.25, random_state=0)
    res = b.tune(verbose=False)
    cfg = res["config"]

    pm = ap.make_from_config(X_tr, A_tr, cfg, random_state=0)
    p, avg_ll, _ = ap.score_and_ll(pm, X_te, A_te)

    # Basic sanity checks: densities positive, LL finite and not astronomically bad
    assert np.isfinite(avg_ll)
    assert np.all(np.isfinite(p))
    assert np.all(p > 0.0)
    # Loose bound: average log-density shouldn’t be worse than a wildly wrong sigma
    assert avg_ll > -20
