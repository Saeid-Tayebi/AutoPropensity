

"""
Lightweight wrapper for fitting and scoring a propensity model from a saved config.

Typical usage
-------------
handler = PropensityModelHandler().fit_from_config(X_tr, A_tr, config=best_config)
p_test, test_ll, ll_per = handler.score_and_ll(X_te, A_te)
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

from .model import PropensityModel


class PropensityModelHandler:
    """Tiny façade over a fitted PropensityModel to simplify scoring and log-likelihood evaluation."""

    def __init__(self) -> None:
        self.pm: PropensityModel | None = None

    def fit_from_config(
        self,
        X_train: np.ndarray,
        A_train: np.ndarray,
        config: Dict[str, Any],
        random_state: int = 0,
    ) -> "PropensityModelHandler":
        """Fit a fresh PropensityModel using a provided configuration dict."""
        self.pm = PropensityModel.from_config(X_train, A_train, config=config, random_state=random_state)
        return self

    def score_and_ll(self, X_test: np.ndarray, A_test: np.ndarray, eps: float = 1e-300) -> Tuple[np.ndarray, float, np.ndarray]:
        """Return (p, avg_log_likelihood, per_sample_log_likelihood)."""
        assert self.pm is not None, "Call fit_from_config first."
        p = self.pm.score(X_test, A_test)
        ll_per = np.log(np.clip(p, eps, None))
        return p, float(np.mean(ll_per)), ll_per

    def score(self, X_test: np.ndarray, A_test: np.ndarray) -> np.ndarray:
        """Return propensities/densities p(A|X) for the provided points."""
        assert self.pm is not None, "Call fit_from_config first."
        return self.pm.score(X_test, A_test)


# ---------------- Convenience: minimal smoke test (both discrete & continuous) ---------------- #

def quick_self_test(seed: int = 0) -> Dict[str, Any]:
    """
    Run a quick end-to-end smoke test:
      - DISCRETE: tune a config, refit with PropensityModel, score on holdout.
      - CONTINUOUS: same, verify densities are finite/positive.

    Returns a dict with simple diagnostics. This is not a unit test; see tests/ for pytest examples.
    """
    from sklearn.model_selection import train_test_split
    from auto_propensity.builder import PropensityModelBuilder

    rng = np.random.default_rng(seed)

    # ----- DISCRETE synthetic data -----
    n, d_x, K = 800, 6, 3
    X = rng.normal(size=(n, d_x))
    logits = np.stack(
        [X @ rng.normal(size=d_x) * 0.8 + rng.normal(scale=0.3, size=n) for _ in range(K)],
        axis=1,
    )
    A = np.argmax(logits + rng.normal(scale=0.5, size=logits.shape), axis=1)

    X_tr, X_te, A_tr, A_te = train_test_split(X, A, test_size=0.25, random_state=seed, stratify=A)
    builder_disc = PropensityModelBuilder(X_tr, A_tr, action_type="discrete", test_size=0.25, random_state=seed)
    builder_disc.trace_on = False
    best_disc = builder_disc.tune(verbose=False)
    handler_disc = PropensityModelHandler().fit_from_config(X_tr, A_tr, config=best_disc["config"], random_state=seed)
    p_disc, avg_ll_disc, _ = handler_disc.score_and_ll(X_te, A_te)

    # ----- CONTINUOUS synthetic data (1-D action) -----
    n2, d_x2 = 600, 5
    X2 = rng.normal(size=(n2, d_x2))
    w = rng.normal(size=d_x2)
    A2 = (X2 @ w) + rng.normal(scale=0.7, size=n2)  # linear-Gaussian
    X2_tr, X2_te, A2_tr, A2_te = train_test_split(X2, A2, test_size=0.25, random_state=seed)

    builder_cont = PropensityModelBuilder(X2_tr, A2_tr, action_type="continuous", test_size=0.25, random_state=seed)
    best_cont = builder_cont.tune(verbose=False)
    handler_cont = PropensityModelHandler().fit_from_config(X2_tr, A2_tr, config=best_cont["config"], random_state=seed)
    p_cont, avg_ll_cont, _ = handler_cont.score_and_ll(X2_te, A2_te)

    return {
        "discrete": {"avg_ll": float(avg_ll_disc), "p_min": float(p_disc.min()), "p_max": float(p_disc.max())},
        "continuous": {"avg_ll": float(avg_ll_cont), "p_min": float(p_cont.min()), "p_max": float(p_cont.max())},
    }


if __name__ == "__main__":
    out = quick_self_test()
    print("Self-test summary:", out)
