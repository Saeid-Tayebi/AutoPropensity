# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

from .builder import PropensityModelBuilder
from .model import PropensityModel


def tune(
    X: np.ndarray,
    A: np.ndarray,
    action_type: str,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    trace_on: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Returns {'config': <best_config>, 'val_loglik': <float>, ...}
    """
    b = PropensityModelBuilder(X, A, action_type=action_type, test_size=test_size, random_state=random_state)
    b.trace_on = trace_on
    return b.tune(verbose=verbose)


def make_from_config(
    X_train: np.ndarray,
    A_train: np.ndarray,
    config: Dict[str, Any],
    *,
    random_state: int = 42,
) -> PropensityModel:
    """Fit & return a PropensityModel using a tuned config."""
    return PropensityModel.from_config(X_train, A_train, config, random_state=random_state)


def create(
    X: np.ndarray,
    A: np.ndarray,
    action_type: str,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    trace_on: bool = False,
    verbose: bool = False,
) -> Tuple[PropensityModel, Dict[str, Any], Dict[str, Any]]:
    """
    One-shot: tune a config on (X, A) then train a fresh model on (X, A).
    Returns (pm, cfg, result_dict).
    """
    result = tune(X, A, action_type, test_size=test_size, random_state=random_state, trace_on=trace_on, verbose=verbose)
    cfg = result["config"]
    pm = make_from_config(X, A, cfg, random_state=random_state)
    return pm, cfg, result


def score(pm: PropensityModel, X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """p(A|X): propensities for discrete, densities for continuous."""
    return pm.score(X, A)


def score_and_ll(pm: PropensityModel, X: np.ndarray, A: np.ndarray, eps: float = 1e-300):
    """Convenience: (p, avg_log_likelihood, per_sample_ll)."""
    p = pm.score(X, A)
    per = np.log(np.clip(p, eps, None))
    return p, float(np.mean(per)), per
