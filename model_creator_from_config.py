
from .propensity_builder import PropensityModelBuilder, PropensityConfig
import numpy as np
from typing import Any, Dict, Union
from dataclasses import dataclass, asdict
"""
PropensityModel — a lightweight, fitted propensity model class.
Build fresh models from a chosen config and score p(A|X) on new data.

Typical usage
-------------
from helpers.propensity_model.propensity_builder import PropensityModelBuilder
from helpers.propensity_model.model_creator_from_config import PropensityModel

# 1) Tune to get a config
builder = PropensityModelBuilder(X_train, A_train, action_type='discrete')
res = builder.tune(verbose=False)
best_cfg = res["config"]

# 2) Fit a fresh model on another split (no leakage)
pm = PropensityModel.from_config(X_val, A_val, best_cfg, random_state=42)

# 3) Score on test
p_test = pm.score(X_test, A_test)
"""


@dataclass
class PropensityModel:
    """Self-contained fitted propensity model built from a config.

    Attributes
    ----------
    config : Dict[str, Any]
        The configuration used to fit the model (as a plain dict).
    fitted : Dict[str, Any]
        Artifacts required for scoring (scaler/reducer/core model/etc.).
    meta : Dict[str, Any]
        Metadata such as {'action_type', 'action_dim'}.
    """
    config: Dict[str, Any]
    fitted: Dict[str, Any]
    meta: Dict[str, Any]

    @classmethod
    def from_config(
        cls,
        X: np.ndarray,
        A: np.ndarray,
        config: Union[PropensityConfig, Dict[str, Any]],
        random_state: int | None = None,
    ) -> "PropensityModel":
        """Fit a *fresh* model on (X, A) using the provided config (no reuse of builder state)."""
        cfg = config if isinstance(config, PropensityConfig) else PropensityConfig(**config)
        X = np.asarray(X)
        A = np.asarray(A)
        fitted, meta, cfg_out = PropensityModelBuilder._fit_core(
            X=X, A=A, cfg=cfg, random_state=random_state or 0
        )
        cfg_dict = cfg_out if isinstance(cfg_out, dict) else asdict(cfg_out)
        return cls(config=cfg_dict, fitted=fitted, meta=meta)

    def score(self, X_new: np.ndarray, A_new: np.ndarray) -> np.ndarray:
        """Compute propensities/densities p(A|X) for new points using the fitted artifacts."""
        X_new = np.asarray(X_new)
        A_new = np.asarray(A_new)
        return PropensityModelBuilder._score_core(
            artifacts=self.fitted,
            action_type=self.meta['action_type'],
            d_a=self.meta['action_dim'],
            X_new=X_new,
            A_new=A_new,
        )
