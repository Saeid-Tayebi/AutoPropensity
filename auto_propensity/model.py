
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Union
import numpy as np

from .builder import PropensityModelBuilder, PropensityConfig
from auto_propensity.builder import PropensityModelBuilder, PropensityConfig
import numpy as np
from typing import Any, Dict, Union
from dataclasses import dataclass, asdict
"""
PropensityModel — lightweight wrapper to fit a model from a config and score p(A|X).

Example
-------
builder = PropensityModelBuilder(X_train, A_train, action_type='discrete')
best = builder.tune(verbose=False)["config"]
pm = PropensityModel.from_config(X_val, A_val, best, random_state=42)
p = pm.score(X_test, A_test)
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


"""
PropensityModel — lightweight wrapper to fit a model from a config and score p(A|X).

Example
-------
builder = PropensityModelBuilder(X_train, A_train, action_type='discrete')
best = builder.tune(verbose=False)["config"]
pm = PropensityModel.from_config(X_val, A_val, best, random_state=42)
p = pm.score(X_test, A_test)
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
