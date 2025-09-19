

import numpy as np
from helpers.propensity_model.model_creator_from_config import PropensityModel

'''
how to use this class:
handler = PropensityModelHandler().fit_from_config(X_tr, A_tr, config=best_config)
p_test, test_ll, ll_per = handler.score_and_ll(X_te, A_te)
'''


class PropensityModelHandler:
    def __init__(self):
        self.pm = None

    def fit_from_config(self, X_train, A_train, config, random_state=0):
        self.pm = PropensityModel.from_config(X_train, A_train, config=config, random_state=random_state)
        return self

    def score_and_ll(self, X_test, A_test, eps=1e-300):
        assert self.pm is not None, "Call fit_from_config first."
        p = self.pm.score(X_test, A_test)
        ll_per = np.log(np.clip(p, eps, None))
        return p, float(np.mean(ll_per)), ll_per

    def score(self, X_test, A_test):
        return self.pm.score(X_test, A_test)
        # if mode == "prob":
        #     p = self.pm.score(X_test, A_test)
        #     return p
        # elif mode == "density":
        #     # Fallback Gaussian density estimate based on residuals
        #     mu = self.pm.predict(X_test)  # expected action
        #     diff = A_test - mu
        #     sigma = np.std(diff, axis=0, ddof=1)
        #     sigma = np.maximum(sigma, 1e-6)  # avoid zero variance
        #     try:
        #         norm_const = 1.0 / np.prod(np.sqrt(2 * np.pi) * sigma)
        #         exp_term = np.exp(-0.5 * np.sum((diff / sigma) ** 2, axis=1))
        #         dens = norm_const * exp_term
        #     except Exception:
        #         dens = np.full(A_test.shape[0], 1e-300)
        #     return np.clip(dens, 1e-300, None)
