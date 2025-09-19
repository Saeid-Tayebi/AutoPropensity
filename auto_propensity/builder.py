
"""
PropensityModelBuilder — pick a propensity model by validation log-likelihood (LL), then
rebuild that model on any dataset and score p(A|X). Works for DISCRETE and CONTINUOUS A.

Quick Start
-----------
builder = PropensityModelBuilder(X, A, action_type='discrete' or 'continuous', test_size=0.25, random_state=42)
res = builder.tune(verbose=False)
best_cfg = res["config"]
pm = PropensityModel.from_config(X_train, A_train, best_cfg, random_state=42)
p = pm.score(X_test, A_test)
"""
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KernelDensity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK, WhiteKernel

from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt


@dataclass
class PropensityConfig:
    action_type: str                          # 'discrete' or 'continuous'
    chosen_family: str                        # 'parametric' or 'nonparametric'
    predictive_model: str                     # e.g., 'logistic', 'multinomial', 'linear_gaussian', 'gp', 'kde', 'mlp'
    dim_reduction: Optional[str] = None       # None, 'pca', or 'pls'
    dim_reduction_params: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = None
    scaler: bool = True
    action_dim: int = 1


class PropensityModelBuilder:
    """Selects and fits a propensity model p(A|X) by validation log-likelihood.

    Parameters
    ----------
    X : np.ndarray
        State features (n, d_x).
    A : np.ndarray
        Actions: discrete labels (n,) or continuous (n, d_a).
    action_type : {'discrete','continuous'}
        Action space type.
    """

    def __init__(self,
                 X: np.ndarray,
                 A: np.ndarray,
                 action_type: str,
                 test_size: float = 0.25,
                 random_state: int = 42):
        assert action_type in ('discrete', 'continuous')
        self.X = np.asarray(X)
        self.A = np.asarray(A)
        self.action_type = action_type
        self.random_state = random_state
        self.test_size = test_size

        self.n, self.d_x = self.X.shape
        if self.action_type == 'discrete':
            self.is_multiclass = len(np.unique(self.A)) > 2
            self.d_a = 1
        else:
            if self.A.ndim == 1:
                self.A = self.A.reshape(-1, 1)
            self.d_a = self.A.shape[1]

        self.X_train, self.X_val, self.A_train, self.A_val = train_test_split(
            self.X, self.A, test_size=self.test_size, random_state=self.random_state, stratify=self.A if self.action_type == 'discrete' else None
        )

        self.best_config: Optional[PropensityConfig] = None
        self.fitted_artifacts: Dict[str, Any] = {}
        self.diagnostics_: Dict[str, Any] = {}
        # light-weight observability
        self.trace_on: bool = False
        self.trace: List[Dict[str, Any]] = []

    def _val_loglik_discrete(self, model, scaler: Optional[StandardScaler], reducer):
        '''
         Computes validation log-likelihood for a discrete classifier via predict_proba.
        '''
        Xv = self.X_val
        if scaler is not None:
            Xv = scaler.transform(Xv)
        if reducer is not None:
            Xv = reducer.transform(Xv)
        proba = model.predict_proba(Xv)
        yv = self.A_val.ravel().astype(int)
        # Use the model's class order to align columns with labels
        classes = getattr(model, "classes_", None)
        labels = classes if classes is not None else np.unique(self.A_train)
        return -log_loss(yv, proba, labels=labels, normalize=True)

    def _val_loglik_continuous_parametric(self, mean_model, sigma, scaler: Optional[StandardScaler], reducer):
        '''
         Computes validation log-likelihood for continuous parametric models using Gaussian (uni/multi) densities.
        '''
        Xv = self.X_val
        Av = self.A_val
        if scaler is not None:
            Xv = scaler.transform(Xv)
        if reducer is not None:
            Xv = reducer.transform(Xv)

        mu = mean_model.predict(Xv)
        if self.d_a == 1:
            ll = np.mean(norm.logpdf(Av.ravel(), loc=mu.ravel(), scale=sigma))
        else:
            cov = sigma if isinstance(sigma, np.ndarray) else (np.eye(self.d_a) * (sigma ** 2))
            ll = np.mean([multivariate_normal.logpdf(Av[i], mean=mu[i], cov=cov) for i in range(len(Av))])
        return ll

    def _val_loglik_continuous_kde(self, kde_joint: KernelDensity, kde_x: KernelDensity):
        '''
        Computes validation average log-density for nonparametric KDE (p(x,a) vs p(x))
        '''
        ll = 0.0
        for i in range(len(self.X_val)):
            x = self.X_val[i]
            a = self.A_val[i]
            xa = np.hstack([x, a]).reshape(1, -1)
            ll += kde_joint.score_samples(xa)[0] - kde_x.score_samples(x.reshape(1, -1))[0]
        return ll / len(self.X_val)

    def _baseline_ll(self) -> Tuple[float, float, float]:
        """Compute a naive baseline log-likelihood and "good/excellent" thresholds.

        Returns
        -------
        baseline_ll, good_ll, excellent_ll
            baseline: naive model (uniform for discrete; A ⟂ X Gaussian for continuous)
            good/excellent: baseline + {0.5, 0.8} * |baseline| (closer to 0 is better)
        """
        if self.action_type == 'discrete':
            # uniform over classes
            K = int(len(np.unique(self.A_train)))
            baseline = -np.log(K)
            good = baseline + 0.5 * abs(baseline)
            excellent = baseline + 0.8 * abs(baseline)
            return baseline, good, excellent
        # continuous: Gaussian on A ignoring X
        Av = self.A_val
        mu = np.mean(self.A_train, axis=0)
        if self.d_a == 1:
            sigma = float(np.std(self.A_train.ravel()) + 1e-6)
            baseline = float(np.mean(norm.logpdf(Av.ravel(), loc=mu.ravel()
                             if hasattr(mu, 'ravel') else mu, scale=sigma)))
        else:
            cov = np.cov(self.A_train.T) + 1e-6 * np.eye(self.d_a)
            baseline = float(np.mean([multivariate_normal.logpdf(Av[i], mean=mu, cov=cov) for i in range(len(Av))]))
        good = baseline + 0.5 * abs(baseline)
        excellent = baseline + 0.8 * abs(baseline)
        return baseline, good, excellent

    def tune(self,
             allow_nonparametric: bool = True,
             allow_dimred: bool = True,
             allow_ann: bool = False,
             verbose: bool = True) -> Dict[str, Any]:
        """
        Grid-search over (scaler, optional dim-reduction, predictive model family & hyperparameters).
        Select the configuration with the highest validation log-likelihood and store its fitted artifacts.

        Returns
        -------
        dict
            {'config': <PropensityConfig as dict>, 'val_loglik': <float>}
        """
        candidates: List[Tuple[PropensityConfig, Dict[str, Any]]] = []

        # Internal guard: in 1D continuous, default to parametric search only for stability
        allow_np = allow_nonparametric
        if self.action_type == 'continuous' and self.d_a == 1 and allow_np:
            allow_np = False

        scalers = [None, StandardScaler()]
        reducers = [None]
        if allow_dimred:
            pca_components = [min(self.d_x, k) for k in [2, 4, 8] if k <= self.d_x]
            reducers += [('pca', PCA(n_components=k)) for k in pca_components]
            if self.action_type == 'continuous':
                pls_components = [min(self.d_x, 2), min(self.d_x, 4)]
                reducers += [('pls', PLSRegression(n_components=k)) for k in pls_components]

        if self.action_type == 'discrete':
            for scaler in scalers:
                for red in reducers:
                    reducer_name = None
                    reducer = None
                    if isinstance(red, tuple):
                        reducer_name, red = red
                    else:
                        reducer = red
                    for C_val in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
                        for solver in ['lbfgs', 'liblinear']:
                            lr = LogisticRegression(C=C_val, max_iter=5000, tol=1e-5,
                                                    solver=solver, class_weight='balanced')

                            Xtr = self.X_train
                            if scaler is not None:
                                scaler.fit(Xtr)
                                Xtr = scaler.transform(Xtr)
                            if isinstance(red, PLSRegression):
                                red.fit(Xtr, self.A_train)
                                Xtr = red.transform(Xtr)
                            elif isinstance(red, PCA):
                                red.fit(Xtr)
                                Xtr = red.transform(Xtr)

                            lr.fit(Xtr, self.A_train.ravel().astype(int))
                            ll = self._val_loglik_discrete(lr, scaler, red)
                            cfg = PropensityConfig(
                                action_type='discrete',
                                chosen_family='parametric',
                                predictive_model='multinomial' if len(np.unique(self.A_train)) > 2 else 'logistic',
                                dim_reduction=None if red is None else (
                                    reducer_name if reducer_name else red.__class__.__name__.lower()),
                                dim_reduction_params=None if red is None else {
                                    'n_components': getattr(red, 'n_components', None)},
                                model_params={'C': C_val, 'solver': solver},
                                scaler=scaler is not None,
                                action_dim=1
                            )
                            cand = {'ll': ll, 'fitted': {'scaler': scaler, 'reducer': red, 'model': lr}}
                            candidates.append((cfg, cand))
                            if self.trace_on:
                                self.trace.append({'config': asdict(cfg), 'val_loglik': ll})

        else:
            for scaler in scalers:
                for red in reducers:
                    reducer_name = None
                    reducer = None
                    if isinstance(red, tuple):
                        reducer_name, red = red
                    else:
                        reducer = red

                    Xtr = self.X_train
                    Atr = self.A_train
                    if scaler is not None:
                        scaler.fit(Xtr)
                        Xtr = scaler.transform(Xtr)
                    if isinstance(red, PLSRegression):
                        red.fit(Xtr, Atr)
                        Xtr = red.transform(Xtr)
                    elif isinstance(red, PCA):
                        red.fit(Xtr)
                        Xtr = red.transform(Xtr)

                    lin = LinearRegression().fit(Xtr, Atr)
                    mu = lin.predict(Xtr)
                    resid = Atr - mu
                    if self.d_a == 1:
                        sigma = np.std(resid.ravel()) + 1e-6
                    else:
                        sigma = np.cov(resid.T) + 1e-6 * np.eye(self.d_a)
                    ll = self._val_loglik_continuous_parametric(lin, sigma, scaler, red)
                    cfg = PropensityConfig(
                        action_type='continuous',
                        chosen_family='parametric',
                        predictive_model='linear_gaussian',
                        dim_reduction=None if red is None else (
                            reducer_name if reducer_name else red.__class__.__name__.lower()),
                        dim_reduction_params=None if red is None else {
                            'n_components': getattr(red, 'n_components', None)},
                        model_params={'sigma': float(sigma) if np.isscalar(sigma) else {'cov': sigma.tolist()}},
                        scaler=scaler is not None,
                        action_dim=self.d_a
                    )
                    cand = {'ll': ll, 'fitted': {'scaler': scaler, 'reducer': red, 'model': lin, 'sigma': sigma}}
                    candidates.append((cfg, cand))
                    if self.trace_on:
                        self.trace.append({'config': asdict(cfg), 'val_loglik': ll})

                    kernel = (
                        CK(1.0, (1e-4, 1e4))
                        * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
                    )
                    try:
                        gp = GaussianProcessRegressor(
                            kernel=kernel,
                            normalize_y=True,
                            random_state=self.random_state,
                            n_restarts_optimizer=5,
                        )
                        gp.fit(Xtr, Atr)
                        mu_gp = gp.predict(Xtr)
                        resid_gp = Atr - mu_gp
                        if self.d_a == 1:
                            sigma_gp = np.std(resid_gp.ravel()) + 1e-6
                        else:
                            sigma_gp = np.cov(resid_gp.T) + 1e-6 * np.eye(self.d_a)
                        ll_gp = self._val_loglik_continuous_parametric(gp, sigma_gp, scaler, red)
                        cfg_gp = PropensityConfig(
                            action_type='continuous',
                            chosen_family='parametric',
                            predictive_model='gaussian_process',
                            dim_reduction=None if red is None else (
                                reducer_name if reducer_name else red.__class__.__name__.lower()),
                            dim_reduction_params=None if red is None else {
                                'n_components': getattr(red, 'n_components', None)},
                            model_params={'sigma': float(sigma_gp) if np.isscalar(sigma_gp) else {'cov': sigma_gp.tolist()},
                                          'kernel': str(gp.kernel_)},
                            scaler=scaler is not None,
                            action_dim=self.d_a
                        )
                        cand_gp = {'ll': ll_gp, 'fitted': {'scaler': scaler,
                                                           'reducer': red, 'model': gp, 'sigma': sigma_gp}}
                        candidates.append((cfg_gp, cand_gp))
                        if self.trace_on:
                            self.trace.append({'config': asdict(cfg_gp), 'val_loglik': ll_gp})
                    except Exception as e:
                        if verbose:
                            print("GP failed:", e)

            if allow_np:
                bw_grid = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
                XA_train = np.hstack([self.X_train, self.A_train])
                for bw in bw_grid:
                    kde_joint = KernelDensity(kernel='gaussian', bandwidth=bw).fit(XA_train)
                    kde_x = KernelDensity(kernel='gaussian', bandwidth=bw).fit(self.X_train)
                    ll = self._val_loglik_continuous_kde(kde_joint, kde_x)
                    cfg_kde = PropensityConfig(
                        action_type='continuous',
                        chosen_family='nonparametric',
                        predictive_model='kde',
                        dim_reduction=None,
                        dim_reduction_params=None,
                        model_params={'bandwidth': float(bw)},
                        scaler=False,
                        action_dim=self.d_a
                    )
                    cand_kde = {'ll': ll, 'fitted': {'kde_joint': kde_joint, 'kde_x': kde_x}}
                    candidates.append((cfg_kde, cand_kde))
                    if self.trace_on:
                        self.trace.append({'config': asdict(cfg_kde), 'val_loglik': ll})

            if allow_ann:
                for hidden in [(64,), (128,), (64, 64)]:
                    mlp = MLPRegressor(hidden_layer_sizes=hidden, random_state=self.random_state, max_iter=500)
                    Xtr = self.X_train
                    scaler = StandardScaler().fit(Xtr)
                    Xtr_s = scaler.transform(Xtr)
                    mlp.fit(Xtr_s, self.A_train)
                    mu = mlp.predict(Xtr_s)
                    resid = self.A_train - mu
                    sigma = np.std(resid.ravel()) + \
                        1e-6 if self.d_a == 1 else (np.cov(resid.T) + 1e-6 * np.eye(self.d_a))
                    ll = self._val_loglik_continuous_parametric(mlp, sigma, scaler, None)
                    cfg_mlp = PropensityConfig(
                        action_type='continuous',
                        chosen_family='parametric',
                        predictive_model='mlp',
                        dim_reduction=None,
                        dim_reduction_params=None,
                        model_params={'hidden_layers': hidden, 'sigma': float(
                            sigma) if np.isscalar(sigma) else {'cov': sigma.tolist()}},
                        scaler=True,
                        action_dim=self.d_a
                    )
                    candidates.append((cfg_mlp, {'ll': ll, 'fitted': {'scaler': scaler,
                                      'reducer': None, 'model': mlp, 'sigma': sigma}}))

        best = max(candidates, key=lambda t: t[1]['ll'])
        self.best_config = best[0]
        self.fitted_artifacts = best[1]['fitted']
        self.diagnostics_['val_loglik'] = best[1]['ll']

        if verbose:
            print("Best config:", asdict(self.best_config))
            print("Validation log-likelihood:", self.diagnostics_['val_loglik'])

        return {'config': asdict(self.best_config), 'val_loglik': self.diagnostics_['val_loglik']}

    # -------- convenience surfaces (no behavior change) --------
    def summarize(self) -> str:
        """Human-readable one-pager of the chosen model."""
        assert self.best_config is not None, "Call tune() first."
        cfg = asdict(self.best_config)
        ll = self.diagnostics_.get('val_loglik', None)
        n_train = len(self.X_train)
        n_val = len(self.X_val)
        if self.action_type == 'discrete':
            classes, counts = np.unique(self.A_train, return_counts=True)
            balance = {int(c): int(k) for c, k in zip(classes, counts)}
        else:
            balance = {'d_a': int(self.d_a)}
        # Baseline and qualitative rating
        base, good, excellent = self._baseline_ll()
        # qualitative rating relative to baseline toward 0
        rel = (ll - base) / (abs(base) + 1e-9) if ll is not None else None
        if rel is None:
            rating = "n/a"
        elif rel >= 0.8:
            rating = "excellent"
        elif rel >= 0.5:
            rating = "good"
        elif rel >= 0.2:
            rating = "fair"
        else:
            rating = "poor"
        lines = [
            "PropensityModelBuilder Summary",
            f"- data: n_train={n_train}, n_val={n_val}, d_x={self.d_x}",
            f"- action_type: {cfg['action_type']}  ({balance})",
            f"- choice: family={cfg['chosen_family']}, model={cfg['predictive_model']}",
            f"- scaler: {cfg['scaler']}  dim_reduction: {cfg['dim_reduction']} {cfg['dim_reduction_params']}",
            f"- model_params: {cfg['model_params']}",
            f"- baseline LL: {base:.6f}  (good≥{good:.6f}, excellent≥{excellent:.6f})",
            f"- qualitative rating: {rating}",
            f"- validation log-likelihood: {ll:.6f}" if ll is not None else "- validation log-likelihood: n/a",
        ]
        return "\n".join(lines)

    def candidates_table(self, top_k: int = 10) -> "list[tuple]":
        """Return top-k candidates as (val_loglik, model, family, dimred, scaler, params). Requires trace_on=True before tune()."""
        if not self.trace:
            return []
        rows = []
        for t in self.trace:
            c = t['config']
            rows.append((
                float(t['val_loglik']),
                c.get('predictive_model'),
                c.get('chosen_family'),
                c.get('dim_reduction'),
                bool(c.get('scaler', False)),
                c.get('model_params'),
            ))
        rows.sort(key=lambda r: r[0], reverse=True)
        return rows[:top_k]

    def plot_family_variants(self, family: str | None = None, top_k: int | None = None):
        """Bar plot of validation log-likelihood for all tried variants.

        Parameters
        ----------
        family : {None, 'parametric', 'nonparametric'} or predictive model name
            If None, include all candidates. If 'parametric'/'nonparametric', filter by chosen_family.
            If a predictive model name is given (e.g., 'multinomial', 'linear_gaussian', 'gaussian_process', 'kde', 'mlp'),
            filter to that model only.
        top_k : int or None
            If set, keep only the top_k candidates by val-LL after filtering.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self.trace:
            raise RuntimeError("No trace available. Set builder.trace_on=True before tune().")

        # Build rows from trace (no pandas dependency)
        rows = []
        for t in self.trace:
            c = t['config']
            ll = float(t['val_loglik'])
            model_name = (c.get('predictive_model') or '').lower()
            label = f"{model_name}|{c.get('dim_reduction') or 'none'}|scale={bool(c.get('scaler', False))}"
            rows.append({
                'predictive_model': model_name,
                'chosen_family': (c.get('chosen_family') or '').lower(),
                'label': label,
                'll': ll,
                'config': c,
            })

        # Optional filter
        if family is not None:
            fam = str(family).lower()
            if fam in {'parametric', 'nonparametric'}:
                rows = [r for r in rows if r['chosen_family'] == fam]
            else:
                rows = [r for r in rows if r['predictive_model'] == fam]

        # Sort and trim
        rows.sort(key=lambda r: r['ll'], reverse=True)
        if top_k is not None:
            rows = rows[:top_k]
        if not rows:
            raise RuntimeError("No candidates match the given filter.")

        labels = [r['label'] for r in rows]
        vals = [r['ll'] for r in rows]
        models = [r['predictive_model'] for r in rows]

        # Colors only for present families
        present = sorted(set(models))
        if self.action_type == 'continuous':
            palette = {
                'gaussian_process': 'tab:blue',
                'linear_gaussian': 'tab:orange',
                'kde': 'tab:purple',
                'mlp': 'tab:green',
            }
        else:
            palette = {
                'multinomial': 'tab:blue',
                'logistic': 'tab:orange',
            }
        model_to_color = {m: palette.get(m, 'gray') for m in present}
        colors = [model_to_color.get(m, 'gray') for m in models]

        fig, ax = plt.subplots(figsize=(max(6, min(16, 0.3*len(labels)+4)), 4))
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(labels)), labels)
        plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
        ax.set_ylabel('Validation log-likelihood')

        # Title suffix clarifies interpretation by action type
        title_suffix = ' (higher is better)' if self.action_type == 'continuous' else ' (closer to 0 is better)'
        ax.set_title('Candidate variants' + (f" — filter: {family}" if family else '') + title_suffix)

        handles, legend_labels = [], []
        # Reference lines only for DISCRETE case (well-calibrated around 0)
        if self.action_type == 'discrete':
            base, good, excellent = self._baseline_ll()
            ax.axhline(excellent, color='green', linestyle='--', linewidth=1.2, label='Excellent (baseline+80%)')
            ax.axhline(good,      color='orange', linestyle='--', linewidth=1.2, label='Good (baseline+50%)')
            ax.axhline(base,      color='red', linestyle='--', linewidth=1.2, label='Baseline')
            # start legend with the reference lines
            handles, legend_labels = ax.get_legend_handles_labels()

        # Extend legend with present families (works for both types)
        for fam in present:
            col = model_to_color.get(fam)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=col))
            legend_labels.append(fam)
        if handles:
            ax.legend(handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    def plot_best_per_family(self):
        """Bar plot of the best candidate from each predictive model family.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self.trace:
            raise RuntimeError("No trace available. Set builder.trace_on=True before tune().")

        # Find best per predictive_model (no pandas dependency)
        best: Dict[str, Dict[str, Any]] = {}
        for t in self.trace:
            c = t['config']
            model = (c.get('predictive_model') or '').lower()
            ll = float(t['val_loglik'])
            label = f"{model}|{c.get('dim_reduction') or 'none'}|scale={bool(c.get('scaler', False))}"
            if (model not in best) or (ll > best[model]['ll']):
                best[model] = {'ll': ll, 'label': label, 'config': c}

        if not best:
            raise RuntimeError("Trace did not record any candidates.")

        items = sorted(best.items(), key=lambda kv: kv[1]['ll'], reverse=True)
        labels = [kv[1]['label'] for kv in items]
        vals = [kv[1]['ll'] for kv in items]
        models = [kv[0] for kv in items]

        # Colors only for present families
        present = sorted(set(models))
        if self.action_type == 'continuous':
            palette = {
                'gaussian_process': 'tab:blue',
                'linear_gaussian': 'tab:orange',
                'kde': 'tab:purple',
                'mlp': 'tab:green',
            }
        else:
            palette = {
                'multinomial': 'tab:blue',
                'logistic': 'tab:orange',
            }
        model_to_color = {m: palette.get(m, 'gray') for m in present}
        colors = [model_to_color.get(m, 'gray') for m in models]

        fig, ax = plt.subplots(figsize=(max(6, min(16, 0.3*len(labels)+4)), 4))
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(labels)), labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Validation log-likelihood')

        title_suffix = f' (higher is better) best ={np.max(vals):.2f}' if self.action_type == 'continuous' else f' (closer to 0 is better) best={np.max(vals):.2f}'
        ax.set_title('Best of each predictive model' + title_suffix)

        handles, legend_labels = [], []
        if self.action_type == 'discrete':
            base, good, excellent = self._baseline_ll()
            ax.axhline(excellent, color='green', linestyle='--', linewidth=1.2, label='Excellent (baseline+80%)')
            ax.axhline(good,      color='orange', linestyle='--', linewidth=1.2, label='Good (baseline+50%)')
            ax.axhline(base,      color='red', linestyle='--', linewidth=1.2, label='Baseline')
            handles, legend_labels = ax.get_legend_handles_labels()

        for fam in present:
            col = model_to_color.get(fam)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=col))
            legend_labels.append(fam)
        if handles:
            ax.legend(handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    # ---- New: build fresh model artifacts on any dataset using a chosen config ----

    def build_artifacts_from_config(
        self,
        X_new_dataset: np.ndarray,
        A_new_dataset: np.ndarray,
        config: "PropensityConfig | Dict[str, Any]"
    ) -> Dict[str, Any]:
        """Fit a *fresh* propensity model and return **raw artifacts** dict (legacy/advanced).
        Returns {'fitted': ..., 'meta': ..., 'config': ...}. No class state is reused.
        """
        cfg = config if isinstance(config, PropensityConfig) else PropensityConfig(**config)
        fitted, meta, cfg_out = self._fit_core(
            X=np.asarray(X_new_dataset),
            A=np.asarray(A_new_dataset),
            cfg=cfg,
            random_state=self.random_state
        )
        return {'fitted': fitted, 'meta': meta, 'config': cfg_out}

    def build_model_from_config(
        self,
        X_new_dataset: np.ndarray,
        A_new_dataset: np.ndarray,
        config: "PropensityConfig | Dict[str, Any]",
    ):
        """Unified API: returns a PropensityModel fitted fresh on (X_new_dataset, A_new_dataset)."""
        return self.build_class_from_config(X_new_dataset, A_new_dataset, config)

    @staticmethod
    def score_with(artifacts: Dict[str, Any],
                   action_type: str,
                   d_a: int,
                   X_new: np.ndarray,
                   A_new: np.ndarray) -> np.ndarray:
        """Score p(A|X) using artifacts returned by build_model_from_config (no builder state)."""
        return PropensityModelBuilder._score_core(artifacts, action_type, d_a, X_new, A_new)

    # ---- New: shared helpers (used by builder and PropensityModel) ----
    @staticmethod
    def _fit_core(X: np.ndarray, A: np.ndarray, cfg: PropensityConfig, random_state: int
                  ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Centralized fit. Builds scaler/reducer/core model based on cfg, trained on (X,A)."""
        Xn = np.asarray(X)
        An = np.asarray(A)
        if cfg.action_type == 'continuous' and An.ndim == 1 and cfg.action_dim > 1:
            An = An.reshape(-1, cfg.action_dim)

        # Scaler / reducer
        scaler = StandardScaler() if cfg.scaler else None
        reducer = None
        if cfg.dim_reduction is not None:
            if cfg.dim_reduction == 'pca':
                n_comp = (cfg.dim_reduction_params or {}).get('n_components', None)
                reducer = PCA(n_components=n_comp)
            elif cfg.dim_reduction == 'pls':
                n_comp = (cfg.dim_reduction_params or {}).get('n_components', None)
                reducer = PLSRegression(n_components=n_comp)

        Xf = Xn
        if scaler is not None:
            scaler.fit(Xf)
            Xf = scaler.transform(Xf)
        if reducer is not None:
            if isinstance(reducer, PLSRegression):
                reducer.fit(Xf, An)
            else:
                reducer.fit(Xf)
            Xf = reducer.transform(Xf)

        fitted: Dict[str, Any] = {'scaler': scaler, 'reducer': reducer}

        # Core models
        if cfg.action_type == 'discrete':
            # defaults ensure robust performance across class imbalance and sklearn versions
            params = cfg.model_params or {}
            C_val = params.get('C', 1.0)
            solver = params.get('solver', 'lbfgs')
            lr = LogisticRegression(C=C_val, max_iter=5000, tol=1e-5, solver=solver, class_weight='balanced')
            lr.fit(Xf, An.ravel().astype(int))
            fitted['model'] = lr
        elif cfg.chosen_family == 'nonparametric' and cfg.predictive_model == 'kde':
            bw = float((cfg.model_params or {}).get('bandwidth', 1.0))
            # KDE in the ORIGINAL space (X, A)
            XA = np.hstack([Xn, An])
            kde_joint = KernelDensity(kernel='gaussian', bandwidth=bw).fit(XA)
            kde_x = KernelDensity(kernel='gaussian', bandwidth=bw).fit(Xn)
            fitted['kde_joint'] = kde_joint
            fitted['kde_x'] = kde_x
        else:
            if cfg.predictive_model == 'linear_gaussian':
                lin = LinearRegression().fit(Xf, An)
                mu = lin.predict(Xf)
                resid = An - mu
                sigma = (np.std(resid.ravel()) + 1e-6) if cfg.action_dim == 1 else (np.cov(resid.T) +
                                                                                    1e-6*np.eye(cfg.action_dim))
                fitted['model'] = lin
                fitted['sigma'] = sigma
            elif cfg.predictive_model in ('gaussian_process', 'gp'):
                kernel = (
                    CK(1.0, (1e-4, 1e4))
                    * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
                )
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    random_state=random_state,
                    n_restarts_optimizer=5,
                )
                gp.fit(Xf, An)
                mu = gp.predict(Xf)
                resid = An - mu
                sigma = (np.std(resid.ravel()) + 1e-6) if cfg.action_dim == 1 else (np.cov(resid.T) +
                                                                                    1e-6*np.eye(cfg.action_dim))
                fitted['model'] = gp
                fitted['sigma'] = sigma
            else:
                raise ValueError(f"Unsupported predictive_model for continuous actions: {cfg.predictive_model}")

        meta = {'action_type': cfg.action_type, 'action_dim': cfg.action_dim}
        return fitted, meta, asdict(cfg)

    @staticmethod
    def _score_core(artifacts: Dict[str, Any], action_type: str, d_a: int,
                    X_new: np.ndarray, A_new: np.ndarray) -> np.ndarray:
        """Centralized scorer."""
        Xn = np.asarray(X_new)
        An = np.asarray(A_new)
        scaler = artifacts.get('scaler', None)
        reducer = artifacts.get('reducer', None)

        if action_type == 'continuous' and An.ndim == 1 and d_a > 1:
            An = An.reshape(-1, d_a)

        if action_type == 'discrete':
            model = artifacts['model']
            Xp = scaler.transform(Xn) if scaler is not None else Xn
            Xp = reducer.transform(Xp) if reducer is not None else Xp
            proba = model.predict_proba(Xp)
            A_int = An.ravel().astype(int)
            classes = getattr(model, 'classes_', None)
            if classes is None:
                # Fallback: assume 0..K-1
                idx = np.clip(A_int, 0, proba.shape[1]-1)
            else:
                # Map true class labels to probability column indices
                mapping = {int(c): i for i, c in enumerate(classes)}
                idx = np.array([mapping.get(int(a), 0) for a in A_int], dtype=int)
                idx = np.clip(idx, 0, proba.shape[1]-1)
            return proba[np.arange(len(A_int)), idx]

        # nonparametric KDE
        if 'kde_joint' in artifacts and 'kde_x' in artifacts:
            kde_joint = artifacts['kde_joint']
            kde_x = artifacts['kde_x']
            out = []
            for i in range(len(Xn)):
                xa = np.hstack([Xn[i], An[i]]).reshape(1, -1)
                out.append(np.exp(kde_joint.score_samples(xa)[0] - kde_x.score_samples(Xn[i].reshape(1, -1))[0]))
            return np.maximum(np.array(out), 1e-300)

        # parametric continuous
        model = artifacts['model']
        sigma = artifacts['sigma']
        Xp = scaler.transform(Xn) if scaler is not None else Xn
        Xp = reducer.transform(Xp) if reducer is not None else Xp
        mu = model.predict(Xp)
        if d_a == 1:
            return np.maximum(norm.pdf(An.ravel(), loc=mu.ravel(), scale=sigma), 1e-300)
        else:
            cov = sigma if isinstance(sigma, np.ndarray) else (np.eye(d_a) * (sigma ** 2))
            return np.maximum(np.array([multivariate_normal.pdf(An[i], mean=mu[i], cov=cov) for i in range(len(An))]), 1e-300)

    # ---- New: return a lightweight model class instance ----
    def build_class_from_config(self,
                                X_new_dataset: np.ndarray,
                                A_new_dataset: np.ndarray,
                                config: "PropensityConfig | Dict[str, Any]"):
        """Return a PropensityModel (standalone) trained fresh on (X_new_dataset, A_new_dataset)."""
        from auto_propensity.model import PropensityModel  # local import avoids circular refs
        cfg = config if isinstance(config, PropensityConfig) else PropensityConfig(**config)
        fitted, meta, cfg_out = self._fit_core(
            X=np.asarray(X_new_dataset),
            A=np.asarray(A_new_dataset),
            cfg=cfg,
            random_state=self.random_state
        )
        return PropensityModel(config=cfg_out, fitted=fitted, meta=meta)

    def score(self, X_new: np.ndarray, A_new: np.ndarray) -> np.ndarray:
        assert self.best_config is not None, "Call tune() first."
        cfg = self.best_config

        Xn = np.asarray(X_new)
        An = np.asarray(A_new)
        if cfg.action_type == 'continuous' and An.ndim == 1 and self.d_a > 1:
            An = An.reshape(-1, self.d_a)
        if cfg.action_type == 'discrete':
            scaler = self.fitted_artifacts.get('scaler', None)
            reducer = self.fitted_artifacts.get('reducer', None)
            model = self.fitted_artifacts['model']
            Xp = Xn
            if scaler is not None:
                Xp = scaler.transform(Xp)
            if reducer is not None:
                Xp = reducer.transform(Xp)
            proba = model.predict_proba(Xp)
            A_int = An.ravel().astype(int)
            classes = getattr(model, 'classes_', None)
            if classes is None:
                idx = np.clip(A_int, 0, proba.shape[1]-1)
            else:
                mapping = {int(c): i for i, c in enumerate(classes)}
                idx = np.array([mapping.get(int(a), 0) for a in A_int], dtype=int)
                idx = np.clip(idx, 0, proba.shape[1]-1)
            return proba[np.arange(len(A_int)), idx]

        if cfg.chosen_family == 'nonparametric' and cfg.predictive_model == 'kde':
            kde_joint = self.fitted_artifacts['kde_joint']
            kde_x = self.fitted_artifacts['kde_x']
            out = []
            for i in range(len(Xn)):
                xa = np.hstack([Xn[i], An[i]]).reshape(1, -1)
                out.append(np.exp(kde_joint.score_samples(xa)[0] - kde_x.score_samples(Xn[i].reshape(1, -1))[0]))
            return np.maximum(np.array(out), 1e-300)
        else:
            scaler = self.fitted_artifacts.get('scaler', None)
            reducer = self.fitted_artifacts.get('reducer', None)
            model = self.fitted_artifacts['model']
            sigma = self.fitted_artifacts['sigma']
            Xp = Xn
            if scaler is not None:
                Xp = scaler.transform(Xp)
            if reducer is not None:
                Xp = reducer.transform(Xp)
            mu = model.predict(Xp)
            if self.d_a == 1:
                return np.maximum(norm.pdf(An.ravel(), loc=mu.ravel(), scale=sigma), 1e-300)
            else:
                cov = sigma if isinstance(sigma, np.ndarray) else (np.eye(self.d_a) * (sigma ** 2))
                return np.maximum(np.array([multivariate_normal.pdf(An[i], mean=mu[i], cov=cov) for i in range(len(An))]), 1e-300)

    def plot_propensity_hist(self, bins: int = 30):
        assert self.action_type == 'discrete', "Only for discrete A."
        p = self.score(self.X_val, self.A_val)
        plt.figure()
        plt.hist(p, bins=bins, edgecolor='k')
        plt.title("Histogram of p(A|X) on validation")
        plt.xlabel("p(A|X)")
        plt.ylabel("count")
        return plt.gcf()

    def plot_density_values(self, n=200):
        assert self.action_type == 'continuous', "Only for continuous A."
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(self.X_val), size=min(n, len(self.X_val)), replace=False)
        p = self.score(self.X_val[idx], self.A_val[idx])
        plt.figure()
        plt.scatter(np.arange(len(p)), p, s=12)
        plt.title("Predicted p(A|X) densities (subset of validation)")
        plt.xlabel("sample")
        plt.ylabel("density")
        return plt.gcf()
