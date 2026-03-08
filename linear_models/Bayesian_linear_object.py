from src.Database_object import Database

import os
import json
import logging
import importlib

import joblib
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
)

logging.getLogger("pymc").setLevel(logging.ERROR)


class BaseLinearBayesianModel:
    """
    Base class for one Bayesian linear/nonlinear model definition.
    Subclasses define:
        - priors
        - feature_cols
        - core_equation()
        - optionally _prepare_dataframe()
        - optionally _formula_latex()
    """

    priors = None
    feature_cols = None

    def __init__(
        self,
        database: Database,
        selector="test",
        export_all=True,
        target_col="Global_gradient",
        class_col="Survival/failure",
        draws=1000,
        tune=2000,
        target_accept=0.95,
    ):
        self._database = database
        self.selector = selector
        self.export_all = export_all
        self.target_col = target_col
        self.class_col = class_col
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept

        self.model_name = self.__class__.__name__
        self.safe_name = self.model_name.lower()

        if self.priors is None:
            raise ValueError(f"{self.model_name} must define class variable 'priors'")
        if self.feature_cols is None:
            raise ValueError(f"{self.model_name} must define class variable 'feature_cols'")

        self.plot_coeffs = list(self.priors.keys())

        self.base_dir = os.path.join("results", "linear_models", self.safe_name)
        self.training_dir = os.path.join(self.base_dir, "training")
        os.makedirs(self.training_dir, exist_ok=True)

        df_getter = getattr(self._database, "get_dataframe", None)
        if callable(df_getter):
            self.df_raw = df_getter().copy()
        else:
            self.df_raw = self._database.get_dataframe.copy()

        self.df_train = None
        self.X_scaled = None
        self.y = None
        self.scaler = None
        self.pm_model = None
        self.idata = None
        self.prior_idata = None

    # ==================================================
    # Data
    # ==================================================

    def _prepare_dataframe(self, df):
        return df

    def get_training_dataframe(self):
        df = self.df_raw.copy()
        df = self._prepare_dataframe(df)

        needed = list(self.feature_cols) + [self.target_col, self.class_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{self.model_name}: missing required columns: {missing}")

        df = df[needed].copy()

        for col in self.feature_cols + [self.target_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df[self.class_col].isin(["Survival", "Failure", 0, 1])].copy()
        df = df.dropna(subset=self.feature_cols + [self.target_col]).copy()
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(f"{self.model_name}: no usable rows left after preprocessing")

        return df

    def _preprocess(self, df):
        X = df[self.feature_cols].copy()
        y = pd.to_numeric(df[self.target_col], errors="coerce").astype(float)

        self.scaler = MinMaxScaler().set_output(transform="pandas")
        X_scaled = self.scaler.fit_transform(X)

        cls = df[self.class_col].replace({"Survival": 0, "Failure": 1})
        cls = pd.to_numeric(cls, errors="coerce").astype(int)
        X_scaled[self.class_col] = cls.values

        X_scaled.index = df.index
        y.index = df.index

        return X_scaled, y

    def save_scaler(self):
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(self.training_dir, "scaler.save"))

    def load_scaler(self):
        path = os.path.join(self.training_dir, "scaler.save")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No scaler found for {self.model_name}: {path}")
        self.scaler = joblib.load(path)
        return self.scaler

    # ==================================================
    # Equation hooks
    # ==================================================

    def core_equation(self, c, v):
        raise NotImplementedError

    def _formula_latex(self):
        return self.model_name

    # ==================================================
    # Fit / load
    # ==================================================

    def fit(self, redo=True):
        self.df_train = self.get_training_dataframe()
        self.X_scaled, self.y = self._preprocess(self.df_train)
        self.save_scaler()

        i_obs = self.y.astype(float).values
        is_survival = self.X_scaled[self.class_col].values == 0
        is_failure = self.X_scaled[self.class_col].values == 1

        with pm.Model() as model:
            v = {col: pm.Data(col, self.X_scaled[col].values) for col in self.feature_cols}
            c = {
                name: pm.Normal(name, mu=p["mu"], sigma=p["sigma"])
                for name, p in self.priors.items()
            }

            y_line = self.core_equation(c, v)
            pm.Deterministic("y", y_line)

            alpha = pm.Normal("alpha", mu=40, sigma=2)
            sigma = pm.HalfNormal("sigma_surv", sigma=0.20)

            pm.Censored(
                "y_surv",
                pm.Normal.dist(mu=y_line[is_survival], sigma=sigma),
                lower=-np.inf,
                upper=i_obs[is_survival],
                observed=i_obs[is_survival],
            )

            pm.SkewNormal(
                "y_fail",
                mu=y_line[is_failure],
                sigma=sigma,
                alpha=alpha,
                observed=i_obs[is_failure],
            )

            self.prior_idata = pm.sample_prior_predictive()

            if redo:
                posterior_idata = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    target_accept=self.target_accept,
                    progressbar=True,
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True},
                )

                posterior_predictive = pm.sample_posterior_predictive(
                    posterior_idata,
                    var_names=["y"],
                    return_inferencedata=True,
                )

                posterior_idata.extend(posterior_predictive)

                self.idata = self._combine_loglik(
                    posterior_idata,
                    is_survival=is_survival,
                    is_failure=is_failure,
                    new_name="y_total",
                )
                self._save_idata()
            else:
                self.idata = self._retrieve_idata()

        self.pm_model = model
        return self

    def _combine_loglik(self, idata, is_survival, is_failure, new_name="y_total"):
        ll_surv = idata.log_likelihood.get("y_surv", None)
        ll_fail = idata.log_likelihood.get("y_fail", None)

        if ll_surv is None or ll_fail is None:
            raise ValueError("Expected 'y_surv' and 'y_fail' in log_likelihood.")

        chain_dim, draw_dim = ll_surv.dims[0], ll_surv.dims[1]
        obs_dim_surv = [d for d in ll_surv.dims if d not in (chain_dim, draw_dim)][0]
        obs_dim_fail = [d for d in ll_fail.dims if d not in (chain_dim, draw_dim)][0]

        n = len(is_survival)

        combined = xr.DataArray(
            np.full((ll_surv.sizes[chain_dim], ll_surv.sizes[draw_dim], n), np.nan, dtype=float),
            dims=(chain_dim, draw_dim, "obs_dim"),
        )

        surv_idx = np.where(is_survival)[0]
        fail_idx = np.where(is_failure)[0]

        combined.loc[{"obs_dim": surv_idx}] = ll_surv.transpose(chain_dim, draw_dim, obs_dim_surv).values
        combined.loc[{"obs_dim": fail_idx}] = ll_fail.transpose(chain_dim, draw_dim, obs_dim_fail).values

        idata.log_likelihood[new_name] = combined
        return idata

    def _save_idata(self):
        path = os.path.join(self.training_dir, "idata.nc")
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass
        self.idata.to_netcdf(path)

    def _retrieve_idata(self):
        path = os.path.join(self.training_dir, "idata.nc")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No idata found for {self.model_name}: {path}")
        return az.from_netcdf(path)

    # ==================================================
    # Predictions / metrics
    # ==================================================

    def posterior_mean_prediction(self):
        y_post = self.idata.posterior["y"].stack(samples=("chain", "draw")).values
        if y_post.ndim == 1:
            return np.full(len(self.y), float(np.mean(y_post)))
        return y_post.mean(axis=1)

    def compute_logprobability(self, plot=True, export=False):
        if "log_likelihood" not in self.idata:
            return None

        loglik = self.idata.log_likelihood
        per_draw_totals = None
        per_obs_blocks = []

        for vname in loglik.data_vars:
            vals = loglik[vname].values
            if vals.ndim == 2:
                s = vals.reshape(-1, 1)
            else:
                s = vals.reshape(-1, vals.shape[-1])

            part_totals = s.sum(axis=1)
            per_draw_totals = part_totals if per_draw_totals is None else (per_draw_totals + part_totals)
            per_obs_blocks.append(s)

        per_obs_all = np.concatenate(per_obs_blocks, axis=1) if per_obs_blocks else np.empty((0, 0))
        logp_mean_per_obs = per_obs_all.mean(axis=0) if per_obs_all.size else np.array([])
        logp_total_per_draw = per_draw_totals if per_draw_totals is not None else np.array([])

        summary = {
            "mean_logprob_total": float(np.mean(logp_total_per_draw)) if logp_total_per_draw.size else np.nan,
            "std_logprob_total": float(np.std(logp_total_per_draw)) if logp_total_per_draw.size else np.nan,
            "mean_logprob_per_obs": float(np.mean(logp_mean_per_obs)) if logp_mean_per_obs.size else np.nan,
        }

        if plot and logp_total_per_draw.size:
            plt.figure(figsize=(10, 6))
            sns.histplot(logp_total_per_draw, bins=40, kde=True)
            plt.axvline(summary["mean_logprob_total"], linestyle="--")
            plt.title(f"{self.model_name}: posterior log-probability distribution")
            plt.grid(True)
            plt.tight_layout()

            if export:
                out = os.path.join(self.base_dir, f"{self.safe_name}_logprob_distribution.png")
                plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

        return summary

    def epic_metrics(self, export=True):
        y_true = self.y.values.astype(float)
        y_hat = self.posterior_mean_prediction()
        n = len(y_true)
        k = len(self.priors)

        resid = y_true - y_hat
        sse = float(np.sum(resid ** 2))
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        aic = n * np.log(max(sse / max(n, 1), 1e-300)) + 2 * k
        bic = n * np.log(max(sse / max(n, 1), 1e-300)) + k * np.log(max(n, 1))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
        r2 = 1.0 - sse / ss_tot
        adj_r2 = 1.0 - ((1.0 - r2) * (n - 1)) / max(n - k - 1, 1)

        fail_mask = self.X_scaled[self.class_col].values == 1
        y_fail = y_true[fail_mask]
        yhat_fail = y_hat[fail_mask]

        if len(y_fail) > 0:
            resid_f = y_fail - yhat_fail
            sse_f = float(np.sum(resid_f ** 2))
            rmse_f = float(np.sqrt(np.mean(resid_f ** 2)))
        else:
            rmse_f = np.nan

        out = {
            "RMSE": rmse,
            "AIC": float(aic),
            "BIC": float(bic),
            "R2": float(r2),
            "R2_Adjusted": float(adj_r2),
            "RMSE_failures": float(rmse_f) if np.isfinite(rmse_f) else np.nan,
        }

        lp = self.compute_logprobability(plot=False, export=False)
        if lp is not None:
            out["mean_logprob_total"] = lp.get("mean_logprob_total", np.nan)

        try:
            out["LOO"] = float(az.loo(self.idata, var_name="y_total").elpd_loo)
        except Exception:
            out["LOO"] = np.nan

        try:
            out["WAIC"] = float(az.waic(self.idata, var_name="y_total").elpd_waic)
        except Exception:
            out["WAIC"] = np.nan

        if export:
            with open(os.path.join(self.base_dir, f"{self.safe_name}_metrics.json"), "w") as f:
                json.dump(self._json_safe(out), f, indent=2)

        return out

    def classifier_plus_metrics(self, export=True):
        y_true_cls = self.X_scaled[self.class_col].values.astype(int)
        y_obs = self.y.values.astype(float)
        y_line = self.posterior_mean_prediction()

        y_pred_cls = (y_obs >= y_line).astype(int)

        tp = int(np.sum((y_true_cls == 1) & (y_pred_cls == 1)))
        tn = int(np.sum((y_true_cls == 0) & (y_pred_cls == 0)))
        fp = int(np.sum((y_true_cls == 0) & (y_pred_cls == 1)))
        fn = int(np.sum((y_true_cls == 1) & (y_pred_cls == 0)))

        total = tp + tn + fp + fn

        precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)

        try:
            mcc = matthews_corrcoef(y_true_cls, y_pred_cls)
        except Exception:
            mcc = np.nan

        acc = accuracy_score(y_true_cls, y_pred_cls) if total > 0 else np.nan

        out = {
            "counts": {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Total": total,
            },
            "Accuracy": float(acc) if np.isfinite(acc) else np.nan,
            "Precision": float(precision) if np.isfinite(precision) else np.nan,
            "Recall": float(recall) if np.isfinite(recall) else np.nan,
            "F1": float(f1) if np.isfinite(f1) else np.nan,
            "MCC": float(mcc) if np.isfinite(mcc) else np.nan,
        }

        if export:
            with open(os.path.join(self.base_dir, f"{self.safe_name}_classification_metrics.json"), "w") as f:
                json.dump(self._json_safe(out), f, indent=2)

        return out

    # ==================================================
    # Plotting
    # ==================================================

    def show_formula(self, export=False):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.axis("off")

        lines = []
        for name in self.plot_coeffs:
            post = self.idata.posterior[name].stack(sample=("chain", "draw")).values
            mean = float(np.mean(post))
            lo, hi = np.percentile(post, [2.5, 97.5])
            lines.append(f"{name} = {mean:.3f}  —  95% CrI [{lo:.3f}, {hi:.3f}]")

        equation_text = (
            f"{self.model_name}\n"
            f"--------------------------------\n\n"
            f"{self._formula_latex()}\n\n"
            + "\n".join(lines)
        )

        ax.text(0.5, 0.5, equation_text, fontsize=15, ha="center", va="center", transform=ax.transAxes)

        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_formula.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_all_features(self, export=False, n_points=300, show_ci=True):
        posterior = self.idata.posterior
        coeff_samples = {
            c: posterior[c].stack(sample=("chain", "draw")).values
            for c in self.plot_coeffs
        }
        n_draws = len(next(iter(coeff_samples.values())))

        for feat in self.feature_cols:
            x_obs_scaled = self.X_scaled.loc[self.y.index, feat].values
            y_obs = self.y.values
            x_grid = np.linspace(0, 1, n_points)

            grid_df = pd.DataFrame()
            for col in self.feature_cols:
                if col == feat:
                    grid_df[col] = x_grid
                else:
                    grid_df[col] = np.full_like(x_grid, self.X_scaled[col].mean())

            v_grid = {col: grid_df[col].values for col in self.feature_cols}

            coeff_arrays = {c: coeff_samples[c][:, None] for c in self.plot_coeffs}
            v_arrays = {f: v_grid[f][None, :] for f in self.feature_cols}
            y_preds = self.core_equation(coeff_arrays, v_arrays)

            if y_preds.shape != (n_draws, n_points):
                y_preds = np.broadcast_to(y_preds, (n_draws, n_points))

            y_mean = y_preds.mean(axis=0)
            y_lower, y_upper = np.percentile(y_preds, [2.5, 97.5], axis=0)

            prior_means = {c: self.priors[c]["mu"] for c in self.plot_coeffs}
            prior_pred = self.core_equation(prior_means, v_grid)
            if np.ndim(prior_pred) == 0:
                prior_pred = np.full_like(x_grid, prior_pred)

            fig, ax = plt.subplots(figsize=(10, 8))
            is_survival = self.X_scaled[self.class_col].values == 0
            is_failure = self.X_scaled[self.class_col].values == 1

            ax.scatter(x_obs_scaled[is_survival], y_obs[is_survival], color="black", alpha=0.5, s=30, marker="o", label="Survival")
            ax.scatter(x_obs_scaled[is_failure], y_obs[is_failure], color="red", alpha=1.0, s=60, marker="x", label="Failure", linewidth=2)

            if show_ci:
                ax.fill_between(x_grid, y_lower, y_upper, alpha=0.2, label="95% credible interval")
            ax.plot(x_grid, y_mean, lw=3, label="Posterior predictive mean")
            ax.plot(x_grid, prior_pred, linestyle="--", lw=1.5, label="Prior")

            ax.set_xlabel(f"{feat} (scaled)")
            ax.set_ylabel("i_c")
            ax.set_title(f"{self.model_name} — {feat}")
            ax.grid(True)
            ax.legend()

            if export:
                plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_{feat}_prediction_plot.png"), dpi=300, bbox_inches="tight")
            plt.close()

    def plot_residuals(self, export=False):
        y_obs = self.y.values
        y_mean = self.posterior_mean_prediction()
        residuals = y_mean - y_obs
        indx = np.arange(len(y_obs))

        is_survival = self.X_scaled[self.class_col].values == 0
        is_failure = self.X_scaled[self.class_col].values == 1

        plt.figure(figsize=(10, 6))
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.scatter(indx[is_survival], residuals[is_survival], color="black", marker="o", label="Survival")
        plt.scatter(indx[is_failure], residuals[is_failure], color="red", marker="x", label="Failure")
        plt.xlabel("Index number")
        plt.ylabel("Residual of i_c")
        plt.title(f"{self.model_name}: Residuals")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_residuals_plot.png"), dpi=300)
        plt.close()

    def plot_priors_posteriors(self, export=False):
        n_params = len(self.priors)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 4 * n_params), constrained_layout=True)

        if n_params == 1:
            axes = [axes]

        for ax, (param, prior_info) in zip(axes, self.priors.items()):
            mu, sigma = prior_info["mu"], prior_info["sigma"]
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
            y = scipy.stats.norm.pdf(x, mu, sigma)
            ax.plot(x, y, label=f"Prior N({mu}, {sigma})", linestyle="--", color="green")

            if param in self.idata.posterior:
                samples = self.idata.posterior[param].stack(samples=("chain", "draw")).values
                sns.kdeplot(samples, label="Posterior", fill=True, color="blue", ax=ax, alpha=0.5)

            ax.set_title(f"{self.model_name}: Prior and posterior - {param}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.grid(True)
            ax.legend()

        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_priors_posteriors_plot.png"), dpi=300)
        plt.close()

    def plot_rmse_distribution(self, export=False):
        y_true = self.y.values
        y_samples = self.idata.posterior["y"].stack(samples=("chain", "draw")).values

        if y_samples.ndim == 1:
            rmse_samples = np.array([np.sqrt(np.mean((y_samples - y_true.mean()) ** 2))])
        else:
            rmse_samples = np.sqrt(np.mean((y_samples - y_true[:, None]) ** 2, axis=0))

        plt.figure(figsize=(10, 6))
        sns.histplot(rmse_samples, kde=True, bins=30, stat="density")
        plt.axvline(np.mean(rmse_samples), linestyle="--", label=f"Mean RMSE: {np.mean(rmse_samples):.5f}")
        plt.xlabel("RMSE")
        plt.ylabel("Density")
        plt.title(f"{self.model_name} - RMSE distribution of posterior")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_rmse_distribution.png"), dpi=300)
        plt.close()

    def plot_arviz_stuff(self, export=False):
        az.plot_trace(
            self.idata,
            var_names=[v for v in self.idata.posterior.data_vars if self.idata.posterior[v].ndim == 2],
            kind="rank_vlines",
        )
        plt.subplots_adjust(hspace=1, wspace=0.2)
        plt.suptitle(f"Trace plot for {self.model_name}")
        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_traceplot.png"), dpi=300, bbox_inches="tight")
        plt.close()

        az.plot_energy(self.idata)
        plt.suptitle(f"Energy plot for {self.model_name}")
        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_energyplot.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plotall(self, export_all=True):
        self.show_formula(export=export_all)
        self.plot_arviz_stuff(export=export_all)
        self.plot_all_features(export=export_all)
        self.plot_residuals(export=export_all)
        self.plot_priors_posteriors(export=export_all)
        self.plot_rmse_distribution(export=export_all)
        self.epic_metrics(export=export_all)
        self.classifier_plus_metrics(export=export_all)

    # ==================================================
    # Helpers
    # ==================================================

    def _json_safe(self, x):
        if isinstance(x, dict):
            return {k: self._json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [self._json_safe(v) for v in list(x)]
        if isinstance(x, (float, np.floating)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x


class BayesianModel:
    """
    Manager / trainer class.

    Usage:
        trainer = BayesianModel(db, selector="test")
        model = trainer.train("LDaq_linear", plot_all=True, export_all=True)
    """

    def __init__(
        self,
        database: Database,
        selector="test",
        target_col="Global_gradient",
        class_col="Survival/failure",
        draws=1000,
        tune=2000,
        target_accept=0.95,
    ):
        self._database = database
        self.selector = selector
        self.target_col = target_col
        self.class_col = class_col
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.last_model = None

    def _get_registry(self):
        linear_models = importlib.import_module("linear_models.linear_models")
        if not hasattr(linear_models, "get_linear_model_registry"):
            raise ValueError("linear_models.py must define get_linear_model_registry()")
        return linear_models.get_linear_model_registry()

    def available_models(self):
        return list(self._get_registry().keys())

    def train(self, model_name, redo=True, plot_all=True, export_all=True):
        registry = self._get_registry()

        if model_name not in registry:
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {list(registry.keys())}"
            )

        model_cls = registry[model_name]

        model = model_cls(
            database=self._database,
            selector=self.selector,
            export_all=export_all,
            target_col=self.target_col,
            class_col=self.class_col,
            draws=self.draws,
            tune=self.tune,
            target_accept=self.target_accept,
        )

        model.fit(redo=redo)

        if plot_all:
            model.plotall(export_all=export_all)
        else:
            model.epic_metrics(export=export_all)
            model.classifier_plus_metrics(export=export_all)

        self.last_model = model
        return model

    def train_many(self, model_names=None, redo=True, plot_all=True, export_all=True, continue_on_error=True):
        registry = self._get_registry()

        if model_names is None:
            model_names = list(registry.keys())

        results = {}

        for model_name in model_names:
            print(f"Training {model_name}...")
            try:
                self.train(
                    model_name=model_name,
                    redo=redo,
                    plot_all=plot_all,
                    export_all=export_all,
                )
                results[model_name] = "done"
            except Exception as e:
                results[model_name] = f"failed: {e}"
                print(f"{model_name} failed: {e}")
                if not continue_on_error:
                    raise

        return results