from Database_object import Database

import os
import json
import math
import itertools
import logging
import importlib

import joblib
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from matplotlib.colors import LinearSegmentedColormap

logging.getLogger("pymc").setLevel(logging.ERROR)


class BaseBayesianLogisticModel:
    """
    Base class for one Bayesian logistic model definition.

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
        class_col="Survival/failure",
        draws=1000,
        tune=2000,
        target_accept=0.95,
        prediction_threshold=0.5,
    ):
        self._database = database
        self.selector = selector
        self.export_all = export_all
        self.class_col = class_col

        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.prediction_threshold = prediction_threshold

        self.model_name = self.__class__.__name__
        self.safe_name = self.model_name.lower()

        if self.priors is None:
            raise ValueError(f"{self.model_name} must define class variable 'priors'")
        if self.feature_cols is None:
            raise ValueError(f"{self.model_name} must define class variable 'feature_cols'")

        self.plot_coeffs = list(self.priors.keys())

        self.base_dir = os.path.join("results", "logistic_models", self.safe_name)
        self.training_dir = os.path.join(self.base_dir, "training")
        os.makedirs(self.training_dir, exist_ok=True)

        df_getter = getattr(self._database, "get_dataframe", None)
        if callable(df_getter):
            self.df_raw = df_getter().copy()
        else:
            self.df_raw = self._database.get_dataframe.copy()

        self.df_train = None
        self.X = None
        self.X_scaled = None
        self.y = None
        self.scaler = None

        self.idata = None
        self.posterior_idata = None
        self.prior_idata = None
        self.pm_model = None

        self.mean_p_failure_prior = None
        self.y_pred_class_prior = None
        self.mean_p_failure_posterior = None
        self.y_pred_class_posterior = None

    # ==================================================
    # Data prep
    # ==================================================

    def _prepare_dataframe(self, df):
        return df

    def get_training_dataframe(self):
        df = self.df_raw.copy()
        df = self._prepare_dataframe(df)

        needed = list(self.feature_cols) + [self.class_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{self.model_name}: missing required columns: {missing}")

        df = df[needed].copy()

        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        target_map = {"Survival": 0, "Failure": 1}
        df[self.class_col] = df[self.class_col].replace(target_map)
        df[self.class_col] = pd.to_numeric(df[self.class_col], errors="coerce")

        df = df.dropna(subset=self.feature_cols + [self.class_col]).copy()
        df[self.class_col] = df[self.class_col].astype(int)
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(f"{self.model_name}: no usable rows left after preprocessing")

        return df

    def _preprocess(self, df):
        X = df[self.feature_cols].copy()
        y = df[self.class_col].astype(int).copy()

        self.scaler = MinMaxScaler().set_output(transform="pandas")
        X_scaled = self.scaler.fit_transform(X)

        self.X = X.copy()
        self.X_scaled = X_scaled.copy()
        self.y = y.copy()

        self.save_scaler()
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
    # Model equation hooks
    # ==================================================

    def core_equation(self, c, v):
        """
        Must return mu_logit.
        """
        raise NotImplementedError

    def _mu_logit(self, v, c):
        return self.core_equation(c, v)

    def _formula_latex(self):
        return self.model_name

    # ==================================================
    # Fit / load
    # ==================================================

    def fit(self, redo=True):
        self.df_train = self.get_training_dataframe()
        self.X_scaled, self.y = self._preprocess(self.df_train)

        with pm.Model() as model:
            v = {col: pm.Data(col, self.X_scaled[col].values) for col in self.feature_cols}
            c = {
                name: pm.Normal(name, mu=p["mu"], sigma=p["sigma"])
                for name, p in self.priors.items()
            }

            mu_logit = self._mu_logit(v, c)
            pm.Deterministic("mu_logit", mu_logit)

            p_failure = pm.Deterministic("p_failure", pm.math.sigmoid(mu_logit))
            pm.Bernoulli("y", p=p_failure, observed=self.y.values)

            self.prior_idata = pm.sample_prior_predictive(samples=500)

            if redo:
                self.posterior_idata = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    target_accept=self.target_accept,
                    progressbar=True,
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True},
                )

                posterior_predictive = pm.sample_posterior_predictive(
                    self.posterior_idata,
                    var_names=["y", "p_failure", "mu_logit"],
                    return_inferencedata=True,
                )
                self.posterior_idata.extend(posterior_predictive)

                self.idata = self.posterior_idata
                self._save_idata()
            else:
                self.idata = self._retrieve_idata()

        self.pm_model = model
        self._calculate_all_predictions(threshold=self.prediction_threshold)
        return self

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
    # Predictions
    # ==================================================

    def _calculate_all_predictions(self, threshold=None):
        if threshold is None:
            threshold = self.prediction_threshold

        if self.prior_idata is not None:
            if hasattr(self.prior_idata, "prior") and "p_failure" in self.prior_idata.prior:
                pf = self.prior_idata.prior["p_failure"].stack(sample=("chain", "draw")).values
                if pf.ndim == 1:
                    self.mean_p_failure_prior = pf
                else:
                    self.mean_p_failure_prior = pf.mean(axis=1)
                self.y_pred_class_prior = (self.mean_p_failure_prior >= threshold).astype(int)
            elif hasattr(self.prior_idata, "prior_predictive") and "y" in self.prior_idata.prior_predictive:
                arr = self.prior_idata.prior_predictive["y"].values
                arr_flat = arr.reshape(-1, arr.shape[-1])
                self.mean_p_failure_prior = np.mean(arr_flat, axis=0)
                self.y_pred_class_prior = (self.mean_p_failure_prior >= threshold).astype(int)

        if self.idata is not None:
            if "p_failure" in self.idata.posterior:
                pf = self.idata.posterior["p_failure"].stack(sample=("chain", "draw")).values
                if pf.ndim == 1:
                    self.mean_p_failure_posterior = pf
                else:
                    self.mean_p_failure_posterior = pf.mean(axis=1)
            elif hasattr(self.idata, "posterior_predictive") and "y" in self.idata.posterior_predictive:
                arr = self.idata.posterior_predictive["y"].values
                arr_flat = arr.reshape(-1, arr.shape[-1])
                self.mean_p_failure_posterior = np.mean(arr_flat, axis=0)

            if self.mean_p_failure_posterior is not None:
                self.y_pred_class_posterior = (self.mean_p_failure_posterior >= threshold).astype(int)

    def posterior_mean_probability(self):
        if self.mean_p_failure_posterior is None:
            self._calculate_all_predictions()
        return self.mean_p_failure_posterior

    # ==================================================
    # Metrics
    # ==================================================

    def collect_metrics(self, export=True):
        y_true = self.y.values.astype(int)
        p_pred = np.asarray(self.mean_p_failure_posterior) if self.mean_p_failure_posterior is not None else None
        y_pred = np.asarray(self.y_pred_class_posterior) if self.y_pred_class_posterior is not None else None

        out = {}

        try:
            total_ll = 0.0
            if hasattr(self.idata, "log_likelihood"):
                for var in self.idata.log_likelihood.data_vars:
                    vals = self.idata.log_likelihood[var].stack(sample=("chain", "draw")).values
                    total_ll += float(np.nanmean(vals))
            out["mean_logprob_total"] = total_ll
        except Exception:
            out["mean_logprob_total"] = np.nan

        if y_pred is not None:
            out["Accuracy_all"] = float(accuracy_score(y_true, y_pred))
            out["Precision_all"] = float(precision_score(y_true, y_pred, zero_division=0))
            out["Recall_all"] = float(recall_score(y_true, y_pred, zero_division=0))
            out["F1_score_all"] = float(f1_score(y_true, y_pred, zero_division=0))
            try:
                out["MCC_all"] = float(matthews_corrcoef(y_true, y_pred))
            except Exception:
                out["MCC_all"] = np.nan
        else:
            out["Accuracy_all"] = np.nan
            out["Precision_all"] = np.nan
            out["Recall_all"] = np.nan
            out["F1_score_all"] = np.nan
            out["MCC_all"] = np.nan

        try:
            if p_pred is not None and len(np.unique(y_true)) > 1:
                out["ROC_AUC_all"] = float(roc_auc_score(y_true, p_pred))
            else:
                out["ROC_AUC_all"] = np.nan
        except Exception:
            out["ROC_AUC_all"] = np.nan

        try:
            if p_pred is not None:
                out["Log_loss_all"] = float(log_loss(y_true, p_pred, labels=[0, 1]))
            else:
                out["Log_loss_all"] = np.nan
        except Exception:
            out["Log_loss_all"] = np.nan

        try:
            out["WAIC"] = float(az.waic(self.idata).elpd_waic)
        except Exception:
            out["WAIC"] = np.nan

        try:
            out["LOO"] = float(az.loo(self.idata).elpd_loo)
        except Exception:
            out["LOO"] = np.nan

        if export:
            with open(os.path.join(self.base_dir, f"{self.safe_name}_metrics.json"), "w") as f:
                json.dump(self._json_safe(out), f, indent=2, allow_nan=False)

        return out

    # ==================================================
    # Plotting
    # ==================================================

    def show_formula(self, export=True):
        formula = self._formula_latex()
        equation_text = f"{self.model_name}\n--------------------------------\n\n{formula}\n"

        for name in self.plot_coeffs:
            post = self.idata.posterior[name].stack(sample=("chain", "draw")).values
            mean = float(np.mean(post))
            lo, hi = np.percentile(post, [2.5, 97.5])
            equation_text += f"\n{name} = {mean:.3f}  —  95% CrI [{lo:.3f}, {hi:.3f}]"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        ax.text(0.5, 0.5, equation_text, fontsize=15, ha="center", va="center", transform=ax.transAxes)

        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_formula.png"), bbox_inches="tight", dpi=300)
        plt.close()

    def plot_arviz_stuff(self, export=True):
        az.plot_trace(
            self.idata,
            var_names=[v for v in self.idata.posterior.data_vars if self.idata.posterior[v].ndim == 2],
            kind="rank_vlines",
        )
        plt.subplots_adjust(hspace=1, wspace=0.2)
        plt.suptitle(f"Trace plot for {self.model_name}")
        if export:
            plt.savefig(os.path.join(self.training_dir, f"{self.safe_name}_traceplot.png"), dpi=300, bbox_inches="tight")
        plt.close()

        az.plot_energy(self.idata)
        plt.suptitle(f"Energy plot for {self.model_name}")
        if export:
            plt.savefig(os.path.join(self.training_dir, f"{self.safe_name}_energyplot.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_priors_posteriors_of_coefficients(self, export=True):
        coeffs = list(self.priors.keys())
        n_coeffs = len(coeffs)
        ncols = 2 if n_coeffs > 1 else 1
        nrows = math.ceil(n_coeffs / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, coeff in zip(axes, coeffs):
            mu = self.priors[coeff]["mu"]
            sigma = self.priors[coeff]["sigma"]
            x_vals = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)

            posterior_samples = self.idata.posterior[coeff].values.flatten()
            sns.kdeplot(posterior_samples, fill=True, label="Posterior", color="blue", ax=ax, alpha=0.5)

            prior_vals = scipy.stats.norm.pdf(x_vals, loc=mu, scale=sigma)
            ax.plot(x_vals, prior_vals, "g--", label=f"Prior N({mu}, {sigma})")
            ax.set_title(f"Prior vs Posterior for {coeff}")
            ax.set_xlabel(coeff)
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True)

        for ax in axes[len(coeffs):]:
            ax.axis("off")

        fig.tight_layout()
        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_priors_vs_posteriors.png"), dpi=300)
        plt.close()

    def plot_logistic_fit(self, export=True):
        posterior = self.idata.posterior
        c_vals = {name: posterior[name].mean(dim=("chain", "draw")).values for name in self.plot_coeffs}
        formula = self._formula_latex()

        for feat in self.feature_cols:
            fixed_vals = self.X_scaled.mean()
            x_vals = np.linspace(self.X_scaled[feat].min(), self.X_scaled[feat].max(), 200)

            vars_dict = {col: np.full_like(x_vals, fixed_vals[col], dtype=float) for col in self.feature_cols}
            vars_dict[feat] = x_vals

            mu_vals = self._mu_logit(vars_dict, c_vals)
            p_vals = 1 / (1 + np.exp(-mu_vals))

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(self.X_scaled.loc[self.y == 0, feat], self.y[self.y == 0], color="black", marker="o", alpha=0.7, label="Survival")
            ax.scatter(self.X_scaled.loc[self.y == 1, feat], self.y[self.y == 1], color="red", marker="x", alpha=0.7, label="Failure")

            ax.plot(x_vals, p_vals, lw=2, label="Predicted probability")
            ax.axhline(0.5, color="red", ls="--", lw=1, label="Decision boundary")

            ax.set_xlabel(f"{feat} (scaled 0–1)")
            ax.set_ylabel("Probability of failure")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{self.model_name} — {feat}")
            ax.grid(True)
            ax.legend()

            coeff_text = "\n".join([f"{c} = {float(np.mean(c_vals[c])):.3f}" for c in self.plot_coeffs])
            ax.text(
                0.0, -0.25, f"{formula}", fontsize=13, ha="left", va="top",
                transform=ax.transAxes, family="monospace",
                bbox=dict(facecolor="white", alpha=0.9, boxstyle="square,pad=0.7")
            )
            ax.text(
                0.0, -0.38, coeff_text, fontsize=11, ha="left", va="top",
                transform=ax.transAxes, family="monospace",
                bbox=dict(facecolor="white", alpha=0.0, boxstyle="square,pad=0.7", linestyle="dotted")
            )

            if export:
                plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_logistic_fit_{feat}.png"), dpi=300, bbox_inches="tight")
            plt.close()

    def plot_contours(self, export=True, bonus_text=False):
        if len(self.feature_cols) < 2:
            return

        special_boy_cmap = LinearSegmentedColormap.from_list("rg", ["g", "w", "r"], N=256)
        posterior = self.idata.posterior
        c_vals = {name: posterior[name].mean(dim=("chain", "draw")).values for name in self.plot_coeffs}
        formula = self._formula_latex()

        X_scaled = self.X_scaled
        fixed_vals = X_scaled.mean()
        i = 0

        fmaps = {
            "Water_level_diff": "Water level difference",
            "Seepage_length": "Seepage length",
            "Friction_angle": "Friction angle",
            "Aquifer_thickness": "Aquifer thickness",
            "Blanket_thickness": "Blanket thickness",
            "Uniformity_coefficient": "Uniformity coefficient",
            "Hydraulic_conductivity_KC": "Hydraulic conductivity",
            "Bedding_angle": "Bedding angle",
        }

        for x_var, y_var in itertools.combinations(self.feature_cols, 2):
            pad = 0.03
            x_vals = np.linspace(X_scaled[x_var].min() - pad, X_scaled[x_var].max() + pad, 100)
            y_vals = np.linspace(X_scaled[y_var].min() - pad, X_scaled[y_var].max() + pad, 100)
            X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

            vars_dict = {col: np.full_like(X_grid, fixed_vals[col], dtype=float) for col in self.feature_cols}
            vars_dict[x_var] = X_grid
            vars_dict[y_var] = Y_grid

            mu_grid = self._mu_logit(vars_dict, c_vals)
            p_grid = 1 / (1 + np.exp(-mu_grid))

            fig, ax = plt.subplots(figsize=(6, 5))
            cf = ax.contourf(X_grid, Y_grid, p_grid, levels=np.linspace(0, 1, 21), cmap=special_boy_cmap, alpha=0.6)
            plt.colorbar(cf, ax=ax, label="Failure probability [-]")
            ax.contour(X_grid, Y_grid, p_grid, levels=[0.5], colors="red", linestyles="dashed", linewidths=3)

            ax.scatter(X_scaled[self.y == 0][x_var], X_scaled[self.y == 0][y_var], marker=".", color="black", label="Observed survivals")
            ax.scatter(X_scaled[self.y == 1][x_var], X_scaled[self.y == 1][y_var], marker="x", color="black", label="Observed failures")

            ax.set_xlabel(f"{fmaps.get(x_var, x_var)} (normalized)")
            ax.set_ylabel(f"{fmaps.get(y_var, y_var)} (normalized)")
            ax.set_title(f"Failure probability for the {self.model_name} model", pad=20)
            ax.grid(True)
            ax.legend(loc="best")
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)

            if bonus_text:
                coeff_text = "\n".join([f"{c} = {float(np.mean(c_vals[c])):.3f}" for c in self.plot_coeffs])
                ax.text(
                    0.0, -0.30, f"{formula}",
                    fontsize=13, ha="left", va="top",
                    transform=ax.transAxes, family="monospace",
                    bbox=dict(facecolor="white", alpha=0.9, boxstyle="square,pad=0.7")
                )
                ax.text(
                    0.0, -0.42, coeff_text,
                    fontsize=11, ha="left", va="top",
                    transform=ax.transAxes, family="monospace",
                    bbox=dict(facecolor="white", alpha=0.0, boxstyle="square,pad=0.7", linestyle="dotted")
                )

            if export:
                outdir = os.path.join(self.base_dir, "contours")
                os.makedirs(outdir, exist_ok=True)
                plt.savefig(os.path.join(outdir, f"{self.safe_name}_contour_scaled_{i}.png"), dpi=300, bbox_inches="tight")
            plt.close()
            i += 1

    def plot_confusion_matrix(self, threshold=None, export=True):
        if threshold is None:
            threshold = self.prediction_threshold
        else:
            self._calculate_all_predictions(threshold=threshold)

        if self.y_pred_class_posterior is None:
            print("No posterior class predictions available.")
            return

        y_obs = self.y.values
        labels = ["Survival", "Failure"]

        cm_posterior = confusion_matrix(y_obs, self.y_pred_class_posterior)
        cm_posterior_pct = cm_posterior / max(cm_posterior.sum(), 1) * 100

        cm_actual = confusion_matrix(y_obs, y_obs)
        cm_actual_pct = cm_actual / max(cm_actual.sum(), 1) * 100
        cm_diff_pct = cm_posterior_pct - cm_actual_pct

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        grey_cmap = LinearSegmentedColormap.from_list("lightgrey", ["#bfbfbf", "white"], N=256)

        ConfusionMatrixDisplay(confusion_matrix=cm_posterior_pct, display_labels=labels).plot(
            cmap=plt.cm.Blues, ax=axes[0], values_format=".1f", colorbar=False
        )
        axes[0].set_title("Posterior Confusion Matrix (%)")

        ConfusionMatrixDisplay(confusion_matrix=cm_actual_pct, display_labels=labels).plot(
            cmap=grey_cmap, ax=axes[1], values_format=".1f", colorbar=False
        )
        axes[1].set_title("Observed Distribution (%)")

        ConfusionMatrixDisplay(confusion_matrix=cm_diff_pct, display_labels=labels).plot(
            cmap=grey_cmap, ax=axes[2], values_format="+.1f", colorbar=False
        )
        axes[2].set_title("Posterior - Observed (%)")

        fig.suptitle(f"{self.model_name} confusion matrices (threshold = {threshold})")
        fig.tight_layout()

        if export:
            plt.savefig(os.path.join(self.base_dir, f"{self.safe_name}_confusion_matrices.png"), dpi=300)
        plt.close()

    def plotall(self, export_all=True):
        self._calculate_all_predictions(threshold=self.prediction_threshold)
        self.show_formula(export=export_all)
        self.plot_arviz_stuff(export=export_all)
        self.plot_priors_posteriors_of_coefficients(export=export_all)
        self.plot_logistic_fit(export=export_all)
        self.plot_contours(export=export_all)
        self.plot_confusion_matrix(export=export_all)
        self.collect_metrics(export=export_all)

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


class BayesianLogisticTrainer:
    """
    Manager / trainer class.

    Usage:
        trainer = BayesianLogisticTrainer(db, selector="test")
        model = trainer.train("HLU_log", plot_all=True, export_all=True)
    """

    def __init__(
        self,
        database: Database,
        selector="test",
        class_col="Survival/failure",
        draws=1000,
        tune=2000,
        target_accept=0.95,
        prediction_threshold=0.5,
    ):
        self._database = database
        self.selector = selector
        self.class_col = class_col
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.prediction_threshold = prediction_threshold
        self.last_model = None

    def _get_registry(self):
        logistic_stuff = importlib.import_module("logistic_models.logistic_models")
        if not hasattr(logistic_stuff, "get_logistic_model_registry"):
            raise ValueError("logistic_stuff.py must define get_logistic_model_registry()")
        return logistic_stuff.get_logistic_model_registry()

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
            class_col=self.class_col,
            draws=self.draws,
            tune=self.tune,
            target_accept=self.target_accept,
            prediction_threshold=self.prediction_threshold,
        )

        model.fit(redo=redo)

        if plot_all:
            model.plotall(export_all=export_all)
        else:
            model.collect_metrics(export=export_all)

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