"""Statistics, plotting, and model-comparison utilities for the BEP database.

This version preserves the original behaviour while making the code easier to
read with added comments and docstrings.
"""
try:
    from src.Database_object import Database
except Exception:
    from Database_object import Database

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, math
from scipy.stats import gaussian_kde

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample, class_weight
from matplotlib.ticker import FuncFormatter

import json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, accuracy_score
)


class Statistics:
    """Generate descriptive statistics and diagnostic plots for the database."""
    def __init__(self, database: Database, selector='test', export_all=True):
        """Initialize the object and prepare the main internal state."""
        self._database = database

        # robust dataframe getter
        if callable(getattr(self._database, "get_dataframe", None)):
            self.df = self._database.get_dataframe().copy()
        else:
            self.df = self._database.get_dataframe.copy()

        self.features = list(database._get_modelling_features())
        self.selector = selector
        self.export_all = export_all
        self.y = "Global_gradient"

        self.log_scale_features = {
            'd10', 'd50', 'd60', 'd70',
            'Hydraulic_conductivity', 'Hydraulic_conductivity_KC'
        }

        # All statistics output is grouped by selector so separate runs do not overwrite each other.
        self.base_dir = f"results/statistics/database-statistics_{self.selector}"
        self.hist_dir = os.path.join(self.base_dir, "histograms")
        self.kde_dir = os.path.join(self.base_dir, "kde_plots")
        self.scatter_dir = os.path.join(self.base_dir, "scatter_plots")
        self.pair_dir = self.base_dir
        self.corr_dir = os.path.join(self.base_dir, "correlations")

        for d in [self.base_dir, self.hist_dir, self.kde_dir, self.scatter_dir, self.pair_dir, self.corr_dir]:
            os.makedirs(d, exist_ok=True)

        # keep only features that actually exist
        self.features = [f for f in self.features if f in self.df.columns]
        self.num_features = len(self.features)

        # force features + y to numeric
        for col in self.features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        if self.y in self.df.columns:
            self.df[self.y] = pd.to_numeric(self.df[self.y], errors="coerce")

        # imputed-only file
        imputed_only_path = f"src/bayesian_network_results/Imputed_only_{self.selector}_with_{len(self.df)}_rows.pkl"
        if os.path.exists(imputed_only_path):
            self.imputed_only = pd.read_pickle(imputed_only_path).copy()
        else:
            self.imputed_only = pd.DataFrame(index=self.df.index)

        for col in self.features:
            if col not in self.imputed_only.columns:
                self.imputed_only[col] = np.nan
            self.imputed_only[col] = pd.to_numeric(self.imputed_only[col], errors="coerce")

        self.cols = 3
        self.rows = max(1, math.ceil(max(1, self.num_features) / self.cols))
        self.fig_height = max(5, self.rows * 4)

    @staticmethod
    def _labelname(name):
        """Return a cleaner display label for a feature or metric name."""
        to_real_name = {
            "Water_level_diff": "Water level difference",
            "Aquifer_thickness": "Aquifer thickness",
            "Blanket_thickness": "Blanket thickness",
            "Seepage_length": "Seepage length",
            "Hydraulic_conductivity_KC": "Hydraulic conductivity",
            "Hydraulic_conductivity": "Hydraulic conductivity",
            "Uniformity_coefficient": "Uniformity coefficient",
            "Friction_angle": "Friction angle",
            "Bedding_angle": "Bedding angle",
        }
        return to_real_name.get(name, name)

    @staticmethod
    def _labelunits(name):
        """Return display units for a given feature when available."""
        to_units = {
            "Water_level_diff": "Water level difference [m]",
            "Aquifer_thickness": "Aquifer thickness [m]",
            "Blanket_thickness": "Blanket thickness [m]",
            "Seepage_length": "Seepage length [m]",
            "Hydraulic_conductivity_KC": "Hydraulic conductivity [m/s]",
            "Hydraulic_conductivity": "Hydraulic conductivity [m/s]",
            "Uniformity_coefficient": "Uniformity coefficient [-]",
            "Porosity": "Porosity [-]",
            "Friction_angle": "Friction angle [°]",
            "Bedding_angle": "Bedding angle [°]",
            "d10": "d10 [m]",
            "d50": "d50 [m]",
            "d60": "d60 [m]",
            "d70": "d70 [m]",
        }
        return to_units.get(name, name)

    @staticmethod
    def _safe_filename(name):
        """Convert a label into a filesystem-safe filename."""
        return (
            str(name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

    def _valid_series(self, col, positive_only=False):
        """Return a cleaned numeric series that is safe to plot or analyse."""
        if col not in self.df.columns:
            return pd.Series(dtype=float)

        s = pd.to_numeric(self.df[col], errors="coerce").dropna()
        if positive_only:
            s = s[s > 0]
        return s

    def _class_series(self, col, klass, positive_only=False):
        """Return the class/target series used in classification-style plots."""
        if col not in self.df.columns or 'Survival/failure' not in self.df.columns:
            return pd.Series(dtype=float)

        s = pd.to_numeric(
            self.df.loc[self.df['Survival/failure'] == klass, col],
            errors="coerce"
        ).dropna()

        if positive_only:
            s = s[s > 0]
        return s

    def _imputed_series(self, col, positive_only=False):
        """Return the series that marks which values were imputed."""
        if col not in self.imputed_only.columns:
            return pd.Series(dtype=float)

        s = pd.to_numeric(self.imputed_only[col], errors="coerce").dropna()
        if positive_only:
            s = s[s > 0]
        return s

    def _feature_has_variation(self, arr):
        """Check whether a feature has enough variation for plotting or statistics."""
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return False
        return np.nanstd(arr) > 0 and np.unique(arr).size > 1

    def _get_bins(self, col, data):
        """Create a sensible set of bins for histogram-style plots."""
        data = pd.to_numeric(data, errors="coerce").dropna()
        if len(data) == 0:
            return None

        if col in self.log_scale_features:
            data = data[data > 0]
            if len(data) == 0:
                return None
            dmin, dmax = data.min(), data.max()
            if dmin == dmax:
                return np.logspace(np.log10(dmin) - 0.25, np.log10(dmax) + 0.25, 10)
            return np.logspace(np.log10(dmin), np.log10(dmax), 20)

        dmin, dmax = data.min(), data.max()
        if dmin == dmax:
            width = max(abs(dmin) * 0.05, 0.5)
            return np.linspace(dmin - width, dmax + width, 10)

        return np.linspace(dmin, dmax, 20)

    def quickreport(self):
        """Print a quick textual overview of the dataframe and feature availability."""
        n_total = len(self.df)
        n_fail = int((self.df['Survival/failure'] == 'Failure').sum()) if 'Survival/failure' in self.df.columns else 0

        if self.num_features > 0:
            filled_fraction = self.df[self.features].notna().sum().sum() / (len(self.df) * len(self.features))
            filled_pct = 100 * filled_fraction
        else:
            filled_pct = np.nan

        print(f"There are {n_fail} failures among {n_total} total entries.")
        print(f"There is {filled_pct:.2f}% filled in of modelling features.\n")

    def run_all_statistics(
        self,
        do_quickreport=True,
        do_histograms=True,
        do_kde_plots=True,
        do_scatter_plots=True,
        do_pair_plots=True,
        do_correlation_matrix=True,
        continue_on_error=True,
    ):
        """Run the main statistics and plotting routines in sequence."""
        print("\nRunning statistics...\n")

        tasks = {
            "quickreport": (do_quickreport, self.quickreport),
            "histograms": (do_histograms, self.histograms),
            "kde_plots": (do_kde_plots, self.KDE_plots),
            "scatter_plots": (do_scatter_plots, self.scatter_plots),
            "pair_plots": (do_pair_plots, self.pair_plots),
            "correlation_matrix": (do_correlation_matrix, self.correlation_matrix),
        }

        results = {}

        for task_name, (enabled, task_function) in tasks.items():
            if not enabled:
                results[task_name] = "skipped"
                continue

            print(f"Running {task_name}...")
            try:
                task_function()
                results[task_name] = "done"
            except Exception as e:
                results[task_name] = f"failed: {e}"
                print(f"{task_name} failed: {e}")
                if not continue_on_error:
                    raise

        print("\nStatistics run finished.\n")
        return results

    ##################################################
    #             Histograms of features             #
    ##################################################

    def histograms(self):
        """Generate histograms for the selected features."""
        if self.num_features == 0:
            print("No modelling features available for histograms.")
            return None

        fig, axes = plt.subplots(self.rows, self.cols, figsize=(self.cols * 5, self.fig_height))
        axes = np.atleast_1d(axes).flatten()

        for i, col in enumerate(self.features):
            ax = axes[i]

            positive_only = col in self.log_scale_features
            all_data = self._valid_series(col, positive_only=positive_only)
            surv = self._class_series(col, 'Survival', positive_only=positive_only)
            fail = self._class_series(col, 'Failure', positive_only=positive_only)
            imp = self._imputed_series(col, positive_only=positive_only)

            if len(all_data) == 0:
                ax.set_title(f"{self._labelname(col)} (no data)")
                ax.axis("off")
                continue

            bins = self._get_bins(col, all_data)
            if bins is None:
                ax.set_title(f"{self._labelname(col)} (invalid data)")
                ax.axis("off")
                continue

            datasets = []
            colors = []
            labels = []

            if len(fail) > 0:
                datasets.append(fail)
                colors.append("tab:red")
                labels.append("Failure")
            if len(surv) > 0:
                datasets.append(surv)
                colors.append("darkseagreen")
                labels.append("Survival")
            if len(imp) > 0:
                datasets.append(imp)
                colors.append("lightblue")
                labels.append("Imputed")

            if len(datasets) == 0:
                ax.set_title(f"{self._labelname(col)} (no data)")
                ax.axis("off")
                continue

            ax.hist(
                datasets,
                bins=bins,
                stacked=True,
                color=colors,
                edgecolor='black',
                alpha=0.95,
                label=labels
            )

            if col in self.log_scale_features:
                ax.set_xscale('log')

            ax.set_title(self._labelname(col), fontsize=12)
            ax.set_xlabel(self._labelunits(col), fontsize=11)
            ax.set_ylabel("Instances", fontsize=11)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

            if self.export_all:
                fig_single, ax_single = plt.subplots(figsize=(6, 4.5))
                ax_single.hist(
                    [x for x in [fail, surv, imp] if len(x) > 0],
                    bins=bins,
                    stacked=True,
                    color=[c for x, c in zip([fail, surv, imp], ["tab:red", "darkseagreen", "lightblue"]) if len(x) > 0],
                    edgecolor='black',
                    alpha=0.95,
                    label=[l for x, l in zip([fail, surv, imp], ["Failure", "Survival", "Imputed"]) if len(x) > 0]
                )
                if col in self.log_scale_features:
                    ax_single.set_xscale('log')
                ax_single.set_title(self._labelname(col))
                ax_single.set_xlabel(self._labelunits(col))
                ax_single.set_ylabel("Instances")
                ax_single.grid(True, which='both', linestyle='--', alpha=0.4)
                ax_single.legend()
                fig_single.tight_layout()
                fig_single.savefig(os.path.join(self.hist_dir, f"{self._safe_filename(col)}.png"), dpi=300)
                plt.close(fig_single)

        for j in range(self.num_features, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Histograms of modelling features", fontsize=15, y=1.02)
        fig.tight_layout()

        if self.export_all:
            fig.savefig(os.path.join(self.base_dir, "all_histograms.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return None

    ##################################################
    #                   KDE plots                    #
    ##################################################

    def KDE_plots(self):
        """Generate KDE plots for the selected features where possible."""
        if self.num_features == 0:
            print("No modelling features available for KDE plots.")
            return None

        fig, axes = plt.subplots(self.rows, self.cols, figsize=(self.cols * 5, self.fig_height))
        axes = np.atleast_1d(axes).flatten()

        for i, col in enumerate(self.features):
            ax = axes[i]

            positive_only = col in self.log_scale_features
            surv = self._class_series(col, 'Survival', positive_only=positive_only).to_numpy(dtype=float)
            fail = self._class_series(col, 'Failure', positive_only=positive_only).to_numpy(dtype=float)
            imp = self._imputed_series(col, positive_only=positive_only).to_numpy(dtype=float)

            valid_arrays = [a for a in [surv, fail, imp] if len(a) > 0]
            if len(valid_arrays) == 0:
                ax.set_title(f"{self._labelname(col)} (no data)")
                ax.axis("off")
                continue

            all_vals = np.concatenate(valid_arrays)
            if len(all_vals) == 0:
                ax.set_title(f"{self._labelname(col)} (no data)")
                ax.axis("off")
                continue

            if col in self.log_scale_features:
                if np.any(all_vals <= 0):
                    all_vals = all_vals[all_vals > 0]
                if len(all_vals) == 0:
                    ax.set_title(f"{self._labelname(col)} (no positive data)")
                    ax.axis("off")
                    continue

                xmin = np.min(all_vals)
                xmax = np.max(all_vals)
                if xmin == xmax:
                    xmin *= 0.8
                    xmax *= 1.2
                x_grid = np.logspace(np.log10(xmin), np.log10(xmax), 400)
                ax.set_xscale("log")

                def _plot_log_kde(arr, label, color):
                    arr = arr[np.isfinite(arr)]
                    arr = arr[arr > 0]
                    if not self._feature_has_variation(arr):
                        return
                    try:
                        kde = gaussian_kde(np.log10(arr))
                        y = kde(np.log10(x_grid)) / (x_grid * np.log(10))
                        ax.plot(x_grid, y, label=label, linewidth=2, color=color)
                        ax.fill_between(x_grid, y, alpha=0.20, color=color)
                    except Exception:
                        return

            else:
                xmin = np.min(all_vals)
                xmax = np.max(all_vals)
                if xmin == xmax:
                    xmin -= 0.5
                    xmax += 0.5
                x_grid = np.linspace(xmin, xmax, 400)

                def _plot_log_kde(arr, label, color):
                    arr = arr[np.isfinite(arr)]
                    if not self._feature_has_variation(arr):
                        return
                    try:
                        kde = gaussian_kde(arr)
                        y = kde(x_grid)
                        ax.plot(x_grid, y, label=label, linewidth=2, color=color)
                        ax.fill_between(x_grid, y, alpha=0.20, color=color)
                    except Exception:
                        return

            _plot_log_kde(surv, "Survival", "forestgreen")
            _plot_log_kde(fail, "Failure", "firebrick")
            _plot_log_kde(imp, "Imputed", "steelblue")

            handles, labels = ax.get_legend_handles_labels()
            if len(handles) == 0:
                ax.set_title(f"{self._labelname(col)} (insufficient variation)")
                ax.axis("off")
                continue

            ax.set_title(self._labelname(col), fontsize=12)
            ax.set_xlabel(self._labelunits(col), fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

            if self.export_all:
                fig_single, ax_single = plt.subplots(figsize=(6, 4.5))

                if col in self.log_scale_features:
                    ax_single.set_xscale("log")

                    def _plot_single(arr, label, color):
                        arr = arr[np.isfinite(arr)]
                        arr = arr[arr > 0]
                        if not self._feature_has_variation(arr):
                            return
                        try:
                            kde = gaussian_kde(np.log10(arr))
                            y = kde(np.log10(x_grid)) / (x_grid * np.log(10))
                            ax_single.plot(x_grid, y, label=label, linewidth=2, color=color)
                            ax_single.fill_between(x_grid, y, alpha=0.20, color=color)
                        except Exception:
                            return
                else:
                    def _plot_single(arr, label, color):
                        arr = arr[np.isfinite(arr)]
                        if not self._feature_has_variation(arr):
                            return
                        try:
                            kde = gaussian_kde(arr)
                            y = kde(x_grid)
                            ax_single.plot(x_grid, y, label=label, linewidth=2, color=color)
                            ax_single.fill_between(x_grid, y, alpha=0.20, color=color)
                        except Exception:
                            return

                _plot_single(surv, "Survival", "forestgreen")
                _plot_single(fail, "Failure", "firebrick")
                _plot_single(imp, "Imputed", "steelblue")

                h2, l2 = ax_single.get_legend_handles_labels()
                if len(h2) > 0:
                    ax_single.set_title(self._labelname(col))
                    ax_single.set_xlabel(self._labelunits(col))
                    ax_single.set_ylabel("Density")
                    ax_single.grid(True, which='both', linestyle='--', alpha=0.4)
                    ax_single.legend()
                    fig_single.tight_layout()
                    fig_single.savefig(os.path.join(self.kde_dir, f"{self._safe_filename(col)}.png"), dpi=300)

                plt.close(fig_single)

        for j in range(self.num_features, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("KDE plots of modelling features", fontsize=15, y=1.02)
        fig.tight_layout()

        if self.export_all:
            fig.savefig(os.path.join(self.base_dir, "all_kdes.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return None

    ##################################################
    #                 Scatter plots                  #
    ##################################################

    def scatter_plots(self):
        """Generate scatter plots between features and the target."""
        if self.y not in self.df.columns:
            print(f"Target column '{self.y}' not found, skipping scatter plots.")
            return None

        if self.num_features == 0:
            print("No modelling features available for scatter plots.")
            return None

        fig, axes = plt.subplots(self.rows, self.cols, figsize=(self.cols * 5, self.fig_height))
        axes = np.atleast_1d(axes).flatten()

        survived = self.df[self.df['Survival/failure'] == 'Survival'].copy() if 'Survival/failure' in self.df.columns else pd.DataFrame()
        failed = self.df[self.df['Survival/failure'] == 'Failure'].copy() if 'Survival/failure' in self.df.columns else pd.DataFrame()

        for i, feat in enumerate(self.features):
            ax = axes[i]

            sub = self.df[[feat, self.y]].copy()
            sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
            sub[self.y] = pd.to_numeric(sub[self.y], errors="coerce")
            sub = sub.dropna()

            if feat in self.log_scale_features:
                sub = sub[sub[feat] > 0]

            if len(sub) == 0:
                ax.set_title(f"{self._labelname(feat)} (no data)")
                ax.axis("off")
                continue

            surv = survived[[feat, self.y]].copy() if feat in survived.columns and self.y in survived.columns else pd.DataFrame(columns=[feat, self.y])
            fail = failed[[feat, self.y]].copy() if feat in failed.columns and self.y in failed.columns else pd.DataFrame(columns=[feat, self.y])

            surv[feat] = pd.to_numeric(surv[feat], errors="coerce")
            surv[self.y] = pd.to_numeric(surv[self.y], errors="coerce")
            fail[feat] = pd.to_numeric(fail[feat], errors="coerce")
            fail[self.y] = pd.to_numeric(fail[self.y], errors="coerce")

            surv = surv.dropna()
            fail = fail.dropna()

            if feat in self.log_scale_features:
                surv = surv[surv[feat] > 0]
                fail = fail[fail[feat] > 0]
                ax.set_xscale('log')

            if len(surv) > 0:
                ax.scatter(surv[feat], surv[self.y], color='black', marker='.', alpha=0.5, s=60, label='Survival')
            if len(fail) > 0:
                ax.scatter(fail[feat], fail[self.y], color='red', marker='x', alpha=0.9, s=50, label='Failure')

            ax.set_title(f"{self._labelname(feat)} vs {self._labelname(self.y)}", fontsize=11)
            ax.set_xlabel(self._labelunits(feat), fontsize=11)
            ax.set_ylabel(self._labelname(self.y), fontsize=11)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=10)

            if self.export_all:
                fig_single, ax_single = plt.subplots(figsize=(6, 4.5))
                if feat in self.log_scale_features:
                    ax_single.set_xscale('log')

                if len(surv) > 0:
                    ax_single.scatter(surv[feat], surv[self.y], color='black', marker='.', alpha=0.5, s=60, label='Survival')
                if len(fail) > 0:
                    ax_single.scatter(fail[feat], fail[self.y], color='red', marker='x', alpha=0.9, s=50, label='Failure')

                ax_single.set_title(f"{self._labelname(feat)} vs {self._labelname(self.y)}")
                ax_single.set_xlabel(self._labelunits(feat))
                ax_single.set_ylabel(self._labelname(self.y))
                ax_single.grid(True, which='both', linestyle='--', alpha=0.4)
                ax_single.legend()
                fig_single.tight_layout()
                fig_single.savefig(
                    os.path.join(self.scatter_dir, f"{self._safe_filename(feat)}_vs_{self._safe_filename(self.y)}.png"),
                    dpi=300
                )
                plt.close(fig_single)

        for j in range(self.num_features, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Scatter plots of modelling features", fontsize=15, y=1.02)
        fig.tight_layout()

        if self.export_all:
            fig.savefig(os.path.join(self.base_dir, "all_scatterplots.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return None

    ##############################
    #         Pair plots         #
    ##############################

    def pair_plots(self, max_features=8):
        """Generate pair plots for the selected feature subset."""
        if 'Survival/failure' not in self.df.columns:
            print("Column 'Survival/failure' not found, skipping pair plot.")
            return None

        usable_features = []
        for f in self.features:
            if f not in self.df.columns:
                continue

            s = pd.to_numeric(self.df[f], errors="coerce")
            if f in self.log_scale_features:
                s = s[s > 0]

            if s.notna().sum() < 3:
                continue
            if s.dropna().nunique() < 2:
                continue

            usable_features.append(f)

        if len(usable_features) == 0:
            print("No usable features available for pair plot.")
            return None

        usable_features = usable_features[:max_features]

        pairwise_df = self.df[usable_features + ['Survival/failure']].copy()
        for col in usable_features:
            pairwise_df[col] = pd.to_numeric(pairwise_df[col], errors="coerce")
            if col in self.log_scale_features:
                pairwise_df.loc[pairwise_df[col] <= 0, col] = np.nan

        pairwise_df = pairwise_df.dropna()
        if len(pairwise_df) < 3:
            print("Not enough complete rows for pair plot.")
            return None

        palette = {'Survival': 'green', 'Failure': 'red'}
        markers = ['o', 'X']

        g = sns.pairplot(
            pairwise_df,
            vars=usable_features,
            hue='Survival/failure',
            palette=palette,
            markers=markers,
            corner=True,
            diag_kind='hist',
            plot_kws={'alpha': 0.7, 's': 35, 'edgecolor': 'none'},
            diag_kws={'alpha': 0.6, 'bins': 15}
        )

        g.figure.suptitle("Pair plot of modelling features", y=1.02)

        if self.export_all:
            g.savefig(os.path.join(self.pair_dir, "pairwise_feature_scatter_matrix.png"), dpi=300, bbox_inches="tight")
            plt.close(g.figure)
        else:
            plt.show()

        return None

    ##############################
    #    Correlation matrices    #
    ##############################

    def correlation_matrix(self):
        """Compute and visualize the feature correlation matrix."""
        usable_features = []
        for f in self.features:
            if f not in self.df.columns:
                continue
            s = pd.to_numeric(self.df[f], errors="coerce")
            if s.notna().sum() < 2:
                continue
            usable_features.append(f)

        if len(usable_features) < 2:
            print("Not enough usable features for correlation matrix.")
            return None

        corr_df = self.df[usable_features].copy()

        # remove duplicate column names
        corr_df = corr_df.loc[:, ~corr_df.columns.duplicated()].copy()
        usable_features = corr_df.columns.tolist()
        
        # force numeric
        for col in usable_features:
            corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")
        
        corr = corr_df.corr(method="pearson")
        df_pvalues = pd.DataFrame(np.nan, index=usable_features, columns=usable_features)
        
        for col1 in usable_features:
            for col2 in usable_features:
                s1 = corr_df[col1]
                s2 = corr_df[col2]
        
                # if duplicate names somehow still return DataFrames, reduce to first column
                if isinstance(s1, pd.DataFrame):
                    s1 = s1.iloc[:, 0]
                if isinstance(s2, pd.DataFrame):
                    s2 = s2.iloc[:, 0]
        
                pair = pd.concat([s1, s2], axis=1).dropna()
        
                # not enough data
                if len(pair) < 3:
                    continue
        
                # skip constant columns
                if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                    continue
        
                try:
                    _, p_value = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                    df_pvalues.loc[col1, col2] = p_value
                except Exception:
                    df_pvalues.loc[col1, col2] = np.nan

        label_map = {col: self._labelname(col) for col in usable_features}
        corr_named = corr.rename(index=label_map, columns=label_map)
        p_named = df_pvalues.rename(index=label_map, columns=label_map)

        # 1) standard heatmap
        plt.figure(figsize=(12, 10))
        ax0 = sns.heatmap(
            corr_named,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            linecolor="grey"
        )
        plt.title("Correlation matrix of modelling features", fontsize=15, pad=20)
        plt.tight_layout()

        if self.export_all:
            plt.savefig(os.path.join(self.corr_dir, "correlation_matrix_heatmap.png"), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # 2) significance-highlighted heatmap
        bounds = [-1, -0.4, 0.4, 0.99, 1]
        colors = ["firebrick", "whitesmoke", "green", "whitesmoke"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(13, 11))
        ax2 = sns.heatmap(
            corr_named,
            cmap=cmap,
            norm=norm,
            square=True,
            linecolor='grey',
            linewidths=0.5,
            cbar=False,
            annot=False,
            alpha=0.8
        )

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                value = corr.iloc[i, j]
                pval = df_pvalues.iloc[i, j]

                if i == j or (pd.notna(pval) and pval > 0.05):
                    ax2.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="whitesmoke", linewidth=0))

                text = "nan" if pd.isna(value) else f"{value:.2f}"
                ax2.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="black", fontsize=11)

        for i in range(corr.shape[0] + 1):
            ax2.axhline(i, color='grey', linewidth=0.5)
        for j in range(corr.shape[1] + 1):
            ax2.axvline(j, color='grey', linewidth=0.5)

        plt.title("Correlation matrix with non-significant cells greyed out", fontsize=15, pad=20)
        plt.tight_layout()

        if self.export_all:
            plt.savefig(os.path.join(self.corr_dir, "correlation_matrix_significant.png"), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # 3) export raw values
        if self.export_all:
            corr.to_csv(os.path.join(self.corr_dir, "correlation_matrix_values.csv"))
            df_pvalues.to_csv(os.path.join(self.corr_dir, "correlation_matrix_pvalues.csv"))

        return None



class ExplainingPower:
    
    """Small helper for collecting and returning explaining-power results."""
    def __init__(self, database: Database, alphas=None):
        """Initialize the object and prepare the main internal state."""
        self._database = database
        self.df = self._database.get_dataframe()
        self.features = database._get_modelling_features()
        self.num_features = len(database._get_modelling_features())
        self.log_scale_features = {'d10', 'd50', 'd60', 'd70', 'Hydraulic_conductivity', 'Hydraulic_conductivity_KC'}
        
        self.selector = 'test'
        self.base_dir = f"results/statistics/database-statistics_{self.selector}/"
        self.export_all = True
        
        self.imputed_only = pd.read_pickle(f"src/bayesian_network_results/Imputed_only_{self.selector}_with_{len(self.df)}_rows.pkl")
        
        for i in self.features:
            if i not in self.imputed_only.columns:
                self.imputed_only[i] = np.nan
            
        os.makedirs(f"results/statistics/database-statistics_{self.selector}/", exist_ok=True)
        
        self.cols = 3
        self.rows = math.ceil(self.num_features / self.cols)
        self.fig_height = (self.rows + 1) * 4
        
        # ------------------------------------------------------------------
        # CENTRAL PLOT STYLE SETTINGS
        # ------------------------------------------------------------------
        self.bar_color      = "tab:blue"
        self.bar_edgecolor  = "black"
        self.bar_height     = 0.6
        self.figsize        = (14, 9)

        self.fs_title       = 30   # figure titles
        self.fs_label       = 25   # axis labels (x/y)
        self.fs_ticks       = 23   # tick labels

        self.title_pad      = 20   # extra space above titles
        self.xlabel_pad     = 12   # extra space between x-axis and label

        self.left_margin    = 0.30 # room for long y labels in barh plots
        # ------------------------------------------------------------------

        # Setup / columns
        self.target_col = 'Survival/failure'
        self.exclude = ['Bedding_angle', 'Friction_angle']
        self.predictor_cols = [f for f in self.features if f not in self.exclude]
        self.predictor_cols.append('Tan_Bedding_angle')
        self.predictor_cols.append('Tan_Friction_angle')
        
        self.to_names = {
            "Water_level_diff": "Water level difference",
            "Aquifer_thickness": "Aquifer thickness",
            "Blanket_thickness": "Blanket thickness",
            "Seepage_length": "Seepage length",
            "Hydraulic_conductivity_KC": "Hydraulic conductivity",
            "Uniformity_coefficient": "Uniformity coefficient",
            "Porosity": "Porosity",
            "Tan_Friction_angle": "Tan(Friction angle)",
            "Tan_Bedding_angle": "Tan(Bedding angle)",
            "Bedding_angle": "Tan(Bedding angle)",
        } 
        
        # Map target
        label_map = {'Failure': 1, 'Survival': 0}
        if not set(self.df[self.target_col].dropna().unique()).issubset(set(label_map.keys())):
            raise ValueError("Target column must contain only 'Survival' or 'Failure' strings.")
        self.df['_ybin'] = self.df[self.target_col].map(label_map)

        # Results containers
        self.results = {
            'lasso': [],
            'mutual_information': [],
            'random_forest_importance': [],
            'permutation_importance_cv': [],
            'elastic_net_coef': [],
            'stability_selection_freq': [],
            'boruta': [],
            'shap_importance': []
        }
        
        # Prepare multivariate matrices
        self.multiv_cols = self.predictor_cols + ['_ybin']
        self.mv_df = self.df[self.multiv_cols].dropna()
            
        X_mv = self.mv_df[self.predictor_cols].values
        y_mv = self.mv_df['_ybin'].values

        # Logistic CV hyperparams
        if alphas is not None:
            Cs = [1.0 / a for a in alphas if a and a > 0] or [0.1, 0.5, 1, 2, 5, 10]
        else:
            Cs = [0.1, 0.5, 1, 2, 5, 10]
        
        #Output order (plots):
        #  1) L1-Logistic (univariate)
        #  2) Mutual Information (univariate)
        #  3) Random Forest importance (multivariate)
        #  4) Permutation Importance with CV (AUC-based)
        #  5) Elastic Net Logistic (multivariate coefficients)
        #  6) Stability Selection (L1-logistic selection frequency)
        
        
        def L1_univariate(self):
            # 1) L1-Logistic (univariate)
            for predictor in self.predictor_cols:
                sub_df = self.df[[predictor, '_ybin']].dropna()
                if sub_df.shape[0] < 10 or sub_df['_ybin'].nunique() < 2:
                    continue
                X = StandardScaler().fit_transform(sub_df[[predictor]].values)
                y = sub_df['_ybin'].values
                try:
                    log_cv = LogisticRegressionCV(
                        Cs=Cs, cv=5, penalty='l1', solver='saga',
                        scoring='roc_auc', max_iter=5000, n_jobs=1,
                        refit=True, random_state=42, class_weight='balanced'
                    ).fit(X, y)
                    self.results['lasso'].append((predictor, float(log_cv.coef_.ravel()[0])))
                except Exception:
                    continue

            if len(self.results['lasso']) > 0:
                fig_lasso, ax = plt.subplots(1, 1, figsize=self.figsize)
                dfp = pd.DataFrame(self.results['lasso'], columns=["Predictor", "L1-Logistic Coefficient"]) \
                      .sort_values("L1-Logistic Coefficient", ascending=False)

                dfp["Label"] = dfp["Predictor"].apply(
                    lambda x: self.to_names.get(x, x)
                )
                ax.barh(
                    dfp["Label"], dfp["L1-Logistic Coefficient"],
                    color=self.bar_color, edgecolor=self.bar_edgecolor,
                    linewidth=1.0, height=self.bar_height
                )
                ax.set_title("Univariate LASSO", pad=self.title_pad)
                ax.set_xlabel("Coefficient in LASSO model", labelpad=self.xlabel_pad)
                ax.grid(axis="x", linestyle="--", alpha=0.7)

                # Fonts
                ax.title.set_fontsize(self.fs_title)
                ax.xaxis.label.set_fontsize(self.fs_label)
                ax.yaxis.label.set_fontsize(self.fs_label)
                ax.tick_params(axis='both', labelsize=self.fs_ticks)

                plt.tight_layout()
                plt.subplots_adjust(left=self.left_margin)
                out_dir = f"results/statistics/database-statistics_{self.selector}/"
                os.makedirs(out_dir, exist_ok=True)
                try: fig_lasso.savefig(f"{out_dir}logistic_l1_univariate.png", dpi=300)
                except: pass
                
                
        
        def Mutual_information(self):
            # 2) Mutual Information (univariate)
            for predictor in self.predictor_cols:
                sub_df = self.df[[predictor, '_ybin']].dropna()
                if sub_df.shape[0] < 10 or sub_df['_ybin'].nunique() < 2:
                    continue
                X = StandardScaler().fit_transform(sub_df[[predictor]].values)
                y = sub_df['_ybin'].values
                try:
                    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
                    self.results['mutual_information'].append((predictor, float(mi[0])))
                except Exception:
                    pass

            if len(self.results['mutual_information']) > 0:
                fig_mi, ax = plt.subplots(1, 1, figsize=self.figsize)
                midf = pd.DataFrame(self.results['mutual_information'], columns=["Predictor", "Mutual Info"]) \
                       .sort_values("Mutual Info", ascending=True)
                ax.barh(
                    midf["Predictor"], midf["Mutual Info"],
                    color=self.bar_color, edgecolor=self.bar_edgecolor,
                    linewidth=1.0, height=self.bar_height
                )
                ax.set_title("Mutual Information with Categorical Target (Univariate)", pad=self.title_pad)
                ax.set_xlabel("Mutual Information", labelpad=self.xlabel_pad)
                ax.grid(axis="x", linestyle="--", alpha=0.7)

                ax.title.set_fontsize(self.fs_title)
                ax.xaxis.label.set_fontsize(self.fs_label)
                ax.yaxis.label.set_fontsize(self.fs_label)
                ax.tick_params(axis='both', labelsize=self.fs_ticks)

                plt.tight_layout()
                plt.subplots_adjust(left=self.left_margin)

            # Prepare multivariate matrices
            multiv_cols = self.predictor_cols + ['_ybin']
            mv_df = self.df[multiv_cols].dropna()
            if mv_df['_ybin'].nunique() < 2 or mv_df.shape[0] < 20:
                if self.export_all:
                    out_dir = f"results/statistics/database-statistics_{self.selector}/"
                    os.makedirs(out_dir, exist_ok=True)
                    try: fig_mi.savefig(f"{out_dir}mi_univariate_cls.png", dpi=300)
                    except: pass
                    plt.close()
                else:
                    plt.close()
                return self.results



            
        
        def random_forest(self):
            # 3) Random Forest (multivariate)

            
            classes = np.array([0, 1])
            cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_mv)
            class_weight_dict = {cls: w for cls, w in zip(classes, cw)}

            rf = RandomForestClassifier(
                n_estimators=1000, random_state=42, class_weight=class_weight_dict,
                max_features='sqrt', n_jobs=-1, oob_score=False
            )
            rf.fit(X_mv, y_mv)
            rf_imp = rf.feature_importances_
            rf_df = pd.DataFrame({"Predictor": self.predictor_cols, "RF Importance": rf_imp}) \
                    .sort_values("RF Importance", ascending=True)

            rf_df["Label"] = rf_df["Predictor"].apply(lambda x: self.to_names.get(x, x))
            
            self.results['random_forest_importance'] = list(zip(rf_df["Predictor"], rf_df["RF Importance"]))
            
            fig_rf, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.barh(
                rf_df["Label"], rf_df["RF Importance"],
                color=self.bar_color, edgecolor=self.bar_edgecolor,
                linewidth=1.0, height=self.bar_height
            )
            ax.set_title("Random Forest using the Gini criterion", pad=self.title_pad)
            ax.set_xlabel("Importance score", labelpad=self.xlabel_pad)
            ax.grid(axis="x", linestyle="--", alpha=0.7)

            ax.title.set_fontsize(self.fs_title)
            ax.xaxis.label.set_fontsize(self.fs_label)
            ax.yaxis.label.set_fontsize(self.fs_label)
            ax.tick_params(axis='both', labelsize=self.fs_ticks)

            plt.tight_layout()
            plt.subplots_adjust(left=self.left_margin)
            
            os.makedirs(self.out_dir, exist_ok=True)

            try: fig_rf.savefig(f"{self.out_dir}rf_importance.png", dpi=300)
            except: pass

        
        def permutation_importance(self):
            # 4) Permutation Importance with CV (AUC)
            classes = np.array([0, 1])
            cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_mv)
            class_weight_dict = {cls: w for cls, w in zip(classes, cw)}
            
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            perm_means = []
            for train_idx, val_idx in kf.split(X_mv, y_mv):
                X_tr, X_val = X_mv[train_idx], X_mv[val_idx]
                y_tr, y_val = y_mv[train_idx], y_mv[val_idx]

                rf_cv = RandomForestClassifier(
                    n_estimators=800, random_state=42, class_weight=class_weight_dict,
                    max_features='sqrt', n_jobs=-1
                ).fit(X_tr, y_tr)

                perm = permutation_importance(
                    rf_cv, X_val, y_val, scoring='roc_auc',
                    n_repeats=20, random_state=42, n_jobs=-1
                )
                perm_means.append(perm.importances_mean)

            perm_means = np.stack(perm_means, axis=0)
            perm_mean = perm_means.mean(axis=0)
            perm_std = perm_means.std(axis=0, ddof=1)
            perm_df = pd.DataFrame({
                "Predictor": self.predictor_cols,
                "Mean AUC Drop": perm_mean,
                "Std AUC Drop": perm_std
            }).sort_values("Mean AUC Drop", ascending=True)
            perm_df["Label"] = perm_df["Predictor"].apply(lambda x: self.to_names.get(x, x))
            
            self.results['permutation_importance_cv'] = list(zip(
                perm_df["Predictor"], perm_df["Mean AUC Drop"], perm_df["Std AUC Drop"]
            ))
            
            fig_perm, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.barh(
                perm_df["Label"], perm_df["Mean AUC Drop"],
                color=self.bar_color, edgecolor=self.bar_edgecolor,
                linewidth=1.0, height=self.bar_height
            )
            ax.set_title("Permutation Importance based on random forest", pad=self.title_pad)
            ax.set_xlabel("Permutation importance [%]", labelpad=self.xlabel_pad)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.1f}%"))
            ax.grid(axis="x", linestyle="--", alpha=0.7)

            ax.title.set_fontsize(self.fs_title)
            ax.xaxis.label.set_fontsize(self.fs_label)
            ax.yaxis.label.set_fontsize(self.fs_label)
            ax.tick_params(axis='both', labelsize=self.fs_ticks)

            plt.tight_layout()
            plt.subplots_adjust(left=self.left_margin)
        
            os.makedirs(self.out_dir, exist_ok=True)
            try: fig_perm.savefig(f"{self.out_dir}permutation_importance_cv.png", dpi=300)
            except: pass
        
        
        def elastic_net(self):
            # 5) Elastic Net Logistic (multivariate)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_mv)

            enet_cv = LogisticRegressionCV(
                Cs=Cs, cv=5, penalty='elasticnet', solver='saga',
                l1_ratios=[0.2, 0.5, 0.8], scoring='roc_auc',
                max_iter=8000, n_jobs=-1, refit=True, random_state=42,
                class_weight='balanced'
            ).fit(X_scaled, y_mv)

            enet_df = pd.DataFrame({"Predictor": self.predictor_cols,
                                    "ElasticNet Coef": enet_cv.coef_.ravel()}) \
                      .sort_values("ElasticNet Coef", ascending=False)
            enet_df["Label"] = enet_df["Predictor"].apply(lambda x: self.to_names.get(x, x))
            
            self.results['elastic_net_coef'] = list(zip(enet_df["Predictor"], enet_df["ElasticNet Coef"]))

            fig_enet, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.barh(
                enet_df["Label"], enet_df["ElasticNet Coef"],
                color=self.bar_color, edgecolor=self.bar_edgecolor,
                linewidth=1.0, height=self.bar_height
            )
            ax.set_title("Multivariate LASSO model (elastic net)", pad=self.title_pad)
            ax.set_xlabel("Coefficient", labelpad=self.xlabel_pad)
            ax.grid(axis="x", linestyle="--", alpha=0.7)

            ax.title.set_fontsize(self.fs_title)
            ax.xaxis.label.set_fontsize(self.fs_label)
            ax.yaxis.label.set_fontsize(self.fs_label)
            ax.tick_params(axis='both', labelsize=self.fs_ticks)

            plt.tight_layout()
            plt.subplots_adjust(left=self.left_margin)
            
            os.makedirs(self.out_dir, exist_ok=True)

            try: fig_enet.savefig(f"{self.out_dir}elasticnet_multivariate.png", dpi=300)
            except: pass
        
        
        def stability_selection(self):
            # 6) Stability Selection (bootstrap L1-logistic)
            n_boot = 100
            sel_counts = np.zeros(len(self.predictor_cols), dtype=int)
            for b in range(n_boot):
                X_bs, y_bs = resample(self.X_scaled, y_mv, replace=True, random_state=42 + b, stratify=y_mv)
                try:
                    lr = LogisticRegression(
                        penalty='l1', solver='saga', C=1.0, max_iter=5000,
                        class_weight='balanced', random_state=42 + b
                    ).fit(X_bs, y_bs)
                    sel_counts += (np.abs(lr.coef_.ravel()) > 1e-8).astype(int)
                except Exception:
                    continue

            stab_df = pd.DataFrame({"Predictor": self.predictor_cols,
                                    "Selection Frequency": sel_counts / max(1, n_boot)}) \
                      .sort_values("Selection Frequency", ascending=True)
            stab_df["Label"] = stab_df["Predictor"].apply(lambda x: self.to_names.get(x, x))
            
            self.results['stability_selection_freq'] = list(zip(stab_df["Predictor"], stab_df["Selection Frequency"]))

            fig_stab, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.barh(
                stab_df["Label"], stab_df["Selection Frequency"],
                color=self.bar_color, edgecolor=self.bar_edgecolor,
                linewidth=1.0, height=self.bar_height
            )
            ax.set_title("Stability selection of univariate LASSO model", pad=self.title_pad)
            ax.set_xlabel("Selection Frequency [%]", labelpad=self.xlabel_pad)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.1f}%"))
            ax.grid(axis="x", linestyle="--", alpha=0.7)

            ax.title.set_fontsize(self.fs_title)
            ax.xaxis.label.set_fontsize(self.fs_label)
            ax.yaxis.label.set_fontsize(self.fs_label)
            ax.tick_params(axis='both', labelsize=self.fs_ticks)

            plt.tight_layout()
            plt.subplots_adjust(left=self.left_margin)
            
            os.makedirs(self.out_dir, exist_ok=True)

            try: fig_stab.savefig(f"{self.out_dir}stability_selection.png", dpi=300)
            except: pass
        
        def stability_selection2(self):
            # 7) Stability Selection (bootstrap Elastic Net logistic)
            n_boot = 100
            sel_counts_enet = np.zeros(len(self.predictor_cols), dtype=int)
            
            for b in range(n_boot):
                X_bs, y_bs = resample(
                    self.X_scaled, y_mv,
                    replace=True,
                    random_state=42 + b,
                    stratify=y_mv
                )
                try:
                    lr_enet = LogisticRegression(
                        penalty='elasticnet',
                        solver='saga',
                        C=1.0,              # or best C from your CV
                        l1_ratio=0.5,       # or best l1_ratio from your CV
                        max_iter=5000,
                        class_weight='balanced',
                        random_state=42 + b
                    ).fit(X_bs, y_bs)

                    # mark features with non-zero (or sufficiently large) coefficients
                    sel_counts_enet += (np.abs(lr_enet.coef_.ravel()) > 1e-8).astype(int)
                except Exception:
                    continue

            stab_enet_df = pd.DataFrame({
                "Predictor": self.redictor_cols,
                "Selection Frequency": sel_counts_enet / max(1, n_boot)
            }).sort_values("Selection Frequency", ascending=True)
            stab_enet_df["Label"] = stab_enet_df["Predictor"].apply(lambda x: self.to_names.get(x, x))
            
            self.results['stability_selection_enet'] = list(
                zip(stab_enet_df["Predictor"], stab_enet_df["Selection Frequency"])
            )

            fig_stab_enet, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.barh(
                stab_enet_df["Label"], stab_enet_df["Selection Frequency"],
                color=self.bar_color, edgecolor=self.bar_edgecolor,
                linewidth=1.0, height=self.bar_height
            )
            ax.set_title("Stability selection of multivariate LASSO model", pad=self.title_pad)
            ax.set_xlabel("Selection Frequency [%]", labelpad=self.xlabel_pad)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.1f}%"))
            ax.grid(axis="x", linestyle="--", alpha=0.7)

            ax.title.set_fontsize(self.fs_title)
            ax.xaxis.label.set_fontsize(self.fs_label)
            ax.yaxis.label.set_fontsize(self.fs_label)
            ax.tick_params(axis='both', labelsize=self.fs_ticks)

            plt.tight_layout()
            plt.subplots_adjust(left=self.left_margin)

            # Export
            if self.export_all:
                os.makedirs(self.out_dir, exist_ok=True)

                try: fig_stab_enet.savefig(f"{self.out_dir}stability_selection_enet.png", dpi=300)
                except: pass
                plt.show()
            else:
                plt.show()
    
    def fetch_results(self):
        """Collect and return explaining-power results."""
        return self.results
    
    
    
    
    
    
class Model_comparison:
    """Compare current or legacy model outputs against observed values."""
    def __init__(self, database: Database, selector='test', export_all=True, auto_fill_predictions=True):
        """Initialize the object and prepare the main internal state."""
        self._database = database

        # robust dataframe getter
        if callable(getattr(self._database, "get_dataframe", None)):
            self.df = self._database.get_dataframe().copy()
        else:
            self.df = self._database.get_dataframe.copy()

        self.selector = selector
        self.export_all = export_all
        self.y_dH = "Water_level_diff"
        self.y_ic = "Global_gradient"

        self.base_dir = f"results/comparison/database-comparison_{self.selector}"
        self.metrics_dir = os.path.join(self.base_dir, "metrics")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.tables_dir = os.path.join(self.base_dir, "tables")

        for d in [self.base_dir, self.metrics_dir, self.plots_dir, self.tables_dir]:
            os.makedirs(d, exist_ok=True)

        self.model_registry = self._database.get_existing_model_registry()

        # optionally fill predictions first if database does not already contain them
        if auto_fill_predictions:
            try:
                self._database.fill_existing_model_predictions(overwrite=False, verbose=False)
                if callable(getattr(self._database, "get_dataframe", None)):
                    self.df = self._database.get_dataframe().copy()
                else:
                    self.df = self._database.get_dataframe.copy()
            except Exception as e:
                print(f"Warning: could not auto-fill existing model predictions: {e}")

    @staticmethod
    def _safe_filename(name):
        """Convert a label into a filesystem-safe filename."""
        return (
            str(name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

    @staticmethod
    def _labelname(name):
        """Return a cleaner display label for a feature or metric name."""
        to_real_name = {
            "Water_level_diff": "Water level difference",
            "Global_gradient": "Global gradient",
            "Aquifer_thickness": "Aquifer thickness",
            "Blanket_thickness": "Blanket thickness",
            "Seepage_length": "Seepage length",
            "Hydraulic_conductivity_KC": "Hydraulic conductivity",
            "Hydraulic_conductivity": "Hydraulic conductivity",
            "Uniformity_coefficient": "Uniformity coefficient",
            "Porosity": "Porosity",
            "Friction_angle": "Friction angle",
            "Bedding_angle": "Bedding angle",
            "d10": "d10",
            "d50": "d50",
            "d60": "d60",
            "d70": "d70",
        }
        return to_real_name.get(name, name)

    @staticmethod
    def _json_safe(x):
        """Convert objects into a JSON-serializable representation."""
        if isinstance(x, dict):
            return {k: Model_comparison._json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [Model_comparison._json_safe(v) for v in list(x)]
        if isinstance(x, (float, np.floating)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x

    def _to_failure_binary(self, series):
        """Convert failure indicators into a consistent binary target."""
        if pd.api.types.is_numeric_dtype(series):
            vals = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
            return vals.to_numpy()
        mapping = {"Survival": 0, "Failure": 1}
        return series.map(mapping).fillna(0).astype(int).to_numpy()

    def _regression_metrics(self, y_true, y_pred, k):
        """Compute regression metrics for one prediction series."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        n = len(y_true)
        if n == 0:
            return {
                "N": 0,
                "RMSE": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "R2": np.nan,
                "R2_Adjusted": np.nan,
            }

        resid = y_true - y_pred
        sse = float(np.sum(resid ** 2))
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        mse = max(sse / max(n, 1), 1e-300)
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(max(n, 1))

        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= 0:
            r2 = np.nan
            adj_r2 = np.nan
        else:
            r2 = 1.0 - (sse / ss_tot)
            adj_r2 = 1.0 - ((1.0 - r2) * (n - 1)) / max(n - k - 1, 1)

        return {
            "N": int(n),
            "RMSE": rmse,
            "AIC": float(aic),
            "BIC": float(bic),
            "R2": float(r2) if pd.notna(r2) else np.nan,
            "R2_Adjusted": float(adj_r2) if pd.notna(adj_r2) else np.nan,
        }

    def _classification_metrics_from_critical_head(self, y_true_cls, dH_obs, dH_crit):
        """Compute classification metrics based on a critical-head threshold."""
        y_true_cls = np.asarray(y_true_cls, dtype=int).ravel()
        dH_obs = np.asarray(dH_obs, dtype=float).ravel()
        dH_crit = np.asarray(dH_crit, dtype=float).ravel()

        mask = np.isfinite(dH_obs) & np.isfinite(dH_crit)
        y_true_cls = y_true_cls[mask]
        dH_obs = dH_obs[mask]
        dH_crit = dH_crit[mask]

        if len(dH_obs) == 0:
            return {
                "N": 0,
                "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "MCC": np.nan,
            }

        # predicted failure if observed dH exceeds critical dH
        y_pred_cls = (dH_obs > dH_crit).astype(int)

        tp = int(np.sum((y_true_cls == 1) & (y_pred_cls == 1)))
        tn = int(np.sum((y_true_cls == 0) & (y_pred_cls == 0)))
        fp = int(np.sum((y_true_cls == 0) & (y_pred_cls == 1)))
        fn = int(np.sum((y_true_cls == 1) & (y_pred_cls == 0)))

        acc = accuracy_score(y_true_cls, y_pred_cls)
        prec = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        rec = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)

        try:
            mcc = matthews_corrcoef(y_true_cls, y_pred_cls)
        except Exception:
            mcc = np.nan

        return {
            "N": int(len(y_true_cls)),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "MCC": float(mcc) if pd.notna(mcc) else np.nan,
        }

    def _get_model_prediction_columns(self, model_name, spec):
        """Return the dataframe columns that correspond to model predictions."""
        safe_name = spec.get("safe_name", model_name)
        pred_col_dH = spec.get("prediction_column_dH", f"H_c_{safe_name}")
        pred_col_ic = spec.get("prediction_column_ic", f"i_c_{safe_name}")
        return safe_name, pred_col_dH, pred_col_ic

    def _build_model_summary_table(self):
        """Assemble the summary table for all detected model outputs."""
        rows = []

        for model_name, spec in self.model_registry.items():
            safe_name, pred_col_dH, pred_col_ic = self._get_model_prediction_columns(model_name, spec)

            row = {
                "model_name": model_name,
                "safe_name": safe_name,
                "prediction_column_dH": pred_col_dH,
                "prediction_column_ic": pred_col_ic,
                "param_count": spec.get("param_count", np.nan),
                "available_dH": pred_col_dH in self.df.columns,
                "available_ic": pred_col_ic in self.df.columns,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def compute_existing_model_metrics(self, export=True):
        """
        Compute regression metrics and classification metrics for all existing models
        using the prediction columns already stored in the database.
        """

        required_cols = ["Survival/failure", self.y_dH, self.y_ic]
        missing_required = [c for c in required_cols if c not in self.df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns in dataframe: {missing_required}")

        y_true_cls = self._to_failure_binary(self.df["Survival/failure"])
        dH_obs = pd.to_numeric(self.df[self.y_dH], errors="coerce").to_numpy(dtype=float)
        ic_obs = pd.to_numeric(self.df[self.y_ic], errors="coerce").to_numpy(dtype=float)
        fail_mask = (y_true_cls == 1)

        metrics_rows = []
        classification_rows = []

        for model_name, spec in self.model_registry.items():
            safe_name, pred_col_dH, pred_col_ic = self._get_model_prediction_columns(model_name, spec)
            k = spec.get("param_count", 1)

            if pred_col_dH not in self.df.columns and pred_col_ic not in self.df.columns:
                print(f"Skipping '{model_name}' because no prediction columns are present in the dataframe.")
                continue

            dH_pred = pd.to_numeric(self.df[pred_col_dH], errors="coerce").to_numpy(dtype=float) if pred_col_dH in self.df.columns else np.full(len(self.df), np.nan)
            ic_pred = pd.to_numeric(self.df[pred_col_ic], errors="coerce").to_numpy(dtype=float) if pred_col_ic in self.df.columns else np.full(len(self.df), np.nan)

            reg_dH_all = self._regression_metrics(dH_obs, dH_pred, k)
            reg_dH_fail = self._regression_metrics(dH_obs[fail_mask], dH_pred[fail_mask], k)
            reg_ic_all = self._regression_metrics(ic_obs, ic_pred, k)
            reg_ic_fail = self._regression_metrics(ic_obs[fail_mask], ic_pred[fail_mask], k)
            cls = self._classification_metrics_from_critical_head(y_true_cls, dH_obs, dH_pred)

            metrics_rows.append({
                "model": model_name,
                "safe_name": safe_name,

                "N_dH_all": reg_dH_all["N"],
                "RMSE_dH_all": reg_dH_all["RMSE"],
                "AIC_dH_all": reg_dH_all["AIC"],
                "BIC_dH_all": reg_dH_all["BIC"],
                "R2_dH_all": reg_dH_all["R2"],
                "R2adj_dH_all": reg_dH_all["R2_Adjusted"],

                "N_dH_fail": reg_dH_fail["N"],
                "RMSE_dH_fail": reg_dH_fail["RMSE"],
                "AIC_dH_fail": reg_dH_fail["AIC"],
                "BIC_dH_fail": reg_dH_fail["BIC"],
                "R2_dH_fail": reg_dH_fail["R2"],
                "R2adj_dH_fail": reg_dH_fail["R2_Adjusted"],

                "N_ic_all": reg_ic_all["N"],
                "RMSE_ic_all": reg_ic_all["RMSE"],
                "AIC_ic_all": reg_ic_all["AIC"],
                "BIC_ic_all": reg_ic_all["BIC"],
                "R2_ic_all": reg_ic_all["R2"],
                "R2adj_ic_all": reg_ic_all["R2_Adjusted"],

                "N_ic_fail": reg_ic_fail["N"],
                "RMSE_ic_fail": reg_ic_fail["RMSE"],
                "AIC_ic_fail": reg_ic_fail["AIC"],
                "BIC_ic_fail": reg_ic_fail["BIC"],
                "R2_ic_fail": reg_ic_fail["R2"],
                "R2adj_ic_fail": reg_ic_fail["R2_Adjusted"],
            })

            classification_rows.append({
                "model": model_name,
                "safe_name": safe_name,
                "N": cls["N"],
                "TP": cls["TP"],
                "TN": cls["TN"],
                "FP": cls["FP"],
                "FN": cls["FN"],
                "Accuracy": cls["Accuracy"],
                "Precision": cls["Precision"],
                "Recall": cls["Recall"],
                "F1": cls["F1"],
                "MCC": cls["MCC"],
            })

        metrics_df = pd.DataFrame(metrics_rows)
        classification_df = pd.DataFrame(classification_rows)

        if export:
            metrics_df.to_csv(os.path.join(self.tables_dir, "existing_model_regression_metrics.csv"), index=False)
            classification_df.to_csv(os.path.join(self.tables_dir, "existing_model_classification_metrics.csv"), index=False)
            self._build_model_summary_table().to_csv(os.path.join(self.tables_dir, "existing_model_summary.csv"), index=False)

            with open(os.path.join(self.metrics_dir, "existing_model_regression_metrics.json"), "w") as f:
                json.dump(self._json_safe(metrics_df.to_dict(orient="records")), f, indent=2, allow_nan=False)

            with open(os.path.join(self.metrics_dir, "existing_model_classification_metrics.json"), "w") as f:
                json.dump(self._json_safe(classification_df.to_dict(orient="records")), f, indent=2, allow_nan=False)

        return {
            "regression": metrics_df,
            "classification": classification_df,
        }

    def plot_existing_model_predictions_vs_real(self, target="Water_level_diff", lims=None, export=True):
        """
        For each model, plot observed vs predicted values.
        target: 'Water_level_diff' or 'Global_gradient'
        """

        if target not in ["Water_level_diff", "Global_gradient"]:
            raise ValueError("target must be 'Water_level_diff' or 'Global_gradient'.")

        if target not in self.df.columns:
            raise ValueError(f"Column '{target}' not found in dataframe.")

        if target == "Water_level_diff":
            ylabel = "Observed water level difference [m]"
        else:
            ylabel = "Observed global gradient [-]"

        true_vals = pd.to_numeric(self.df[target], errors="coerce")

        for model_name, spec in self.model_registry.items():
            safe_name, pred_col_dH, pred_col_ic = self._get_model_prediction_columns(model_name, spec)
            pred_col = pred_col_dH if target == "Water_level_diff" else pred_col_ic

            if pred_col not in self.df.columns:
                continue

            pred_vals = pd.to_numeric(self.df[pred_col], errors="coerce")
            plot_df = pd.DataFrame({
                "observed": true_vals,
                "predicted": pred_vals,
                "class": self.df["Survival/failure"],
            }).dropna()

            if len(plot_df) == 0:
                continue

            x = plot_df["predicted"].to_numpy(dtype=float)
            y = plot_df["observed"].to_numpy(dtype=float)

            if lims is None:
                max_val = np.nanmax(np.concatenate([x, y]))
                this_lim = max(1e-6, 1.05 * max_val)
            else:
                this_lim = lims

            failures = plot_df["class"] == "Failure"
            survivals = plot_df["class"] == "Survival"

            fig, ax = plt.subplots(figsize=(6.5, 5.5))

            x_line = np.linspace(0, this_lim, 300)
            ax.plot(x_line, x_line, "k--", linewidth=1.0, alpha=0.9)
            ax.plot(x_line, 2 * x_line, "k:", linewidth=0.9, alpha=0.8)
            ax.plot(x_line, 0.5 * x_line, "k-.", linewidth=0.9, alpha=0.8)

            ax.scatter(
                plot_df.loc[survivals, "predicted"],
                plot_df.loc[survivals, "observed"],
                color="black", marker=".", s=35, alpha=0.5, label="Survival"
            )
            ax.scatter(
                plot_df.loc[failures, "predicted"],
                plot_df.loc[failures, "observed"],
                color="red", marker="x", s=45, alpha=0.9, label="Failure"
            )

            ax.set_xlim(0, this_lim)
            ax.set_ylim(0, this_lim)
            ax.set_xlabel(f"Predicted {self._labelname(target)}")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model_name}: observed vs predicted")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()

            if export:
                fig.savefig(
                    os.path.join(self.plots_dir, f"{self._safe_filename(safe_name)}_{self._safe_filename(target)}_observed_vs_predicted.png"),
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close(fig)
            else:
                plt.show()

        return None

    def plot_ic_pairs_existing_models(self, lims=0.25, export=True):
        """
        Plot observed failure global gradients against predicted critical global gradients
        for all models in one figure.
        """

        if self.y_ic not in self.df.columns or "Survival/failure" not in self.df.columns:
            raise ValueError("Required columns 'Global_gradient' and/or 'Survival/failure' not found.")

        base = self.df[self.df["Survival/failure"] == "Failure"].copy()
        base[self.y_ic] = pd.to_numeric(base[self.y_ic], errors="coerce")
        base = base.dropna(subset=[self.y_ic])

        if len(base) == 0:
            print("No failure rows with valid Global_gradient found.")
            return None

        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        x_line = np.linspace(0, lims, 300)

        ax.plot(x_line, x_line, "k--", linewidth=1.0, alpha=0.9)
        ax.plot(x_line, 2 * x_line, "k:", linewidth=0.9, alpha=0.8)
        ax.plot(x_line, 0.5 * x_line, "k-.", linewidth=0.9, alpha=0.8)

        marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "<", ">"]

        for i, (model_name, spec) in enumerate(self.model_registry.items()):
            safe_name, _, pred_col_ic = self._get_model_prediction_columns(model_name, spec)

            if pred_col_ic not in base.columns:
                continue

            x_pred = pd.to_numeric(base[pred_col_ic], errors="coerce")
            y_true = pd.to_numeric(base[self.y_ic], errors="coerce")

            plot_df = pd.DataFrame({"x": x_pred, "y": y_true}).dropna()
            if len(plot_df) == 0:
                continue

            ax.scatter(
                plot_df["x"],
                plot_df["y"],
                marker=marker_cycle[i % len(marker_cycle)],
                s=60,
                edgecolors="black",
                facecolors="lightgrey",
                linewidths=1.2,
                alpha=0.95,
                label=model_name
            )

        ax.set_xlim(0, lims)
        ax.set_ylim(0, lims)
        ax.set_xlabel("Predicted critical global gradient [-]")
        ax.set_ylabel("Observed global gradient of failures [-]")
        ax.set_title("Observed vs predicted global gradients of failures")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()

        if export:
            fig.savefig(
                os.path.join(self.plots_dir, "all_models_failure_global_gradient_comparison.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()

        return None

    def plot_metric_bars(self, export=True):
        """
        Make a few summary bar plots from the exported/computed metrics.
        """

        metrics = self.compute_existing_model_metrics(export=export)
        reg = metrics["regression"].copy()
        cls = metrics["classification"].copy()

        # 1) RMSE on failures for dH
        if "RMSE_dH_fail" in reg.columns and len(reg) > 0:
            reg_plot = reg.sort_values("RMSE_dH_fail", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(reg_plot["model"], reg_plot["RMSE_dH_fail"])
            ax.set_xlabel("RMSE on failures for ΔH")
            ax.set_ylabel("Model")
            ax.set_title("Existing models ranked by ΔH RMSE on failures")
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
            fig.tight_layout()

            if export:
                fig.savefig(os.path.join(self.plots_dir, "rank_rmse_dH_failures.png"), dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

        # 2) MCC classification
        if "MCC" in cls.columns and len(cls) > 0:
            cls_plot = cls.sort_values("MCC", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(cls_plot["model"], cls_plot["MCC"])
            ax.set_xlabel("Matthews correlation coefficient")
            ax.set_ylabel("Model")
            ax.set_title("Existing models ranked by classification MCC")
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
            fig.tight_layout()

            if export:
                fig.savefig(os.path.join(self.plots_dir, "rank_classification_mcc.png"), dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

        return None

    def run_all_model_comparisons(
        self,
        do_metrics=True,
        do_prediction_plots=True,
        do_ic_pairs=True,
        do_metric_bars=True,
        continue_on_error=True,
        export=True,
    ):
        """Run the full legacy-model comparison workflow."""
        print("\nRunning model comparison...\n")

        tasks = {
            "compute_existing_model_metrics": (
                do_metrics,
                lambda: self.compute_existing_model_metrics(export=export)
            ),
            "plot_predictions_vs_real_dH": (
                do_prediction_plots,
                lambda: self.plot_existing_model_predictions_vs_real(
                    target="Water_level_diff",
                    export=export
                )
            ),
            "plot_predictions_vs_real_ic": (
                do_prediction_plots,
                lambda: self.plot_existing_model_predictions_vs_real(
                    target="Global_gradient",
                    export=export
                )
            ),
            "plot_ic_pairs_existing_models": (
                do_ic_pairs,
                lambda: self.plot_ic_pairs_existing_models(export=export)
            ),
            "plot_metric_bars": (
                do_metric_bars,
                lambda: self.plot_metric_bars(export=export)
            ),
        }

        results = {}

        for task_name, (enabled, task_function) in tasks.items():
            if not enabled:
                results[task_name] = "skipped"
                continue

            print(f"Running {task_name}...")
            try:
                task_output = task_function()
                results[task_name] = "done" if task_output is None else "done"
            except Exception as e:
                results[task_name] = f"failed: {e}"
                print(f"{task_name} failed: {e}")
                if not continue_on_error:
                    raise

        print("\nModel comparison finished.\n")
        return results
