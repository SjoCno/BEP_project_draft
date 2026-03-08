import numpy as np
import pandas as pd

from .Bayesian_linear_object import BaseLinearBayesianModel


class Dbl_exp(BaseLinearBayesianModel):
    priors = {
        "c1": {"mu": 0.010, "sigma": 0.010},
        "c2": {"mu": 3.0,   "sigma": 1.0},
        "c3": {"mu": 0.015, "sigma": 0.010},
        "c0": {"mu": 0.020, "sigma": 0.010},
    }
    feature_cols = ["Blanket_thickness"]

    def core_equation(self, c, v):
        return c["c0"] + c["c1"] * (1 - np.exp(-c["c2"] * v["Blanket_thickness"])) + c["c3"] * v["Blanket_thickness"]

    def _formula_latex(self):
        return r"$i_c = c_1(1 - e^{-c_2 D_{bl}}) + c_3 D_{bl} + c_0$"


class Dbl_lin(BaseLinearBayesianModel):
    priors = {
        "c1": {"mu": 0, "sigma": 1},
        "c0": {"mu": 0.050, "sigma": 1},
    }
    feature_cols = ["Blanket_thickness"]

    def core_equation(self, c, v):
        return c["c1"] * v["Blanket_thickness"] + c["c0"]

    def _formula_latex(self):
        return r"$i_c = c_1 D_{bl} + c_0$"


class Daq_exp(BaseLinearBayesianModel):
    priors = {
        "c1": {"mu": 0.25, "sigma": 1},
        "c2": {"mu": -50, "sigma": 10},
        "c0": {"mu": 0.050, "sigma": 1},
    }
    feature_cols = ["Aquifer_thickness"]

    def core_equation(self, c, v):
        return c["c1"] * np.exp(c["c2"] * v["Aquifer_thickness"]) + c["c0"]

    def _formula_latex(self):
        return r"$i_c = c_1 e^{c_2 D_{aq}} + c_0$"


class Daq_over_L_lin(BaseLinearBayesianModel):
    priors = {
        "c1": {"mu": 0, "sigma": 1},
        "c0": {"mu": 0.050, "sigma": 1},
    }
    feature_cols = ["D_over_L"]

    def _prepare_dataframe(self, df):
        df = df.copy()
        df["D_over_L"] = pd.to_numeric(df["Aquifer_thickness"], errors="coerce") / pd.to_numeric(df["Seepage_length"], errors="coerce")
        return df

    def core_equation(self, c, v):
        return c["c1"] * v["D_over_L"] + c["c0"]

    def _formula_latex(self):
        return r"$i_c = c_1 \frac{D_{aq}}{L} + c_0$"


class LDaq_exp(BaseLinearBayesianModel):
    priors = {
        "c1": {"mu": 0.25, "sigma": 1},
        "c2": {"mu": -50, "sigma": 10},
        "c3": {"mu": 0.25, "sigma": 1},
        "c4": {"mu": -50, "sigma": 10},
        "c0": {"mu": 0.050, "sigma": 1},
    }
    feature_cols = ["Seepage_length", "Aquifer_thickness"]

    def core_equation(self, c, v):
        return (
            c["c1"] * np.exp(c["c2"] * v["Seepage_length"])
            + c["c3"] * np.exp(c["c4"] * v["Aquifer_thickness"])
            + c["c0"]
        )

    def _formula_latex(self):
        return r"$i_c = c_1 e^{c_2 L} + c_3 e^{c_4 D_{aq}} + c_0$"


class LDaq_linear(BaseLinearBayesianModel):
    priors = {
        "c1": {"mu": 0, "sigma": 1},
        "c2": {"mu": 0, "sigma": 1},
        "c0": {"mu": 0.050, "sigma": 1},
    }
    feature_cols = ["Seepage_length", "Aquifer_thickness"]

    def core_equation(self, c, v):
        return c["c1"] * v["Seepage_length"] + c["c2"] * v["Aquifer_thickness"] + c["c0"]

    def _formula_latex(self):
        return r"$i_c = c_1 L + c_2 D_{aq} + c_0$"


class LDblDaq_linear(BaseLinearBayesianModel):
    priors = {
        "c1": {"mu": 0, "sigma": 1},
        "c2": {"mu": 0, "sigma": 1},
        "c3": {"mu": 0, "sigma": 1},
        "c0": {"mu": 0.050, "sigma": 1},
    }
    feature_cols = ["Seepage_length", "Blanket_thickness", "Aquifer_thickness"]

    def core_equation(self, c, v):
        return (
            c["c1"] * v["Seepage_length"]
            + c["c2"] * v["Blanket_thickness"]
            + c["c3"] * v["Aquifer_thickness"]
            + c["c0"]
        )

    def _formula_latex(self):
        return r"$i_c = c_1 L + c_2 D_{bl} + c_3 D_{aq} + c_0$"


LINEAR_MODEL_REGISTRY = {
    "Dbl_exp": Dbl_exp,
    "Dbl_lin": Dbl_lin,
    "Daq_exp": Daq_exp,
    "Daq_over_L_lin": Daq_over_L_lin,
    "LDaq_exp": LDaq_exp,
    "LDaq_linear": LDaq_linear,
    "LDblDaq_linear": LDblDaq_linear,
}


def get_linear_model_registry():
    return LINEAR_MODEL_REGISTRY.copy()