from .Bayesian_logistic_object import BaseBayesianLogisticModel


class HLU_log(BaseBayesianLogisticModel):
    priors = {
        "c1": {"mu": 0, "sigma": 1},
        "c2": {"mu": 10, "sigma": 1},
        "c3": {"mu": 0, "sigma": 1},
        "c0": {"mu": 0.0, "sigma": 1},
    }
    feature_cols = ["Uniformity_coefficient", "Water_level_diff", "Seepage_length"]

    def core_equation(self, c, v):
        return (
            c["c1"] * v["Uniformity_coefficient"]
            + c["c2"] * (v["Water_level_diff"] ** c["c3"] - v["Seepage_length"])
            + c["c0"]
        )

    def _formula_latex(self):
        return r"$\mu = c_1 U_c + c_2(dH^{c_3} - L) + c_0$"


LOGISTIC_MODEL_REGISTRY = {
    "HLU_log": HLU_log,
}


def get_logistic_model_registry():
    return LOGISTIC_MODEL_REGISTRY.copy()