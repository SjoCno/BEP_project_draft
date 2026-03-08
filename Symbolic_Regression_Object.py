from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os, sys, re
import pandas as pd
from pathlib import Path
import sympy as sp
pd.set_option("display.max_colwidth", None)
import reference_columns  # should now work
import symb_regress_func as sr

if "data-sources/" not in sys.path:
    sys.path.append("data-sources/")
if "results/" not in sys.path:
    sys.path.append("results/")
if "utils/" not in sys.path:
    sys.path.append("utils/")

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]                 # .../BEP_database
DATA_SOURCES = ROOT / "data-sources"


# Sanity check (optional during debugging)
if not (DATA_SOURCES / "reference_columns.py").exists():
    raise FileNotFoundError(f"Expected {DATA_SOURCES/'reference_columns.py'}")

# Ensure Python can import modules from data-sources
sys.path.insert(0, str(DATA_SOURCES))

from Database_object import Database

class Symbolic_Regression(Database):
    def __init__(self, df):
        self.df = df


    def get_equations(df, selection=None, failures_only=False, complexity=12, depth=6, save=True, title="symbolic_regression_results.txt"):
        data = df.copy()
        if failures_only:
            data = data[data["Survival/failure"]=="Failure"]
    
        features = reference_columns.modelling_features.copy()
        if "Global_gradient" in features: features.remove("Global_gradient")
    
        scaler = MinMaxScaler().set_output(transform="pandas")
    
        if selection is not None:
            X = data[selection]
        else:
            X = data[features]
    
        X = scaler.fit_transform(X)
        y = data['Survival/failure'].map({'Survival': 0, 'Failure': 1})
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        
        
        model = PySRRegressor(
                niterations=100,
                binary_operators=["+", "-", "*", "/", "pow"],
                unary_operators=["sqrt", "log", "exp"],   # see note about 'exp' below
                model_selection="best",
                #elementwise_loss="loss(x, y) = (x - y)^2",
                elementwise_loss=(
                    "loss(x, y) = -(y * log(1 / (1 + exp(-x))) + "
                    "(1 - y) * log(1 - 1 / (1 + exp(-x))))"
                ),
                maxsize=complexity,
                maxdepth=depth,
                verbosity=0,
                random_state=10,
                progress=False,
                temp_equation_file=True,
                deterministic=True,
                parallelism="serial",
                constraints={"pow": (-1, 1)},          # exponent subtree size ≤ 1
                complexity_of_variables=2,              # variables have complexity 2 (not allowed as exponent)
                nested_constraints={"exp": {"exp": 0}},
            )
    
        print("\n\n\nTraining!!! \n\n\n")
        model.fit(X_train, y_train)
    
        def round_coeffs(equation, decimals=4):
            return re.sub(
                r"([-+]?\d*\.\d+|\d+)",
                lambda m: str(round(float(m.group()), decimals)),
                equation,
            )
    
        equations = model.equations_.sort_values(by="loss")
    
        print("\n\n=== Best Discovered Equations (rounded) ===\n")
        lines = []
        for _, row in equations.iterrows():
            eqn = round_coeffs(row["equation"], 4)
            msg = f"Complexity = {row['complexity']}; Loss = {row['loss']:.2e};\ny = {eqn}\n"
            print(msg)
            lines.append(msg)
    
        if save:
            os.makedirs("SR_results", exist_ok=True)
            out_path = os.path.join("SR_results", title)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("=== Symbolic Regression Results ===\n\n")
                f.writelines(lines)
            print(f"\n Results saved to {out_path}\n")
    
        return model
    
    def preprocess(df, feature_cols):
        """Scale the selected feature columns to [0,1] and return the scaled dataframe."""
        df_copy = df.copy()
    
        # Make features numeric
        for col in feature_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    
        # Drop rows with missing data in these columns
        df_copy = df_copy.dropna(subset=feature_cols)
    
        # Scale features to [0,1]
        scaler = MinMaxScaler()
        df_copy[feature_cols] = scaler.fit_transform(df_copy[feature_cols])
    
        return df_copy
    
    
    original_df = pd.read_pickle("BEP_database_final.pkl")
    
    all_features = [
        'Water_level_diff', 'Seepage_length', 'Porosity', 'Aquifer_thickness',
        'Blanket_thickness', 'd10', 'd50', 'Uniformity_coefficient',
        'Hydraulic_conductivity_KC', 'Friction_angle', 'Bedding_angle'
    ]
    
    df = sr.preprocess(original_df, feature_cols=all_features)
    
    # --- paths ---
    base_dir = os.path.dirname(os.path.abspath(__file__))  # where this script lives
    root_dir = os.path.dirname(base_dir)                   # one level up (BEP_database)
    results_dir = os.path.join(root_dir, "results")        # always .../BEP_database/results
    os.makedirs(results_dir, exist_ok=True)
    
    combinations = [
        ["Water_level_diff", "Seepage_length"],
        ["Water_level_diff", "Seepage_length", "Aquifer_thickness"],
        ["Water_level_diff", "Seepage_length", "Blanket_thickness"],
        ["Water_level_diff", "Seepage_length", "Uniformity_coefficient"],
        ["Water_level_diff", "Seepage_length", "Aquifer_thickness", "Blanket_thickness"],
        ["Water_level_diff", "Seepage_length", "Aquifer_thickness", "Uniformity_coefficient"],
        ["Water_level_diff", "Seepage_length", "Blanket_thickness", "Uniformity_coefficient"],
        ["Water_level_diff", "Seepage_length", "Aquifer_thickness", "Blanket_thickness", "Uniformity_coefficient"],
    ]
    
    prev_cwd = os.getcwd()
    try:
        os.chdir(results_dir)
    
        for idx, selection in enumerate(combinations, start=1):
            title = f"SR_selection_{idx}__{'_'.join(selection)}.txt"
            print(f"\n>>> Running symbolic regression for: {selection}")
            sr.get_equations(df, selection=selection, title=title)
    
    finally:
        os.chdir(prev_cwd)
