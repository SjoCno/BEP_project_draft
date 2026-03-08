"""Core database container and build pipeline for the BEP project.

This version preserves the original thesis functionality, but adds clearer
comments and docstrings and updates paths for the new folder layout.
"""
import pandas as pd
import numpy as np
import sys, os

from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from data.Failures import get_failure_df
from data.STOWA_data.STOWA_data import get_STOWA_df

from src.utils import reference_columns
from src.utils import mappings
from src.Enrichment_object import Enrichment


class Database():
    """Main in-memory database object used throughout the project."""
    def __init__(self):
        """Initialize the object and prepare the main internal state."""
        self._df = pd.DataFrame(columns=self._get_reference_columns())
    
    @property
    def get_dataframe(self):
        """Return a defensive copy of the current dataframe."""
        return self._df.copy()
    
    def add_column(self, name, values):
        """Add a new column to the database dataframe."""
        if name in self._df.columns:
            raise ValueError(f"{name} already exists.")
        self._df[name] = values
    
    def update(self, col, mask, value):
        """Update values in a column for rows selected by a mask function."""
        self._df.loc[mask(self._df), col] = value
        
    def assign(self, col, func, mask=None):
        """Assign a column using a callable, optionally only on a masked subset."""
        if mask is not None:
            self._df.loc[mask(self._df), col] = func(self._df.loc[mask(self._df)])
        else:
            self._df[col] = func(self._df)
    
    def export(self):        
        """Write the current dataframe to the exported pickle file."""
        export_path = "src/BEP_database.pkl"
        if os.path.exists(export_path):
            os.remove(export_path)
        self._df.to_pickle(export_path)
        
        return None
    
    def build_database(self, force_rebuild: bool=False, exported_filepath: str="src/BEP_database.pkl", do_enrichment: bool=True, enrichment_kwargs: dict | None=None, do_bayesian_imputation: bool=True, bayesian_name: str="default", bayesian_plot: bool=False, bayesian_complicated: bool=False):
        """
        Build the BEP database.
    
        Pipeline:
        1) Try loading existing exported database
        2) Build raw database from source files
        3) Enrich raw data (BRO, GeoTOP, RWS, etc.)
        4) Finalize modelling values
        5) Run imputation on modelling columns
        6) Compute derived features
        7) Apply final filtering
        8) Ensure modelling columns are numeric
        9) Compute existing model predictions
        10) Reindex and export database
        """
    
        # --------------------------------------------------
        # 1. Try loading an existing exported database
        # --------------------------------------------------
        if not force_rebuild and os.path.exists(exported_filepath):
            try:
                df = pd.read_pickle(exported_filepath)
    
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"{exported_filepath} does not contain a pandas DataFrame.")
    
                df = df.copy()
                df.replace(np.nan, pd.NA, inplace=True)
    
                expected_cols = set(self._get_reference_columns())
                if not expected_cols.issubset(df.columns):
                    missing = expected_cols - set(df.columns)
                    raise ValueError(f"Exported database is missing columns: {sorted(missing)}")
    
                df.reset_index(drop=True, inplace=True)
    
                df["Year"] = (
                    pd.to_numeric(df["Year"], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                )
    
                df.sort_values(by=["Country", "Year"], inplace=True)
    
                self._df = df
                return None
    
            except Exception as e:
                print(f"Could not load '{exported_filepath}' ({e}). Rebuilding database...\n")
    
        # --------------------------------------------------
        # 2. Build raw database from source files
        # --------------------------------------------------
        failure_df = self.get_failure_data()
        stowa_df = self.get_STOWA_data()
    
        df = pd.concat([failure_df, stowa_df], ignore_index=True)
        df.replace(np.nan, pd.NA, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(-1).astype(int)
        df.sort_values(by=["Country", "Year"], inplace=True)
    
        self._df = df
    
        # remove unusable rows before enrichment
        self._pre_enrichment_filter()
    
        # --------------------------------------------------
        # 3. Enrich database (BRO, GeoTOP, RWS, etc.)
        # --------------------------------------------------
        if do_enrichment:
            if enrichment_kwargs is None:
                enrichment_kwargs = {
                    "do_bro": True,
                    "do_geotop_regis": True,
                    "do_rws": True,
                    "do_waterlevel_diff": True,
                    "do_bro_derivatives": True,
                    "do_geotop_derivatives": True,
                    "do_source_quantifier": True,
                    "days_from_date": 60,
                    "max_stations_to_check": 3,
                    "max_distance_to_check": 10000,
                    "max_depth_regis": 100,
                    "bro_delay": 1.0,
                    "bro_batch_size": 25,
                    "verbose": True,
                    "progress_every": 50,
                }
    
            enr = Enrichment(self)
            enr.enrich_entire_database(**enrichment_kwargs)
    
        # --------------------------------------------------
        # 4. Choose final modelling values
        # --------------------------------------------------
        self.finalize_modelling_values()
    
        # --------------------------------------------------
        # 5. Impute missing modelling values
        # --------------------------------------------------
        if do_bayesian_imputation:
            imputer = Imputation_unit(self)
            imputer.run_mean_imputation()
        # --------------------------------------------------
        # 6. Compute derived modelling features
        # --------------------------------------------------
        self.apply_extra_features()
    
        # --------------------------------------------------
        # 7. Final filtering of database
        # --------------------------------------------------
        self._post_enrichment_filter()
    
        # --------------------------------------------------
        # 8. Ensure modelling features are numeric
        # --------------------------------------------------
        modelling_cols = self._get_all_modelling_features()
    
        self._df[modelling_cols] = (
            self._df[modelling_cols]
            .apply(pd.to_numeric, errors="coerce")
        )
    
        # --------------------------------------------------
        # 9. Compute predictions of existing models
        # --------------------------------------------------
        self.fill_existing_model_predictions(overwrite=True, verbose=True)
    
        # --------------------------------------------------
        # 10. Reindex and export database
        # --------------------------------------------------
        self._df.reset_index(drop=True, inplace=True)
    
        self.export()
    
        return None
    
    ##################################################
    #                 Retrieve data                  #
    ##################################################
    
    def get_failure_data(self):
        """Load the failure data source and merge it into the database."""
        return get_failure_df()
    
    def get_STOWA_data(self):
        """Load the STOWA data source and merge it into the database."""
        return get_STOWA_df()
    
    def _get_reference_columns(self):
        """Return the full reference column list used to initialize the database."""
        return reference_columns.reference_columns
    
    def _get_modelling_features(self):
        """Return the subset of features used for modelling."""
        return reference_columns.modelling_features
    
    def _get_all_modelling_features(self):
        """Return the broader modelling feature list, including derived features."""
        return reference_columns.all_modelling_features

    def _get_imputation_features(self):
        """Return the feature list that is eligible for imputation."""
        return reference_columns.imputation_features
    
    
    # ==================================================
    # Rewritten apply_extra_features
    # ==================================================
    
    def apply_extra_features(self):
    
        """Create additional engineered features used later in analysis or modelling."""
        d50 = self._df['d50'].apply(self._to_scalar)
        d60 = self._df['d60'].apply(self._to_scalar)
        d10 = self._df['d10'].apply(self._to_scalar)
        por = self._df['Porosity'].apply(self._to_scalar)
        phi = self._df['Friction_angle'].apply(self._to_scalar)
        k = self._df['Hydraulic_conductivity'].apply(self._to_scalar)
    
        self._df['Bedding_angle'] = pd.to_numeric(-8.125*np.log(d50)-38.77, errors='coerce')
        self._df['Tan_Bedding_angle'] = np.tan(self._df['Bedding_angle']*(np.pi/180))
        self._df['Tan_Friction_angle'] = np.tan(phi*(np.pi/180))
        self._df['Uniformity_coefficient'] = pd.to_numeric(d60/d10, errors='coerce')
    
        self._df['Kozeny_Carman'] = pd.to_numeric((d50**2 * por**3)/(180*(1-por)**2), errors='coerce')
        self._df['Hydraulic_conductivity_KC'] = self._df['Kozeny_Carman']/1.33e-7
    
        self._df['Permeability'] = np.nan
        self._df.loc[self._df['Permeability'].isna(), 'Permeability'] = self._df.loc[self._df['Permeability'].isna(), 'Kozeny_Carman'].astype(float)
        self._df.loc[self._df['Permeability'].isna(), 'Permeability'] = (k*1.33e-7).astype(float)
    
        dh = self._df['Water_level_diff'].apply(self._to_scalar)
        L = self._df['Seepage_length'].apply(self._to_scalar)
        self._df['Global_gradient'] = pd.to_numeric(dh/L, errors='coerce')
    
        return None
    
    
    # ==================================================
    # Rewritten filters
    # ==================================================
    
    def _pre_enrichment_filter(self):
        """Apply the initial row filters before running enrichment."""
        if "Blanket_thickness" in self._df.columns:
            blanket_thickness = self._df["Blanket_thickness"].apply(self._to_scalar)
            smallest_positive_thickness = blanket_thickness[blanket_thickness > 0].min() if (blanket_thickness > 0).any() else np.nan
            if pd.notna(smallest_positive_thickness):
                zero_thickness_mask = blanket_thickness == 0
                self._df.loc[zero_thickness_mask, "Blanket_thickness"] = [{"type": "point", "value": float(smallest_positive_thickness)} for _ in range(zero_thickness_mask.sum())]
        self._df = self._df[~self._df["Levee/dam"].str.contains("Dam", na=False)]
        self._df = self._df[~((self._df["Year"] == -1) & (self._df["Survival/failure"] == "Survival"))]
        return None
    
    def _post_enrichment_filter(self, say_words=True):
        """Apply cleanup filters after enrichment and feature creation."""
        water_level_diff = self._df["Water_level_diff"].apply(self._to_scalar)
        self._df = self._df[~water_level_diff.isna()]
        water_level_diff = self._df["Water_level_diff"].apply(self._to_scalar)
        self._df = self._df[water_level_diff > 0]
        data_columns = [column for column in self._df.columns if column.endswith("_data")]
        self._df = self._df.drop(self._df[(self._df["Survival/failure"] == "Survival") & (self._df["Source"] == "STOWA database") & self._df[data_columns].isna().all(axis=1)].index)
        return None
    
    def finalize_modelling_values(self):
        """
        Fill the modelling columns with final numeric values.
    
        For each modelling feature, this function checks the source columns in a fixed order:
        Source -> GeoTOP -> BRO
    
        The first valid dictionary found is converted to a number and written directly into
        the modelling column itself (so no extra *_value column is used).
    
        Expected dictionary formats:
            {"type": "point", "value": x}
            {"type": "range", "min": a, "max": b}
    
        Legacy format also supported:
            {"min": a, "max": b}
        """
    
        source_priority = ["Source", "GeoTOP", "BRO"]
    
        range_handling = {
            "Porosity": "mean",
            "Hydraulic_conductivity": "mean",
            "Friction_angle": "mean",
            "d10": "mean",
            "d50": "mean",
            "d60": "mean",
            "d70": "mean",
            "Blanket_thickness": "mean",
            "Seepage_length": "mean",
            "Water_level_diff": "mean",
        }
        default_range_handling = "mean"
    
        def dict_to_number(value_dict, how="mean"):
            """Convert a point/range dictionary to one numeric value."""
            if not isinstance(value_dict, dict):
                return np.nan
    
            value_type = value_dict.get("type")
    
            if value_type == "point":
                return pd.to_numeric(value_dict.get("value"), errors="coerce")
    
            if value_type == "range" or ("min" in value_dict and "max" in value_dict):
                min_value = pd.to_numeric(value_dict.get("min"), errors="coerce")
                max_value = pd.to_numeric(value_dict.get("max"), errors="coerce")
    
                if pd.isna(min_value) or pd.isna(max_value):
                    return np.nan
                if how == "min":
                    return float(min_value)
                if how == "max":
                    return float(max_value)
                return float((min_value + max_value) / 2.0)
    
            return np.nan
    
        def get_best_source_dict(row, feature):
            """Return the first valid source dictionary based on source_priority."""
            for source_name in source_priority:
                column_name = f"{source_name}_{feature}"
                if column_name not in self._df.columns:
                    continue
    
                value = row.get(column_name)
    
                if isinstance(value, dict) and value.get("type") in ["point", "range"]:
                    return value
    
                if isinstance(value, dict) and ("min" in value) and ("max" in value):
                    return {"type": "range", "min": value.get("min"), "max": value.get("max")}
    
            return None
    
        features = self._get_all_modelling_features()
    
        for feature in features:
            if feature not in self._df.columns:
                self._df[feature] = np.nan
    
        for idx in self._df.index:
            row = self._df.loc[idx]
    
            for feature in features:
                handling_method = range_handling.get(feature, default_range_handling)
                chosen_dict = get_best_source_dict(row, feature)
    
                if chosen_dict is not None:
                    self._df.at[idx, feature] = dict_to_number(chosen_dict, how=handling_method)
                    continue
    
                existing_value = row.get(feature)
    
                if isinstance(existing_value, dict):
                    self._df.at[idx, feature] = dict_to_number(existing_value, how=handling_method)
                else:
                    self._df.at[idx, feature] = pd.to_numeric(existing_value, errors="coerce")
    
        return None


    # ==================================================
    # Existing models
    # ==================================================

    @staticmethod
    def sellmeijer(d70, D_aq, K, L):    
        """Compute the Sellmeijer-based response for the current dataframe."""
        d70m = 0.208e-3
        nu = 1e-6
        D_over_L = (D_aq / L)
        
        FR = 0.25 * ((2650 / 1000)-1) * np.tan(37 * (np.pi/180))
        FS = d70m/((K*1e-7 * L)**(1./3)) * (d70/d70m)**0.4
        FG = 0.91*(D_over_L)**((0.28/((D_over_L**2.8)-1))+0.04)
        
        H_c = FR * FS * FG * L
        return H_c

    @staticmethod
    def sellmeijer_D_over_L(d70, D_over_L, K, L):    
        """Compute the Sellmeijer D/L-style response for the current dataframe."""
        d70m = 0.208e-3
        nu = 1e-6
        
        FR = 0.25 * ((2650 / 1000)-1) * np.tan(37 * (np.pi/180))
        FS = d70m/((K*1e-7 * L)**(1./3)) * (d70/d70m)**0.4
        FG = 0.91*(D_over_L)**((0.28/((D_over_L**2.8)-1))+0.04)
        
        H_c = FR * FS * FG * L
        return H_c

    @staticmethod
    def Bligh(d50, L):
        """Compute the Bligh-style response for the current dataframe."""
        conditions = [
            d50 < 0.15e-3,
            (d50 >= 0.15e-3) & (d50 < 0.30e-3),
            (d50 >= 0.30e-3) & (d50 < 2.00e-3),
            (d50 >= 2.0e-3) & (d50 < 16.00e-3),
            d50 >= 16.00e-3
        ]
        choices = [18, 15, 12, 9, 4]
        c = np.select(conditions, choices, default=12)
        H_c = L / c
        return H_c

    @staticmethod
    def schmertmann(d10, U, D_aq, L):
        """Helper method: schmertmann."""
        i_pmt = 0.1358 * U + 0.002
        
        anisotropy = 2
        relative_density = 1
        Lf = L / (anisotropy)**0.5
        D_over_Lf = D_aq / Lf
        
        C_S = (d10/0.20e-3)**0.2
        C_D = (D_over_Lf ** (0.2/(D_over_Lf**2-1))) / 1.4
        C_L = (1.524 / Lf)**0.2
        C_K = (1.5/anisotropy)**0.5
        C_gamma = 1 + 0.4 * (relative_density - 0.6)
        
        H_c = C_D * C_L * C_S * C_K * C_gamma * i_pmt * L
        return H_c

    @staticmethod
    def schmertmann_D_over_L(d10, U, D_over_L, L):
        """Helper method: schmertmann_D_over_L."""
        i_pmt = 0.1358 * U + 0.002
        
        anisotropy = 2
        relative_density = 1
        Lf = L / (anisotropy)**0.5
        
        C_S = (d10/0.20e-3)**0.2
        C_D = (D_over_L ** (0.2/(D_over_L**2-1))) / 1.4
        C_L = (1.524 / Lf)**0.2
        C_K = (1.5/anisotropy)**0.5
        C_gamma = 1 + 0.4 * (relative_density - 0.6)
        
        H_c = C_D * C_L * C_S * C_K * C_gamma * i_pmt * L
        return H_c

    @staticmethod
    def sd(d10, d50, D_aq, K, L):
        """Helper method: sd."""
        d10 = np.asarray(d10, dtype=float)
        d50 = np.asarray(d50, dtype=float)
        D_aq = np.asarray(D_aq, dtype=float)
        K    = np.asarray(K, dtype=float)
        L    = np.asarray(L, dtype=float)

        particle_reynolds_number = d50 * np.sqrt(1.65 * 9.81 * d50) / 1.33e-6

        R = particle_reynolds_number
        phi_cr = np.where(
            R <= 6.61,
            0.1414 * R**(-0.2306),
            np.where(
                R < 282.84,
                (1 + (0.0223 * R)**2.8358)**0.3542 / (3.0946 * R**0.6769),
                0.045
            )
        )

        alpha_r = 6.0
        alpha_f = 5.0
        lR = 18e-6

        S_pipe = np.sqrt(9.81) * (phi_cr * 1.65 * d10)**(1.5) / (1.33e-6 * np.sqrt(alpha_r))

        with np.errstate(divide='ignore', invalid='ignore'):
            lc_over_L = np.exp(-(alpha_f * D_aq / L)**2 * S_pipe)

        S_sand_min_S_pipe = (d50 * 1.33e-6) / (lR * K * D_aq)

        H_c = (S_pipe + (1.0 - lc_over_L) * (S_sand_min_S_pipe)) * L
        return H_c

    @staticmethod
    def icresult(L, D_aq):
        """Helper method: icresult."""
        H_c = (0.00016 * L - 0.00013 * D_aq + 0.029) * L
        return H_c
    
    
    def fill_existing_model_predictions(self, overwrite=True, verbose=True):
        """
        Calculate existing model predictions and store them in the dataframe.
    
        Filled columns:
        - H_c_sellmeijer, i_c_sellmeijer
        - H_c_bligh, i_c_bligh
        - H_c_schmertmann, i_c_schmertmann
        - H_c_SD, i_c_SD
        """
    
        if self._df is None or self._df.empty:
            if verbose:
                print("Database is empty, skipping model predictions.")
            return None
    
        df = self._df.copy()
    
        needed_columns = [
            "d70",
            "d50",
            "d10",
            "Aquifer_thickness",
            "Hydraulic_conductivity_KC",
            "Seepage_length",
            "Uniformity_coefficient",
        ]
    
        for col in needed_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
        L = df["Seepage_length"]
    
        if overwrite or "H_c_sellmeijer" not in df.columns:
            df["H_c_sellmeijer"] = self.sellmeijer(df["d70"], df["Aquifer_thickness"], df["Hydraulic_conductivity_KC"], df["Seepage_length"])
            df["i_c_sellmeijer"] = df["H_c_sellmeijer"] / L
    
        if overwrite or "H_c_bligh" not in df.columns:
            df["H_c_bligh"] = self.Bligh(df["d50"], df["Seepage_length"])
            df["i_c_bligh"] = df["H_c_bligh"] / L
    
        if overwrite or "H_c_schmertmann" not in df.columns:
            df["H_c_schmertmann"] = self.schmertmann(df["d10"], df["Uniformity_coefficient"], df["Aquifer_thickness"], df["Seepage_length"])
            df["i_c_schmertmann"] = df["H_c_schmertmann"] / L
    
        if overwrite or "H_c_SD" not in df.columns:
            df["H_c_SD"] = self.sd(df["d10"], df["d50"], df["Aquifer_thickness"], df["Hydraulic_conductivity_KC"], df["Seepage_length"])
            df["i_c_SD"] = df["H_c_SD"] / L
    
        prediction_columns = [
            "H_c_sellmeijer", "i_c_sellmeijer",
            "H_c_bligh", "i_c_bligh",
            "H_c_schmertmann", "i_c_schmertmann",
            "H_c_SD", "i_c_SD",
        ]
    
        for col in prediction_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= 0, col] = np.nan
    
        self._df = df
    
        if verbose:
            print("Existing model predictions filled.")
    
        return None
    
    
    # ==================================================
    # Generate KML file
    # ==================================================
    
    def generate_kml_from_dataframe_manual_xml(
        self,
        df=None,
        latitude_col='Y/lat',
        longitude_col='X/lon',
        id_col='ID',
        output_filename="output_manual.kml"
    ):
        """
        Generate a KML file from a dataframe.
    
        If df is None, uses the current database dataframe.
        """
    
        if df is None:
            df = self._database.get_dataframe.copy()
    
        kml_header = """<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://earth.google.com/kml/2.2">
    <Document>
        <name>Generated Points from DataFrame (Manual XML)</name>
        <description>Points with assigned IDs generated without simplekml</description>
    """
    
        kml_footer = """</Document>
    </kml>
    """
        placemark_xml_parts = []
    
        for index, row in df.iterrows():
            try:
                lon = float(row[longitude_col])
                lat = float(row[latitude_col])
                placemark_id = str(row[id_col])
    
                coordinates_str = f"{lon},{lat},0"
    
                placemark_template = f"""    <Placemark>
            <name>{placemark_id}</name>
            <Point>
                <coordinates>{coordinates_str}</coordinates>
            </Point>
        </Placemark>
    """
                placemark_xml_parts.append(placemark_template)
    
            except KeyError as e:
                print(f"Error: Column '{e}' not found in DataFrame.")
                return None
            except ValueError as e:
                print(f"Error converting coordinate data in row {index}: {e}. Skipping row.")
                continue
            except Exception as e:
                print(f"Unexpected error in row {index}: {e}. Skipping row.")
                continue
    
        full_kml_content = kml_header + "".join(placemark_xml_parts) + kml_footer
    
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(full_kml_content)
            print(f"KML file '{output_filename}' generated successfully.")
        except IOError as e:
            print(f"Error writing KML file: {e}")
    
        return None
    
    def _to_scalar(self, value, how="mean"):
        """
        Convert supported values to a single numeric scalar.
    
        Supports:
        - plain numeric values
        - {'type': 'point', 'value': x}
        - {'type': 'range', 'min': a, 'max': b}
        - legacy {'min': a, 'max': b}
        """
    
        if value is None:
            return np.nan
    
        # already numeric
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    
        # pandas missing
        try:
            if pd.isna(value):
                return np.nan
        except Exception:
            pass
    
        # dict forms
        if isinstance(value, dict):
            vtype = value.get("type")
    
            if vtype == "point":
                return pd.to_numeric(value.get("value"), errors="coerce")
    
            if vtype == "range":
                a = pd.to_numeric(value.get("min"), errors="coerce")
                b = pd.to_numeric(value.get("max"), errors="coerce")
                if pd.isna(a) or pd.isna(b):
                    return np.nan
                if how == "min":
                    return float(a)
                if how == "max":
                    return float(b)
                return float((a + b) / 2.0)
    
            # legacy range dict
            if ("min" in value) and ("max" in value):
                a = pd.to_numeric(value.get("min"), errors="coerce")
                b = pd.to_numeric(value.get("max"), errors="coerce")
                if pd.isna(a) or pd.isna(b):
                    return np.nan
                if how == "min":
                    return float(a)
                if how == "max":
                    return float(b)
                return float((a + b) / 2.0)
    
        return pd.to_numeric(value, errors="coerce")