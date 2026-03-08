import pandas as pd
import numpy as np
import os, ddlpy, requests, math, re
from datetime import datetime, timedelta
from pyproj import Transformer
import xml.etree.ElementTree as ET
from bromodels import RegisColumn
from bromodels.GTM.GeoTop import GeoTopColumn, geotop_lithology_class
from bromodels.HGM.Regis import regis_stratigraphic_unit
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as st


class Enrichment:
    """
    Enrich the raw database with external data sources and derived geotechnical values.

    This class writes directly into the dataframe stored inside the Database object.
    The goal is not maximum abstraction, but to keep the enrichment workflow readable:
    fetch raw data first, then derive additional columns from those raw payloads.
    """

    def __init__(self, database):
        self._database = database

    @property
    def df(self):
        return self._database._df

    # ==================================================
    # Central loop
    # ==================================================

    def enrich_entire_database(
            self,
            do_bro: bool = True,
            do_geotop_regis: bool = True,
            do_rws: bool = True,
            do_waterlevel_diff: bool = True,
            do_bro_derivatives: bool = True,
            do_geotop_derivatives: bool = True,
            do_source_quantifier: bool = True,
            do_manual_seepage_lengths: bool = True,
            days_from_date: int = 60,
            max_stations_to_check: int = 3,
            max_distance_to_check: int = 10000,
            max_depth_regis: int = 100,
            bro_delay: float = 1.0,
            bro_batch_size: int = 25,
            verbose: bool = True,
            progress_every: int = 50,
        ):

            """
            Run the full enrichment loop over all rows in the database.

            The order is intentionally simple:
            1) fetch raw external data
            2) derive columns from those raw payloads
            3) optionally apply manually assigned seepage lengths
            """
    
            # Ensure raw cols exist
            if do_bro:
                for col in ["BHRG_data", "BHRP_data", "BHRGT_data"]:
                    if col not in self.df.columns:
                        self.df[col] = pd.NA
    
            if do_geotop_regis:
                for col in ["GeoTOP_data", "REGIS_data"]:
                    if col not in self.df.columns:
                        self.df[col] = pd.NA
    
            if do_rws:
                if "RWS_Waterlevel_data" not in self.df.columns:
                    self.df["RWS_Waterlevel_data"] = [{} for _ in range(len(self.df))]
    
            if do_manual_seepage_lengths:
                if "Seepage_length" not in self.df.columns:
                    self.df["Seepage_length"] = pd.NA
    
            n = len(self.df)
            for i, idx in enumerate(self.df.index, start=1):
                if verbose and (i == 1 or i % progress_every == 0 or i == n):
                    print(f"Enriching row {i}/{n} (idx={idx})...")
    
                row = self.df.loc[idx]
    
                # 1) RAW FETCHES
                if do_bro:
                    need_bro = (
                        self._is_missing(row.get("BHRG_data"))
                        or self._is_missing(row.get("BHRP_data"))
                        or self._is_missing(row.get("BHRGT_data"))
                    )
                    if need_bro:
                        self.enrich_with_bro_data(idx, delay=bro_delay, batch_size=bro_batch_size, verbose=verbose)
    
                if do_geotop_regis:
                    row = self.df.loc[idx]
                    need_gt_rg = self._is_missing(row.get("GeoTOP_data")) or self._is_missing(row.get("REGIS_data"))
                    if need_gt_rg:
                        self.enrich_with_geotop_regis_data(idx, max_depth=max_depth_regis)
    
                if do_rws:
                    row = self.df.loc[idx]
                    if self._is_missing(row.get("RWS_Waterlevel_data")):
                        self.enrich_with_rws_waterlevel_data(
                            idx,
                            days_from_date=days_from_date,
                            max_stations_to_check=max_stations_to_check,
                            max_distance_to_check=max_distance_to_check,
                            verbose=verbose
                        )
    
                # 2) DERIVED: Water level diff
                if do_waterlevel_diff:
                    self.get_waterlevel_differences(idx)
    
                # 3) DERIVED: BRO blanket thickness + BRO info
                if do_bro_derivatives:
                    row = self.df.loc[idx]
                    has_any_bro = (
                        not self._is_missing(row.get("BHRG_data"))
                        or not self._is_missing(row.get("BHRP_data"))
                        or not self._is_missing(row.get("BHRGT_data"))
                    )
                    if has_any_bro:
                        try:
                            self.get_soil_thicknesses_BRO(idx)
                        except Exception as e:
                            if verbose:
                                print(f"[Row {idx}] get_soil_thicknesses_BRO failed: {e}")
    
                        try:
                            self.get_info_from_BRO_data(idx)
                        except Exception as e:
                            if verbose:
                                print(f"[Row {idx}] get_info_from_BRO_data failed: {e}")
    
                # 4) DERIVED: GeoTOP info
                if do_geotop_derivatives:
                    row = self.df.loc[idx]
                    if not self._is_missing(row.get("GeoTOP_data")):
                        try:
                            self.get_info_from_GeoTOP(idx)
                        except Exception as e:
                            if verbose:
                                print(f"[Row {idx}] get_info_from_GeoTOP failed: {e}")
    
                # 5) DERIVED: Source quantifier
                if do_source_quantifier:
                    row = self.df.loc[idx]
                    if not self._is_missing(row.get("Soil type (source)")):
                        try:
                            self.source_quantifier(idx)
                        except Exception as e:
                            if verbose:
                                print(f"[Row {idx}] source_quantifier failed: {e}")
    
            if do_manual_seepage_lengths:
                self.apply_manual_seepage_lengths(overwrite=True, verbose=verbose)
    
            return None

    # ==================================================
    # RWS water level enrichment (raw dict stored as-is)
    # ==================================================

    def enrich_with_rws_waterlevel_data(self, idx, days_from_date=60, max_stations_to_check=3, max_distance_to_check=10000, verbose=True):
        """Fetch nearby Rijkswaterstaat water-level measurements and store the raw payload."""
        backup_name = "rws_waterlevel"
        backup_params = {
            "days_from_date": days_from_date,
            "max_stations_to_check": max_stations_to_check,
            "max_distance_to_check": max_distance_to_check,
        }

        if 'RWS_Waterlevel_data' not in self.df.columns:
            self.df['RWS_Waterlevel_data'] = [{} for _ in range(len(self.df))]

        row = self.df.loc[idx]
        backup_df = self._load_backup_df(backup_name, backup_params)
        cached_payload = self._find_backup_payload(backup_df, row)

        if cached_payload is not None:
            self.df.at[idx, 'RWS_Waterlevel_data'] = pd.NA if cached_payload == "__MISSING__" else cached_payload
            if verbose:
                print(f"[Row {idx}] RWS loaded from backup.")
            return None

        # Convert latitude/longitude from the database to the coordinate system used for station distance checks.
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
        locs_df_raw = ddlpy.locations()

        locs_df = locs_df_raw[
            (locs_df_raw['Grootheid.Code'] == 'WATHTE') |
            (locs_df_raw['Grootheid.Code'] == 'WATHTBRKD') |
            (locs_df_raw['Grootheid.Code'] == 'WATDTE')
        ].copy()

        if locs_df.empty:
            if verbose:
                print("No relevant RWS locations found.")
            return None

        locs_df.loc[:, 'X_rws_for_dist'] = locs_df['X']
        locs_df.loc[:, 'Y_rws_for_dist'] = locs_df['Y']

        if (row.get('Country') == 'The Netherlands') and pd.notna(row.get('X/lon')) and pd.notna(row.get('Y/lat')):
            if verbose:
                print(f"\n--- RWS Row {idx} ---")

            try:
                X_src_row = float(row['X/lon'])
                Y_src_row = float(row['Y/lat'])
                X_dst_row_transformed, Y_dst_row_transformed = transformer.transform(X_src_row, Y_src_row)
            except Exception as e:
                if verbose:
                    print(f"[Row {idx}] Coordinate transform failed: {e}")
                return None

            temp = locs_df.copy()
            temp['calculated_distance'] = np.sqrt(
                (temp['X_rws_for_dist'] - X_dst_row_transformed)**2 +
                (temp['Y_rws_for_dist'] - Y_dst_row_transformed)**2
            )

            sorted_stations = temp[temp['calculated_distance'] <= max_distance_to_check].sort_values('calculated_distance').head(max_stations_to_check)
            if sorted_stations.empty:
                if verbose:
                    print(f"[Row {idx}] No stations found within {max_distance_to_check}m.")
                self.df.at[idx, 'RWS_Waterlevel_data'] = pd.NA
                backup_df = self._upsert_backup_payload(backup_df, row, pd.NA)
                self._save_backup_df(backup_df, backup_name, backup_params)
                return None

            try:
                date_str = f"{row['Day+month']}-{int(row['Year'])}"
                date_obj = datetime.strptime(date_str, "%d-%m-%Y")
            except Exception as e:
                if verbose:
                    print(f"[Row {idx}] Date parsing failed: {e}")
                return None

            start_date = (date_obj - timedelta(days=days_from_date)).strftime("%Y-%m-%d")
            end_date = date_obj.strftime("%Y-%m-%d")

            found_payload = None
            for _, station_row in sorted_stations.iterrows():
                try:
                    measurements = ddlpy.measurements(station_row, start_date, end_date)
                    if measurements is not None and (not measurements.empty):
                        found_payload = {
                            'Grootheid.Code': station_row.get('Grootheid.Code'),
                            'Closest_station': station_row.get('Locatie_MessageID'),
                            'Distance_to_closest_station': float(np.round(station_row.get('calculated_distance', np.nan), 1)),
                            'Hoedanigheid.Code': station_row.get('Hoedanigheid.Code', 'Unknown'),
                            'Numeric_Value': measurements['Meetwaarde.Waarde_Numeriek'] / 1.0e2
                        }
                        self.df.at[idx, 'RWS_Waterlevel_data'] = found_payload
                        break
                except Exception:
                    continue

            if found_payload is None:
                self.df.at[idx, 'RWS_Waterlevel_data'] = pd.NA
                backup_df = self._upsert_backup_payload(backup_df, row, pd.NA)
            else:
                backup_df = self._upsert_backup_payload(backup_df, row, found_payload)

            self._save_backup_df(backup_df, backup_name, backup_params)

        return None
    # ==================================================
    # BRO raw fetch (per row)
    # ==================================================

    def enrich_with_bro_data(self, idx, delay=1, batch_size=25, verbose=True):
        """Fetch raw BRO borehole information for one row and cache the result."""
        backup_name = "bro_data"
        backup_params = {
            "delay": delay,
            "batch_size": batch_size,
        }

        for col in ["BHRG_data", "BHRP_data", "BHRGT_data"]:
            if col not in self.df.columns:
                self.df[col] = pd.NA

        row = self.df.loc[idx]
        backup_df = self._load_backup_df(backup_name, backup_params)
        cached_payload = self._find_backup_payload(backup_df, row)

        if cached_payload is not None:
            if cached_payload == "__MISSING__":
                self.df.at[idx, "BHRG_data"] = pd.NA
                self.df.at[idx, "BHRP_data"] = pd.NA
                self.df.at[idx, "BHRGT_data"] = pd.NA
            else:
                self.df.at[idx, "BHRG_data"] = cached_payload.get("BHRG_data", pd.NA)
                self.df.at[idx, "BHRP_data"] = cached_payload.get("BHRP_data", pd.NA)
                self.df.at[idx, "BHRGT_data"] = cached_payload.get("BHRGT_data", pd.NA)
            if verbose:
                print(f"[Row {idx}] BRO loaded from backup.")
            return None

        if row.get("Country") != "The Netherlands" or pd.isna(row.get("X/lon")) or pd.isna(row.get("Y/lat")):
            self.df.at[idx, "BHRG_data"] = pd.NA
            self.df.at[idx, "BHRP_data"] = pd.NA
            self.df.at[idx, "BHRGT_data"] = pd.NA
            backup_df = self._upsert_backup_payload(backup_df, row, pd.NA)
            self._save_backup_df(backup_df, backup_name, backup_params)
            return None

        try:
            # BRO services work in RD coordinates, so first convert lon/lat to EPSG:28992.
            x = float(row["X/lon"])
            y = float(row["Y/lat"])

            transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
            x_rd, y_rd = transformer.transform(x, y)

            self.df.at[idx, "BHRG_data"] = self._get_bhrg_data(x_rd, y_rd, verbose=verbose)
            self.df.at[idx, "BHRP_data"] = self._get_bhrp_data(x_rd, y_rd, verbose=verbose)
            self.df.at[idx, "BHRGT_data"] = self._get_bhrgt_data(x_rd, y_rd, verbose=verbose)

        except Exception as e:
            if verbose:
                print(f"[Row {idx}] BRO enrichment failed: {e}")
            self.df.at[idx, "BHRG_data"] = pd.NA
            self.df.at[idx, "BHRP_data"] = pd.NA
            self.df.at[idx, "BHRGT_data"] = pd.NA

        payload = {
            "BHRG_data": self.df.at[idx, "BHRG_data"],
            "BHRP_data": self.df.at[idx, "BHRP_data"],
            "BHRGT_data": self.df.at[idx, "BHRGT_data"],
        }
        if self._is_missing(payload["BHRG_data"]) and self._is_missing(payload["BHRP_data"]) and self._is_missing(payload["BHRGT_data"]):
            payload = pd.NA

        backup_df = self._upsert_backup_payload(backup_df, row, payload)
        self._save_backup_df(backup_df, backup_name, backup_params)
        return None
    # --- BRO helpers (unchanged output structure) ---

    def _get_bhrg_data(self, x, y, c_system='RD', verbose=True):
        bbox_size = 400
        half_size = bbox_size / 2
        bbox = f"{x - half_size},{y - half_size},{x + half_size},{y + half_size}"

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetFeatureInfo",
            "BBOX": bbox,
            "CRS": "EPSG:28992",
            "WIDTH": 101,
            "HEIGHT": 101,
            "LAYERS": "bhrg",
            "STYLES": "",
            "FORMAT": "image/png",
            "QUERY_LAYERS": "bhrg",
            "INFO_FORMAT": "application/json",
            "I": 50,
            "J": 50,
            "FEATURE_COUNT": 1
        }

        wms_url = "https://service.pdok.nl/bzk/bro-geologisch-booronderzoek/wms/v1_0"
        resp = requests.get(wms_url, params=params)
        resp.raise_for_status()
        wms_response_json = resp.json()
        features = wms_response_json.get("features", [])
        if not features:
            return pd.NA

        properties = features[0].get("properties", {})
        bro_id = properties.get("broId")
        if not bro_id:
            return pd.NA

        info_url = f"https://publiek.broservices.nl/sr/bhrg/v3/objects/{bro_id}"
        headers = {"Accept": "application/xml"}
        r = requests.get(info_url, headers=headers)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        ns = {
            "bhrgcom": "http://www.broservices.nl/xsd/bhrgcommon/3.1",
            "gml": "http://www.opengis.net/gml/3.2"
        }

        def get_text(node, path):
            el = node.find(path, ns)
            return el.text.strip() if el is not None and el.text else "N/A"

        location = get_text(root, ".//gml:Point/gml:pos")
        if location != "N/A":
            location = list(map(float, location.strip().split()))
        depth = get_text(root, ".//bhrgcom:finalDepthBoring")

        if c_system == 'RD' and location != "N/A":
            dist = math.hypot(y - float(location[1]), x - float(location[0]))
        else:
            dist = -999

        layers = root.findall(".//bhrgcom:Layer", ns)
        layer_info = []
        for layer in layers:
            begin = get_text(layer, ".//bhrgcom:upperBoundary")
            end = get_text(layer, ".//bhrgcom:lowerBoundary")
            soil_name = get_text(layer, ".//bhrgcom:soilNameNEN5104")
            color = get_text(layer, ".//bhrgcom:colour")
            layer_info.append(f"{begin}m to {end}m | Soil: {soil_name}, Colour: {color}")

        return {"distance": dist, "finalDepth": depth, "layers": layer_info, "broId": bro_id, "location": location}

    def _get_bhrp_data(self, x, y, c_system='RD', verbose=True):
        bbox_size = 2000
        half_size = bbox_size / 2
        bbox = f"{x - half_size},{y - half_size},{x + half_size},{y + half_size}"

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetFeatureInfo",
            "BBOX": bbox,
            "CRS": "EPSG:28992",
            "WIDTH": 101,
            "HEIGHT": 101,
            "LAYERS": "bhr_kenset",
            "STYLES": "",
            "FORMAT": "image/png",
            "QUERY_LAYERS": "bhr_kenset",
            "INFO_FORMAT": "application/json",
            "I": 50,
            "J": 50,
            "FEATURE_COUNT": 1
        }

        wms_url = "https://service.pdok.nl/bzk/brobhrpkenset/wms/v1_0"
        resp = requests.get(wms_url, params=params)
        resp.raise_for_status()

        if "application/json" not in resp.headers.get("Content-Type", ""):
            return pd.NA

        data = resp.json()
        features = data.get("features", [])
        if not features:
            return pd.NA

        props = features[0].get("properties", {})
        bro_id = props.get("bro_id")
        if not bro_id:
            return pd.NA

        info_url = f"https://publiek.broservices.nl/sr/bhrp/v2/objects/{bro_id}"
        headers = {"Accept": "application/xml"}
        r = requests.get(info_url, headers=headers)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        ns = {
            "bhrcommon": "http://www.broservices.nl/xsd/bhrcommon/2.0",
            "dsbhr": "http://www.broservices.nl/xsd/dsbhr/2.0",
            "gml": "http://www.opengis.net/gml/3.2"
        }

        def get_text(node, path):
            el = node.find(path, ns)
            return el.text.strip() if el is not None and el.text else "N/A"

        loc_el = root.find(".//dsbhr:deliveredLocation/bhrcommon:location/gml:pos", ns)
        location = loc_el.text.strip() if loc_el is not None and loc_el.text else "N/A"
        if location != "N/A":
            location = list(map(float, location.strip().split()))

        depth = get_text(root, ".//dsbhr:boring/bhrcommon:boredTrajectory/bhrcommon:endDepth")

        if c_system == 'RD' and location != "N/A":
            dist = math.hypot(y - float(location[1]), x - float(location[0]))
        else:
            dist = -999

        layers = root.findall(".//bhrcommon:soilLayer", ns)
        layer_info = []
        for layer in layers:
            upper = get_text(layer, ".//bhrcommon:upperBoundary")
            lower = get_text(layer, ".//bhrcommon:lowerBoundary")
            comp = layer.findall(".//bhrcommon:layerComponent", ns)
            details = []
            for c in comp:
                soil_name = get_text(c, ".//bhrcommon:soilType/bhrcommon:standardSoilName")
                details.append(f"Soil: {soil_name}")
            layer_info.append(f"{upper}m to {lower}m | Components: {', '.join(details)}")

        return {"distance": dist, "finalDepth": depth, "layers": layer_info, "broId": bro_id, "location": location}

    def _get_bhrgt_data(self, x, y, c_system='RD', verbose=True):
        bbox_size = 2000
        half_size = bbox_size / 2
        bbox = f"{x - half_size},{y - half_size},{x + half_size},{y + half_size}"

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetFeatureInfo",
            "BBOX": bbox,
            "CRS": "EPSG:28992",
            "WIDTH": 101,
            "HEIGHT": 101,
            "LAYERS": "bhrgt_kenset",
            "STYLES": "",
            "FORMAT": "image/png",
            "QUERY_LAYERS": "bhrgt_kenset",
            "INFO_FORMAT": "application/json",
            "I": 50,
            "J": 50,
            "FEATURE_COUNT": 1
        }

        wms_url = "https://service.pdok.nl/bzk/bro-geotechnischbooronderzoek/wms/v1_0"
        resp = requests.get(wms_url, params=params)
        resp.raise_for_status()

        if "application/json" not in resp.headers.get("Content-Type", ""):
            return pd.NA

        data = resp.json()
        features = data.get("features", [])
        if not features:
            return pd.NA

        props = features[0].get("properties", {})
        bro_id = props.get("bro_id")
        if not bro_id:
            return pd.NA

        info_url = f"https://publiek.broservices.nl/sr/bhrgt/v2/objects/{bro_id}"
        headers = {"Accept": "application/xml"}
        r = requests.get(info_url, headers=headers)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        ns = {
            "bhrgtcom": "http://www.broservices.nl/xsd/bhrgtcommon/2.1",
            "dsbhr": "http://www.broservices.nl/xsd/dsbhr-gt/2.1",
            "gml": "http://www.opengis.net/gml/3.2"
        }

        def get_text(node, path):
            el = node.find(path, ns)
            return el.text.strip() if el is not None and el.text else "N/A"

        loc_el = root.find(".//dsbhr:deliveredLocation/bhrgtcom:location/gml:Point/gml:pos", ns)
        location = loc_el.text.strip() if loc_el is not None and loc_el.text else "N/A"
        if location != "N/A":
            location = list(map(float, location.strip().split()))

        if c_system == 'RD' and location != "N/A":
            dist = math.hypot(y - float(location[1]), x - float(location[0]))
        else:
            dist = -999

        depth = get_text(root, ".//dsbhr:boring/bhrgtcom:finalDepthBoring")

        layers = root.findall(".//bhrgtcom:layer", ns)
        layer_info = []
        for layer in layers:
            upper = get_text(layer, "bhrgtcom:upperBoundary")
            lower = get_text(layer, "bhrgtcom:lowerBoundary")
            soil_name = get_text(layer, "bhrgtcom:soil/bhrgtcom:soilNameNEN5104")
            layer_info.append(f"{upper}m to {lower}m | Soil: {soil_name}")

        return {"distance": dist, "finalDepth": depth, "layers": layer_info, "broId": bro_id, "location": location}

    # ==================================================
    # GeoTOP + REGIS raw
    # ==================================================

    def enrich_with_geotop_regis_data(self, idx, max_depth=100):
        """Fetch GeoTOP and REGIS profiles for one row and cache the raw result."""
        backup_name = "geotop_regis_data"
        backup_params = {
            "max_depth": max_depth,
        }

        for col in ["GeoTOP_data", "REGIS_data"]:
            if col not in self.df.columns:
                self.df[col] = pd.NA

        row = self.df.loc[idx]
        backup_df = self._load_backup_df(backup_name, backup_params)
        cached_payload = self._find_backup_payload(backup_df, row)

        if cached_payload is not None:
            if cached_payload == "__MISSING__":
                self.df.at[idx, "GeoTOP_data"] = pd.NA
                self.df.at[idx, "REGIS_data"] = pd.NA
            else:
                self.df.at[idx, "GeoTOP_data"] = cached_payload.get("GeoTOP_data", pd.NA)
                self.df.at[idx, "REGIS_data"] = cached_payload.get("REGIS_data", pd.NA)
            return None

        x = row.get("X/lon")
        y = row.get("Y/lat")

        if pd.isna(x) or pd.isna(y):
            self.df.at[idx, "GeoTOP_data"] = np.nan
            self.df.at[idx, "REGIS_data"] = np.nan
            backup_df = self._upsert_backup_payload(backup_df, row, pd.NA)
            self._save_backup_df(backup_df, backup_name, backup_params)
            return None

        try:
            # GeoTOP and REGIS also use Dutch RD coordinates.
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
            x_rd, y_rd = transformer.transform(float(x), float(y))

            self.df.at[idx, "GeoTOP_data"] = self._get_geotop_profile(x_rd, y_rd)
            self.df.at[idx, "REGIS_data"] = Enrichment._get_regis_profile(x_rd, y_rd, max_depth=max_depth)
        except Exception as e:
            self.df.at[idx, "GeoTOP_data"] = f"Error: {e}"
            self.df.at[idx, "REGIS_data"] = f"Error: {e}"

        payload = {
            "GeoTOP_data": self.df.at[idx, "GeoTOP_data"],
            "REGIS_data": self.df.at[idx, "REGIS_data"],
        }
        if self._is_missing(payload["GeoTOP_data"]) and self._is_missing(payload["REGIS_data"]):
            payload = pd.NA

        backup_df = self._upsert_backup_payload(backup_df, row, payload)
        self._save_backup_df(backup_df, backup_name, backup_params)
        return None
    def _get_geotop_profile(self, x, y):
        """Read the GeoTOP profile at one RD coordinate and return simplified layers."""
        ds_gt = GeoTopColumn(x=x, y=y)
        if 'z' not in ds_gt or 'lithok' not in ds_gt or len(ds_gt['z']) == 0 or len(ds_gt['lithok']) == 0:
            return []

        z_gt = ds_gt['z'].values
        lith = ds_gt['lithok'].values

        lith_df = geotop_lithology_class()
        code_idx_to_class = dict(enumerate(lith_df['LITHO_CLASS_CD']))
        code_to_name = dict(zip(lith_df['LITHO_CLASS_CD'], lith_df['DESCRIPTION']))
        lith_names = [code_to_name.get(code_idx_to_class.get(int(c), ''), 'unknown') for c in lith]

        if len(lith_names) == 0 or len(z_gt) == 0:
            return []

        layers = []
        current_lith = lith_names[0]
        start_z = z_gt[0]
        for i in range(1, len(z_gt)):
            if lith_names[i] != current_lith:
                end_z = z_gt[i]
                layers.append({'top': start_z, 'bottom': end_z, 'lithology': current_lith, 'thickness': end_z - start_z})
                current_lith = lith_names[i]
                start_z = z_gt[i]
        layers.append({'top': start_z, 'bottom': z_gt[-1], 'lithology': current_lith, 'thickness': z_gt[-1] - start_z})

        return layers

    def _get_regis_profile(self, x, y, max_depth=100):
        """Read the REGIS profile at one RD coordinate and simplify it up to max_depth."""
        ds_rg = RegisColumn(x=x, y=y)
        if len(ds_rg['top']) == 0 or len(ds_rg['bottom']) == 0:
            return []

        top = ds_rg['top'].values
        bottom = ds_rg['bottom'].values
        thickness = np.where((top == -9999) | (bottom == -9999), 0, top - bottom)
        kh = ds_rg['kh'].values
        kv = ds_rg['kv'].values
        layer_codes = [b.tobytes().decode("utf-8").rstrip("\x00").strip() for b in ds_rg['layer'].values]

        # keep your mapping as-is (shortened assumption: you already had it)
        # (If you want, we can keep the full mapping block here exactly like your original file.)
        # For safety: just use codes directly if mapping missing.
        codes_fixed = layer_codes

        regis_df = regis_stratigraphic_unit()
        code_to_name = dict(zip(regis_df['HYD_UNIT_CD'], regis_df['DESCRIPTION']))
        layer_names = [code_to_name.get(code, 'unknown') for code in codes_fixed]

        valid = (thickness > 0) & (kh != -9999) & (kv != -9999)
        thickness = thickness[valid]
        codes = np.array(codes_fixed)[valid]
        names = np.array(layer_names)[valid]
        kh = kh[valid]
        kv = kv[valid]

        cum = 0
        out = []
        for i in range(len(thickness)):
            if cum + thickness[i] <= max_depth:
                out.append({'code': codes[i], 'name': names[i], 'thickness': thickness[i], 'kh': kh[i], 'kv': kv[i]})
                cum += thickness[i]
            else:
                rem = max_depth - cum
                if rem > 0:
                    out.append({'code': codes[i], 'name': names[i], 'thickness': rem, 'kh': kh[i], 'kv': kv[i]})
                break

        return out

    # ==================================================
    # Water level difference (stores dict-style point)
    # ==================================================

    def get_waterlevel_differences(self, idx, force_recalculate=True):
        """
        Compute Water_level_diff strictly as:
            Water_level_from_NAP - Altitude
    
        If both ingredients are available, this overwrites any existing value.
        This prevents stale or imputed values from surviving when the true
        physical difference can be computed.
        """
        # Make sure all columns used in the calculation exist before doing any work.
        for col in ["Water_level_from_NAP", "Altitude", "Water_level_diff"]:
            if col not in self.df.columns:
                self.df[col] = pd.NA
    
        row = self.df.loc[idx]
    
        # --------------------------------------------------
        # 1) Water level from RWS data
        # --------------------------------------------------
        rws = row.get("RWS_Waterlevel_data")
        wl_from_nap = None
    
        try:
            if isinstance(rws, dict) and "Numeric_Value" in rws and rws["Numeric_Value"] is not None:
                numeric_vals = pd.to_numeric(rws["Numeric_Value"], errors="coerce")
                numeric_vals = numeric_vals[~pd.isna(numeric_vals)]
                if len(numeric_vals) > 0:
                    wl_from_nap = float(np.max(numeric_vals))
        except Exception:
            wl_from_nap = None
    
        self.df.at[idx, "Water_level_from_NAP"] = wl_from_nap
    
        # --------------------------------------------------
        # 2) Altitude from sampled_hegihts.csv
        # --------------------------------------------------
        if not hasattr(self, "_heights_df_cache") or self._heights_df_cache is None:
            heights_df = pd.read_csv("sampled_hegihts.csv").drop(columns=["x_coord", "y_coord"], errors="ignore")
            heights_df = heights_df.rename(columns={"Column27": "ID", "heights1": "Heights"})
            self._heights_df_cache = heights_df.set_index("ID")
    
        row = self.df.loc[idx]
        alt = pd.to_numeric(row.get("Altitude"), errors="coerce")
    
        if pd.isna(alt):
            row_id = row.get("ID")
            if pd.notna(row_id) and row_id in self._heights_df_cache.index:
                alt_from_file = pd.to_numeric(self._heights_df_cache.at[row_id, "Heights"], errors="coerce")
                if pd.notna(alt_from_file):
                    self.df.at[idx, "Altitude"] = float(alt_from_file)
    
        # --------------------------------------------------
        # 3) Strict water level difference
        # --------------------------------------------------
        row = self.df.loc[idx]
        wl = pd.to_numeric(row.get("Water_level_from_NAP"), errors="coerce")
        alt = pd.to_numeric(row.get("Altitude"), errors="coerce")
    
        if pd.notna(wl) and pd.notna(alt):
            diff = float(wl - alt)
            if force_recalculate or self._is_missing(row.get("Water_level_diff")):
                self.df.at[idx, "Water_level_diff"] = self._make_point(diff)
        else:
            # Keep missing if one of the ingredients is missing
            self.df.at[idx, "Water_level_diff"] = pd.NA
    
        return None

    # ==================================================
    # Mapping-based material properties (now return dict-style range)
    # ==================================================

    def _write_unmapped_to_file(self, description, filepath):
        try:
            with open(filepath, "r") as f:
                existing = set(line.strip() for line in f)
        except FileNotFoundError:
            existing = set()

        if description not in existing:
            with open(filepath, "a") as f:
                f.write(f"{description}\n")

        return None

    def get_porosity_by_description(self, description):
        """Map a soil description to a porosity range dictionary."""
        from mappings import porosity_mapping
        if not isinstance(description, str):
            return None

        if description.strip() not in porosity_mapping:
            self._write_unmapped_to_file(description, r"src/unmapped_descriptions/unmapped_descriptions_porosity.txt")
            return None

        mapped = porosity_mapping.get(description.strip())
        porosity_table = {
            "Sand; Coarse": {"min": 0.26, "max": 0.43},
            "Sand; Fine": {"min": 0.29, "max": 0.46},
            "Sand/Gravelly Sand": {"min": 0.22, "max": 0.43},
            "Silty Sands": {"min": 0.25, "max": 0.49},
            "Clayey Sands": {"min": 0.15, "max": 0.37},
            "Sandy Gravel": {"min": 0.21, "max": 0.32},
        }
        r = porosity_table.get(mapped)
        return self._ensure_value_dict(r)

    def get_friction_angle_by_description(self, description):
        """Map a soil description to a friction-angle range dictionary."""
        from mappings import friction_angle_mapping
        if not isinstance(description, str):
            return None

        if description.strip() not in friction_angle_mapping:
            self._write_unmapped_to_file(description, r"src/unmapped_descriptions/unmapped_descriptions_friction_angle.txt")
            return None

        mapped = friction_angle_mapping.get(description.strip())
        friction_angle_table = {
            "Well graded gravel, sandy gravel, with little or no fines": {"min": 33, "max": 40},
            "Sand": {"min": 37, "max": 38},
            "Loose sand": {"min": 29, "max": 30},
            "Medium sand": {"min": 30, "max": 36},
            "Dense sand": {"min": 36, "max": 41},
            "Silty sands": {"min": 32, "max": 35},
            "Clayey sands": {"min": 30, "max": 40},
            "Loamy sand, sandy clay Loam": {"min": 31, "max": 34},
        }
        r = friction_angle_table.get(mapped.strip()) if isinstance(mapped, str) else None
        return self._ensure_value_dict(r)

    def get_hydraulic_conductivity_by_description(self, description):
        """Map a soil description to a hydraulic-conductivity range dictionary."""
        from mappings import hydraulic_conductivity_mapping
        if not isinstance(description, str):
            return None

        if description.strip() not in hydraulic_conductivity_mapping:
            self._write_unmapped_to_file(description, r"C:/Users/Sjoer/Desktop/BEP_database/src/unmapped_descriptions/unmapped_descriptions_permeability.txt")
            return None

        mapped = hydraulic_conductivity_mapping.get(description.strip())
        hydraulic_conductivity_table = {
            "Fine Sand": {"min": 2e-7, "max": 2e-4},
            "Medium Sand": {"min": 9e-7, "max": 5e-4},
            "Coarse Sand": {"min": 9e-7, "max": 6e-3},
            "Silty Sand": {"min": 1e-8, "max": 5e-6},
            "Clayey Sand": {"min": 5.5e-9, "max": 5.5e-6},
            "Gravel": {"min": 3e-4, "max": 3e-2},
            "Sandy Gravel": {"min": 5e-4, "max": 5e-2},
            "Silty Gravel/Silty Sandy Gravel": {"min": 5e-8, "max": 5e-6},
        }
        r = hydraulic_conductivity_table.get(mapped.strip()) if isinstance(mapped, str) else None
        return self._ensure_value_dict(r)

    def get_particle_sizes_by_description(self, description):
        """Map a soil description to representative particle sizes."""
        from mappings import particle_size_mapping

        particle_size_table = {
            "Gravel": {"d10": 0.00090, "d50": 0.00270, "d60": 0.00300, "d70": 0.00340},
            "Coarse sand": {"d10": 0.000190, "d50": 0.000469, "d60": 0.000547, "d70": 0.000628},
            "Medium sand": {"d10": 0.000087, "d50": 0.000163, "d60": 0.000182, "d70": 0.000211},
            "Fine sand": {"d10": 0.000047, "d50": 0.000152, "d60": 0.000173, "d70": 0.000201},
            "Silty sand": {"d10": 0.000030, "d50": 0.000075, "d60": 0.000081, "d70": 0.000095},
            "Clayey sand": {"d10": 0.0000090, "d50": 0.0000183, "d60": 0.0000190, "d70": 0.0000220},
            "Sand": {"d10": 0.000087, "d50": 0.000163, "d60": 0.000182, "d70": 0.000211},
        }

        if not isinstance(description, str):
            return None

        if description.strip() not in particle_size_mapping:
            self._write_unmapped_to_file(description, r"src/unmapped_descriptions/unmapped_descriptions_particle_sizes.txt")
            return None

        mapped = particle_size_mapping.get(description.strip())
        return particle_size_table.get(mapped) if mapped else None

    # ==================================================
    # BRO-derived blanket thickness -> Blanket_thickness (dict range)
    # ==================================================

    def get_soil_thicknesses_BRO(self, idx, source_columns=['BHRG_data', 'BHRP_data', 'BHRGT_data']):
        """Estimate blanket thickness from the available BRO layer descriptions."""
        def get_layer_thickness(layer_str):
            match = re.match(r'([\d.]+)m to ([\d.]+)m', layer_str)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                return end - start
            return 0

        def extract_blanket_layers(layers):
            blanket_min = []
            blanket_max = []
            for layer_str in layers:
                match = re.search(r'Soil:\s*([^,]+)', layer_str)
                soil_type = match.group(1).strip() if match else ''
                thickness = get_layer_thickness(layer_str)

                if 'Klei' in soil_type:
                    blanket_min.append(thickness)
                    blanket_max.append(thickness)
                elif 'Zand' in soil_type:
                    if thickness <= 0.5:
                        blanket_min.append(thickness)
                        blanket_max.append(thickness)
                    elif thickness <= 1.0:
                        blanket_max.append(thickness)

            blanket_thickness_min = max(blanket_min) if blanket_min else 0.0
            blanket_thickness_max = max(blanket_max) if blanket_max else 0.0
            return blanket_thickness_min, blanket_thickness_max

        if "Blanket_thickness" not in self.df.columns:
            self.df["Blanket_thickness"] = pd.NA

        row = self.df.loc[idx]
        if not self._is_missing(row.get("Blanket_thickness")):
            return None

        min_vals, max_vals = [], []
        for col in source_columns:
            data = row.get(col)
            if isinstance(data, dict) and 'layers' in data:
                bmin, bmax = extract_blanket_layers(data['layers'])
                min_vals.append(bmin)
                max_vals.append(bmax)

        if min_vals or max_vals:
            self.df.at[idx, "Blanket_thickness"] = self._make_range(
                round(max(min_vals), 3) if min_vals else 0.0,
                round(max(max_vals), 3) if max_vals else 0.0
            )

        return None

    # ==================================================
    # BRO-derived soil params -> BRO_* (dicts)
    # ==================================================

    def get_info_from_BRO_data(self, idx):
        """Derive BRO-based soil property columns from the nearest available BRO payload."""
        # BRO_* columns (NO _range)
        for c in ["BRO_Porosity", "BRO_Hydraulic_conductivity", "BRO_Friction_angle", "BRO_d10", "BRO_d50", "BRO_d60", "BRO_d70"]:
            if c not in self.df.columns:
                self.df[c] = pd.NA

        row = self.df.loc[idx]

        # Use the closest available BRO dataset, because several BRO object types can exist for one location.
        candidates = {}
        for key in ["BHRG_data", "BHRP_data", "BHRGT_data"]:
            d = row.get(key)
            if isinstance(d, dict) and pd.notna(d.get("distance", np.nan)):
                candidates[key] = d.get("distance")

        if not candidates:
            return None

        chosen_key = min(candidates, key=candidates.get)
        layers_info = row.get(chosen_key, {}).get("layers", [])
        if not layers_info:
            return None

        soil_descriptions = []
        for entry in layers_info:
            if isinstance(entry, str) and 'Soil:' in entry:
                split_entry = entry.split('Soil:')[-1].split(',')[0].strip()
                if split_entry != 'N/A':
                    soil_descriptions.append(split_entry)

        if not soil_descriptions:
            return None

        porosities = [self.get_porosity_by_description(s) for s in soil_descriptions]
        permeabilities = [self.get_hydraulic_conductivity_by_description(s) for s in soil_descriptions]
        angles = [self.get_friction_angle_by_description(s) for s in soil_descriptions]
        particles = [self.get_particle_sizes_by_description(s) for s in soil_descriptions if self.get_particle_sizes_by_description(s)]

        porosities = [p for p in porosities if not self._is_missing(p)]
        permeabilities = [k for k in permeabilities if not self._is_missing(k)]
        angles = [a for a in angles if not self._is_missing(a)]

        # Keep “style” close: store list of range dicts (like before), just without *_range column names
        if self._is_missing(row.get("BRO_Porosity")) and porosities:
            self.df.at[idx, "BRO_Porosity"] = porosities
        if self._is_missing(row.get("BRO_Hydraulic_conductivity")) and permeabilities:
            self.df.at[idx, "BRO_Hydraulic_conductivity"] = permeabilities
        if self._is_missing(row.get("BRO_Friction_angle")) and angles:
            self.df.at[idx, "BRO_Friction_angle"] = angles

        # Particle sizes: store mean as point dicts
        if particles:
            def avg(key):
                vals = [p[key] for p in particles if isinstance(p, dict) and key in p]
                return float(np.mean(vals)) if vals else np.nan

            if self._is_missing(row.get("BRO_d10")):
                self.df.at[idx, "BRO_d10"] = self._make_point(avg("d10"))
            if self._is_missing(row.get("BRO_d50")):
                self.df.at[idx, "BRO_d50"] = self._make_point(avg("d50"))
            if self._is_missing(row.get("BRO_d60")):
                self.df.at[idx, "BRO_d60"] = self._make_point(avg("d60"))
            if self._is_missing(row.get("BRO_d70")):
                self.df.at[idx, "BRO_d70"] = self._make_point(avg("d70"))

        return None

    # ==================================================
    # GeoTOP-derived params -> GeoTOP_* (dicts)
    # ==================================================

    def get_info_from_GeoTOP(self, idx):
        """Derive GeoTOP-based blanket, aquifer, and soil-property columns."""
        # GeoTOP_* columns (NO _range)
        out_cols = [
            "GeoTOP_Blanket_thickness",
            "GeoTOP_Aquifer_thickness",
            "GeoTOP_Aquifer_Soils",
            "GeoTOP_Porosity",
            "GeoTOP_Hydraulic_conductivity",
            "GeoTOP_Friction_angle",
            "GeoTOP_d10",
            "GeoTOP_d50",
            "GeoTOP_d60",
            "GeoTOP_d70",
        ]
        for c in out_cols:
            if c not in self.df.columns:
                self.df[c] = pd.NA

        row = self.df.loc[idx]
        profile = row.get("GeoTOP_data")

        def invalid(p):
            if p is None:
                return True
            if isinstance(p, float) and np.isnan(p):
                return True
            if isinstance(p, (list, np.ndarray)) and len(p) == 0:
                return True
            return False

        def compute(profile_layers):
            max_blanket = 8.0
            prof = [dict(x) for x in profile_layers]
            for layer in prof:
                layer["thickness"] = abs(layer.get("thickness", 0))

            candidates = []
            i = 0
            while i < len(prof):
                aquifer_th = 0.0
                aquifer_idx = []
                j = i
                while j < len(prof):
                    lith = str(prof[j].get("lithology", "")).lower()
                    th = float(prof[j].get("thickness", 0))
                    if "zand" in lith:
                        aquifer_th += th
                        aquifer_idx.append(j)
                    elif "klei" in lith and th <= 0.5:
                        aquifer_th += th
                        aquifer_idx.append(j)
                    else:
                        break
                    j += 1

                if aquifer_idx:
                    blanket_th = sum(l["thickness"] for l in prof[:aquifer_idx[0]])
                    candidates.append((aquifer_th, aquifer_idx, blanket_th))

                i = j + 1

            if not candidates:
                return pd.NA, pd.NA, []

            best = max(candidates, key=lambda x: x[0])
            aqu_th, idxs, bl_th = best
            bl_th = min(bl_th, max_blanket)

            return self._make_range(bl_th, bl_th), self._make_range(aqu_th, aqu_th), [prof[k]["lithology"] for k in idxs]

        if invalid(profile):
            return None

        bt, at, soils = compute(profile)
        self.df.at[idx, "GeoTOP_Blanket_thickness"] = bt
        self.df.at[idx, "GeoTOP_Aquifer_thickness"] = at
        self.df.at[idx, "GeoTOP_Aquifer_Soils"] = soils

        if soils:
            por = [self.get_porosity_by_description(s) for s in soils]
            k = [self.get_hydraulic_conductivity_by_description(s) for s in soils]
            phi = [self.get_friction_angle_by_description(s) for s in soils]
            parts = [self.get_particle_sizes_by_description(s) for s in soils if self.get_particle_sizes_by_description(s)]

            por = [x for x in por if not self._is_missing(x)]
            k = [x for x in k if not self._is_missing(x)]
            phi = [x for x in phi if not self._is_missing(x)]

            # Keep close to your earlier style: list of range dicts
            self.df.at[idx, "GeoTOP_Porosity"] = por if por else pd.NA
            self.df.at[idx, "GeoTOP_Hydraulic_conductivity"] = k if k else pd.NA
            self.df.at[idx, "GeoTOP_Friction_angle"] = phi if phi else pd.NA

            def avg(key):
                vals = [p[key] for p in parts if isinstance(p, dict) and key in p]
                return float(np.mean(vals)) if vals else np.nan

            self.df.at[idx, "GeoTOP_d10"] = self._make_point(avg("d10"))
            self.df.at[idx, "GeoTOP_d50"] = self._make_point(avg("d50"))
            self.df.at[idx, "GeoTOP_d60"] = self._make_point(avg("d60"))
            self.df.at[idx, "GeoTOP_d70"] = self._make_point(avg("d70"))

        return None

    # ==================================================
    # Source quantifier -> Source_* (dicts, NO _range)
    # ==================================================

    def source_quantifier(self, idx):
        """Convert the source soil description into Source_* material property columns."""
        out_cols = [
            "Source_Porosity",
            "Source_Hydraulic_conductivity",
            "Source_Friction_angle",
            "Source_d10",
            "Source_d50",
            "Source_d60",
            "Source_d70",
        ]
        for c in out_cols:
            if c not in self.df.columns:
                self.df[c] = pd.NA

        row = self.df.loc[idx]
        desc = row.get("Soil type (source)")
        if pd.isna(desc):
            return None

        self.df.at[idx, "Source_Porosity"] = self.get_porosity_by_description(desc)
        self.df.at[idx, "Source_Hydraulic_conductivity"] = self.get_hydraulic_conductivity_by_description(desc)
        self.df.at[idx, "Source_Friction_angle"] = self.get_friction_angle_by_description(desc)

        ps = self.get_particle_sizes_by_description(desc)
        if ps:
            self.df.at[idx, "Source_d10"] = self._make_point(ps.get("d10", np.nan))
            self.df.at[idx, "Source_d50"] = self._make_point(ps.get("d50", np.nan))
            self.df.at[idx, "Source_d60"] = self._make_point(ps.get("d60", np.nan))
            self.df.at[idx, "Source_d70"] = self._make_point(ps.get("d70", np.nan))

        return None
    
    def apply_manual_seepage_lengths(self, overwrite=True, verbose=True):
        """
        Apply manually assigned seepage lengths by ID.
    
        Parameters
        ----------
        overwrite : bool
            If True, overwrite existing Seepage_length values.
            If False, only fill missing values.
        verbose : bool
            If True, print a short summary.
    
        Notes
        -----
        Values are stored in the same dict-style format used elsewhere:
            {"type": "point", "value": ...}
        """
    
        if "Seepage_length" not in self.df.columns:
            self.df["Seepage_length"] = pd.NA
    
        seepage_map = {
            "ILPD0007": 35,
            "STOW0011": 213,
            "STOW0012": 207,
            "STOW0051": 123,
            "STOW0052": 137,
            "STOW0053": 158,
            "STOW0055": 188,
            "STOW0056": 99,
            "STOW0057": 155,
            "STOW0058": 154,
            "STOW0059": 152,
            "STOW0060": 115,
            "STOW0103": 168,
            "STOW0120": 195,
            "STOW0124": 201,
            "STOW0126": 224,
            "STOW0130": 304,
            "STOW0132": 88,
            "STOW0204": 360,
            "STOW0205": 28,
            "STOW0206": 37,
            "STOW0210": 306,
            "STOW0223": 120,
            "STOW0255": 64,
            "STOW0256": 64,
            "STOW0257": 64,
            "STOW0260": 227,
            "STOW0263": 309,
            "STOW0269": 261,
            "STOW0270": 197,
            "STOW0271": 171,
            "STOW0272": 191,
            "STOW0273": 73,
            "STOW0274": 216,
            "STOW0275": 240,
            "STOW0276": 259,
            "STOW0277": 149,
            "STOW0280": 258,
            "STOW0281": 62,
            "STOW0282": 195,
            "STOW0284": 338,
            "STOW0285": 320,
            "STOW0287": 311,
            "STOW0310": 71,
            "STOW0311": 72,
            "STOW0312": 124,
            "STOW0313": 124,
            "STOW0314": 126,
            "STOW0316": 178,
            "STOW0317": 200,
            "STOW0318": 187,
            "STOW0352": 63,
            "STOW0383": 322,
            "STOW0454": 116,
            "STOW0477": 180,
            "STOW0479": 139,
            "STOW0485": 57,
            "STOW0492": 109,
            "STOW0495": 127,
            "STOW0496": 49,
            "STOW0498": 198,
            "STOW0499": 66,
            "STOW0516": 138,
            "STOW0517": 150,
            "STOW0535": 137,
            "STOW0538": 365,
            "STOW0539": 84,
            "STOW0566": 139,
            "STOW0572": 55,
            "STOW0573": 243,
            "STOW0574": 334,
            "STOW0575": 326,
            "STOW0576": 154,
            "STOW0580": 218,
            "STOW0584": 102,
            "STOW0592": 178,
            "STOW0595": 228,
            "STOW0596": 210,
            "STOW0629": 190,
            "STOW1113": 154,
            "STOW1114": 177,
            "STOW1115": 71,
            "STOW1119": 195,
            "STOW1121": 121,
            "STOW1122": 117,
            "STOW1123": 116,
            "STOW1125": 255,
            "STOW1128": 125,
        }
    
        ids = self.df["ID"].astype(str)
    
        n_set = 0
        for point_id, value in seepage_map.items():
            mask = ids == point_id
    
            if not mask.any():
                continue
    
            if overwrite:
                self.df.loc[mask, "Seepage_length"] = [self._make_point(value)] * int(mask.sum())
                n_set += int(mask.sum())
            else:
                sub_idx = self.df.index[mask]
                for idx in sub_idx:
                    if self._is_missing(self.df.at[idx, "Seepage_length"]):
                        self.df.at[idx, "Seepage_length"] = self._make_point(value)
                        n_set += 1
    
        if verbose:
            print(f"Applied manual seepage lengths to {n_set} rows.")
    
        return None
    
    # ==================================================
    # Imputation methods
    # ==================================================

    def run_bayesian_network(self, imputation_features=None, name='not_specified', plot=False, complicated=False):
        """Run Bayesian imputation and save the imputed dataframe plus inference output."""
        self._database._df = self.df

        if imputation_features is None:
            imputation_features = self._database._get_imputation_features().copy()

        copydf = self.df.copy()

        os.makedirs("src/bayesian_network_results", exist_ok=True)

        imputed_path = f"src/bayesian_network_results/Imputed_database_{name}_with_{len(copydf)}_rows.pkl"
        idata_path = f"src/bayesian_network_results/Bayesian_network_idata_{name}_with_{len(copydf)}_rows.nc"

        if os.path.exists(imputed_path):
            try:
                saved_df = pd.read_pickle(imputed_path)
                if copydf[imputation_features].shape == saved_df[imputation_features].shape:
                    print("\nBayesian network model already exists, applying it to the database.\n")

                    if os.path.exists(idata_path):
                        trace = az.from_netcdf(idata_path)

                        if plot:
                            try:
                                az.plot_trace(
                                    trace,
                                    var_names=[v for v in trace.posterior.data_vars if trace.posterior[v].ndim == 2],
                                    kind="rank_vlines"
                                )
                                plt.subplots_adjust(hspace=1, wspace=0.2)
                                plt.suptitle(f"Trace plot for {name}")

                                az.plot_energy(trace)
                                plt.suptitle(f"Energy plot for {name}")
                            except Exception:
                                print("Cant plot.")

                    self._apply_bayesian_imputation_to_df(name=name, imputation_features=imputation_features)
                    return None
            except Exception as e:
                print(f"Could not load previous Bayesian network results: {e}")

        print("\n\nRunning Bayesian network...\n\n")

        data = copydf[imputation_features].copy()
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        log_cols = ['Hydraulic_conductivity', 'd10', 'd50', 'd60', 'd70']
        for col in log_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data.loc[data[col] <= 0, col] = np.nan
                data[col] = np.log10(data[col])

        nan_mask = pd.isna(data).to_numpy()
        n_missing = nan_mask.sum()

        print(f"\n\nn_missing is: {n_missing}.\n\n")

        if n_missing == 0:
            print("No missing values found.")
            return None

        data_np = data.to_numpy(dtype=float)
        means = np.nanmean(data_np, axis=0)
        stds = np.nanstd(data_np, axis=0)

        means = np.where(np.isnan(means), 0.0, means)
        stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds)

        data_scaled_np = (data_np - means) / stds
        data_scaled = pd.DataFrame(data_scaled_np, columns=data.columns, index=data.index)

        n = len(imputation_features)

        if not complicated:
            with pm.Model() as model:
                mus = pm.Normal("mu", mu=0.5, sigma=0.5, shape=n)
                cov_chol, _, _ = pm.LKJCholeskyCov("cov", n=n, eta=1.0, sd_dist=pm.Exponential.dist(1))

                x_unobs = pm.Normal("x_unobs", 0, 1, shape=(n_missing,))

                x_scaled_np_safe = data_scaled.to_numpy().copy()
                x_scaled_np_safe[nan_mask] = 0.0

                x_scaled = pt.as_tensor_variable(x_scaled_np_safe)
                x_filled = pt.set_subtensor(x_scaled[nan_mask], x_unobs)
                x_filled = pm.Deterministic("x_filled", x_filled)

                pm.Potential("x_logp", pm.logp(rv=pm.MvNormal.dist(mu=mus, chol=cov_chol), value=x_filled))

                trace = pm.sample(
                    1000,
                    tune=1000,
                    chains=2,
                    progressbar=True,
                    return_inferencedata=True,
                    target_accept=0.95,
                    init="jitter+adapt_diag"
                )

            x_unobs_mean = trace.posterior["x_unobs"].mean(dim=("chain", "draw")).values
            data_scaled_imputed = data_scaled.to_numpy().copy()
            data_scaled_imputed[nan_mask] = x_unobs_mean
            data_imputed = data_scaled_imputed * stds + means

        else:
            X_raw = data.values.astype(float)
            N, D = X_raw.shape

            U = np.empty_like(X_raw, dtype=float)
            for j in range(D):
                col = X_raw[:, j]
                mask = ~np.isnan(col)

                if mask.sum() == 0:
                    U[:, j] = np.nan
                    continue

                ranks = st.rankdata(col[mask], method="average") / (mask.sum() + 1.0)
                z = st.norm.ppf(ranks)
                U[mask, j] = z
                U[~mask, j] = np.nan

            obs_rows, obs_cols = np.where(~np.isnan(U))
            obs_vals = U[obs_rows, obs_cols]

            with pm.Model() as copula:
                L, _, _ = pm.LKJCholeskyCov("chol", n=D, eta=2.0, sd_dist=pm.Exponential.dist(1.0))
                Z = pm.MvNormal("Z", mu=pt.zeros(D), chol=L, shape=(N, D))
                pm.Normal("obs", mu=Z[obs_rows, obs_cols], sigma=1e-2, observed=obs_vals)

                idata = pm.sample(
                    1000,
                    tune=1000,
                    chains=2,
                    target_accept=0.95,
                    progressbar=True,
                    return_inferencedata=True
                )

            Z_hat = idata.posterior["Z"].mean(dim=("chain", "draw")).values
            U_imputed = np.where(np.isnan(U), Z_hat, U)

            X_imputed = X_raw.copy()
            for j in range(D):
                col = X_raw[:, j]
                mask_miss = np.isnan(col)

                if mask_miss.any():
                    observed = col[~np.isnan(col)]
                    if observed.size == 0:
                        X_imputed[mask_miss, j] = np.nan
                        continue

                    p = st.norm.cdf(U_imputed[mask_miss, j])
                    observed_sorted = np.sort(observed)
                    q = np.quantile(observed_sorted, p, method="linear")
                    X_imputed[mask_miss, j] = q

            data_imputed = X_imputed

        for col_name in imputation_features:
            if col_name in log_cols:
                col_idx = imputation_features.index(col_name)
                data_imputed[:, col_idx] = 10 ** data_imputed[:, col_idx]

        copydf.loc[:, imputation_features] = pd.DataFrame(
            data_imputed,
            columns=imputation_features,
            index=copydf.index
        )

        copydf.to_pickle(imputed_path)

        if not complicated:
            trace.to_netcdf(idata_path)
            try:
                vars_to_plot = list(trace.posterior.data_vars)
                az.plot_posterior(trace, var_names=vars_to_plot)
                az.plot_energy(trace)
                plt.suptitle(f"Energy plot for {name}")
            except Exception:
                print("Cant plot.")
        else:
            idata.to_netcdf(idata_path)
            try:
                vars_to_plot = list(idata.posterior.data_vars)
                az.plot_posterior(idata, var_names=vars_to_plot)
                az.plot_energy(idata)
                plt.suptitle(f"Energy plot for {name}")
            except Exception:
                print("Cant plot.")

        self._apply_bayesian_imputation_to_df(name=name, imputation_features=imputation_features)

        print("\nDatabase Imputed.")
        return None

    def _apply_bayesian_imputation_to_df(self, name, imputation_features=None):
        """Apply saved Bayesian-imputed values only where the current dataframe is still missing."""
        self.df = self._database._df

        if imputation_features is None:
            imputation_features = self._database._get_imputation_features().copy()

        imputed_path = f"src/bayesian_network_results/Imputed_database_{name}_with_{len(self.df)}_rows.pkl"

        if os.path.exists(imputed_path):
            df_imputed = pd.read_pickle(imputed_path)

            for col in imputation_features:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')

                mask = self.df[col].isna()
                self.df.loc[mask, col] = df_imputed.loc[mask, col]

            self._database._df = self.df
        else:
            print("No Bayesian imputed file found, so create it first.")

        return self.df

    def run_mean_imputation(self, imputation_features=None, name='not_specified'):
        """Fill missing imputation features with column means and save the result."""
        self.df = self._database._df

        if imputation_features is None:
            imputation_features = self._database._get_imputation_features().copy()

        copydf = self.df.copy()

        os.makedirs("src/mean_imputation_results", exist_ok=True)

        imputed_path = f"src/mean_imputation_results/Imputed_database_{name}_with_{len(copydf)}_rows.pkl"

        if os.path.exists(imputed_path):
            try:
                saved_df = pd.read_pickle(imputed_path)
                if copydf[imputation_features].shape == saved_df[imputation_features].shape:
                    print("\nMean imputation file already exists, applying it to the database.\n")
                    self._apply_mean_imputation_to_df(name=name, imputation_features=imputation_features)
                    return None
            except Exception as e:
                print(f"Could not load previous mean imputation results: {e}")

        print("\n\nRunning mean imputation...\n\n")

        data = copydf[imputation_features].copy()

        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        n_missing = data.isna().sum().sum()
        print(f"\n\nn_missing is: {n_missing}.\n\n")

        if n_missing == 0:
            print("No missing values found.")
            return None

        for col in imputation_features:
            mean_value = data[col].dropna().mean()
            if pd.notna(mean_value):
                data[col] = data[col].fillna(mean_value)

        copydf.loc[:, imputation_features] = data
        copydf.to_pickle(imputed_path)

        self._apply_mean_imputation_to_df(name=name, imputation_features=imputation_features)

        print("\nDatabase imputed with column means.")
        return None

    def _apply_mean_imputation_to_df(self, name, imputation_features=None):
        """Apply saved mean-imputed values only where the current dataframe is still missing."""
        self.df = self._database._df

        if imputation_features is None:
            imputation_features = self._database._get_imputation_features().copy()

        imputed_path = f"src/mean_imputation_results/Imputed_database_{name}_with_{len(self.df)}_rows.pkl"

        if os.path.exists(imputed_path):
            df_imputed = pd.read_pickle(imputed_path)

            for col in imputation_features:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')

                mask = self.df[col].isna()
                self.df.loc[mask, col] = df_imputed.loc[mask, col]

            self._database._df = self.df
        else:
            print("No mean-imputed file found, so create it first.")

        return self.df
    
    
    # ==================================================
    # Backup helpers
    # ==================================================

    def _get_backup_path(self, enrichment_name: str, params: dict):
        """
        Build the pickle path used to cache raw enrichment results.

        The filename is based on the enrichment name plus the parameter values,
        so different settings create different cache files.
        """
        root = os.path.join("src", "enrichment_backups", enrichment_name)
        os.makedirs(root, exist_ok=True)

        parts = []
        for key, value in sorted(params.items(), key=lambda x: x[0]):
            if isinstance(value, float):
                value = f"{value}"
            value = str(value).replace(" ", "_").replace(".", "p").replace("/", "-").replace("\\", "-").replace(":", "-")
            parts.append(f"{key}={value}")

        filename = enrichment_name if not parts else enrichment_name + "__" + "__".join(parts)
        return os.path.join(root, f"{filename}.pkl")

    def _row_backup_key(self, row):
        """Create the row identifier used in enrichment backup files."""
        x = pd.to_numeric(row.get("X/lon"), errors="coerce")
        y = pd.to_numeric(row.get("Y/lat"), errors="coerce")
        year = pd.to_numeric(row.get("Year"), errors="coerce")
        day_month = row.get("Day+month")
        return {
            "X/lon": None if pd.isna(x) else float(x),
            "Y/lat": None if pd.isna(y) else float(y),
            "Year": None if pd.isna(year) else int(year),
            "Day+month": None if pd.isna(day_month) else str(day_month),
        }

    def _empty_backup_df(self):
        """Return an empty backup table with the expected columns."""
        return pd.DataFrame(columns=["X/lon", "Y/lat", "Year", "Day+month", "payload"])

    def _load_backup_df(self, enrichment_name: str, params: dict):
        """Load a backup table if it exists, otherwise return an empty one."""
        path = self._get_backup_path(enrichment_name, params)

        if os.path.exists(path):
            try:
                df = pd.read_pickle(path)
                if isinstance(df, pd.DataFrame):
                    expected = ["X/lon", "Y/lat", "Year", "Day+month", "payload"]
                    for col in expected:
                        if col not in df.columns:
                            df[col] = pd.NA
                    return df[expected].copy()
            except Exception:
                pass

        return self._empty_backup_df()

    def _save_backup_df(self, backup_df: pd.DataFrame, enrichment_name: str, params: dict):
        """Write the backup table for one enrichment method to disk."""
        path = self._get_backup_path(enrichment_name, params)
        backup_df.to_pickle(path)
        return path

    def _find_backup_payload(self, backup_df: pd.DataFrame, row):
        """Return cached payload for a row, '__MISSING__' marker, or None if absent."""
        if backup_df.empty:
            return None

        key = self._row_backup_key(row)
        mask = (
            (backup_df["X/lon"] == key["X/lon"]) &
            (backup_df["Y/lat"] == key["Y/lat"]) &
            (backup_df["Year"] == key["Year"]) &
            (backup_df["Day+month"] == key["Day+month"])
        )
        hits = backup_df.loc[mask]

        if hits.empty:
            return None

        payload = hits.iloc[-1]["payload"]
        return "__MISSING__" if self._is_missing(payload) else payload

    def _upsert_backup_payload(self, backup_df: pd.DataFrame, row, payload):
        """Replace or append one cached payload row in the backup dataframe."""
        key = self._row_backup_key(row)
        mask = (
            (backup_df["X/lon"] == key["X/lon"]) &
            (backup_df["Y/lat"] == key["Y/lat"]) &
            (backup_df["Year"] == key["Year"]) &
            (backup_df["Day+month"] == key["Day+month"])
        )

        backup_df = backup_df.loc[~mask].copy()
        new_row = pd.DataFrame([{
            "X/lon": key["X/lon"],
            "Y/lat": key["Y/lat"],
            "Year": key["Year"],
            "Day+month": key["Day+month"],
            "payload": payload,
        }])
        return pd.concat([backup_df, new_row], ignore_index=True)

    # ==================================================
    # Helpers for dict-style point/range values
    # ==================================================

    def _is_missing(self, v):
        """Return True for values that should be treated as missing in this project."""
        if v is None:
            return True
        try:
            if pd.isna(v):
                return True
        except Exception:
            pass
        if isinstance(v, (dict, list)) and len(v) == 0:
            return True
        return False

    def _make_point(self, v):
        """Convert one numeric value to the standard {"type": "point"} format."""
        if v is None:
            return pd.NA
        try:
            if pd.isna(v):
                return pd.NA
        except Exception:
            pass

        if isinstance(v, dict) and v.get("type") == "point":
            return v

        v_num = pd.to_numeric(v, errors="coerce")
        if pd.isna(v_num):
            return pd.NA

        return {"type": "point", "value": float(v_num)}

    def _make_range(self, min_v, max_v):
        """Convert two numeric bounds to the standard {"type": "range"} format."""
        a = pd.to_numeric(min_v, errors="coerce")
        b = pd.to_numeric(max_v, errors="coerce")
        if pd.isna(a) or pd.isna(b):
            return pd.NA
        return {"type": "range", "min": float(a), "max": float(b)}

    def _ensure_value_dict(self, v):
        """
        If v is already dict-style point/range -> keep.
        If v is {'min':..,'max':..} -> convert to {'type':'range',...}
        If v is numeric -> point
        Else -> pd.NA
        """
        if self._is_missing(v):
            return pd.NA

        if isinstance(v, dict) and v.get("type") in ["point", "range"]:
            return v

        if isinstance(v, dict) and ("min" in v) and ("max" in v):
            return {"type": "range", "min": v.get("min"), "max": v.get("max")}

        v_num = pd.to_numeric(v, errors="coerce")
        if pd.isna(v_num):
            return pd.NA

        return {"type": "point", "value": float(v_num)}
