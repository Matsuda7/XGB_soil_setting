#!/usr/bin/env python3
"""Predict N-values directly at every local 10 m analysis-grid point."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from pathlib import Path
from module.paths import project_path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from module.terrain import load_or_create_dem_cache, sample_terrain_features
from module.bedrock import FEATURE_COLUMN as BEDROCK_RBF_FEATURE
from module.bedrock import interpolate_bedrock_rbf


# Configuration
DEFAULT_CONFIG = project_path("config/prediction_phi.json")
GEOLOGY_DIR = project_path("data/raw/geology/seamlessV2/")
JSHIS_CSV_NAME = "Z-V4-JAPAN-AMP-VS400_M250.csv"
JSHIS_DIR = project_path("data/raw/jshis/")
MODEL_DATASET = project_path("data/training/phi/model_dataset.csv")
MODEL_PATH = project_path("results/phi/model/xgb_spt_model.json")
PREPROCESSOR_PATH = project_path("results/phi/model/xgb_preprocessor.joblib")
METRICS_PATH = project_path("results/phi/model/metrics.json")
BEDROCK_POINTS_PATH = project_path("data/training/phi/assumed_bedrock_points.csv")
N_MIN = 0.0
N_MAX = 50.0
PHI_MAX_DEG = 40.0
JSHIS_COLUMNS = ["CODE", "JCODE", "AVS", "ARV", "AVS_EB", "AVS_REF"]
LOGGER = logging.getLogger(__name__)


def load_prediction_config(path: Path) -> dict:
    """Load and validate grid/prediction settings from JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction configuration does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "grid_input",
        "dem_input",
        "dem_cache",
        "borehole_data",
        "grid_crs",
        "prediction_depth_m",
        "grid_spacing_m",
        "display_spacing_m",
        "n_plot_min",
        "n_plot_max",
        "input_chunk_size",
        "output_chunk_dir",
        "n_value_plot",
        "phi_plot",
        "bedrock_plot",
        "bedrock_text",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("Prediction configuration is missing keys: " + ", ".join(missing))
    config["grid_input"] = project_path(config["grid_input"])
    config["dem_input"] = project_path(config["dem_input"])
    config["dem_cache"] = project_path(config["dem_cache"])
    config["borehole_data"] = project_path(config["borehole_data"])
    config["output_chunk_dir"] = project_path(config["output_chunk_dir"])
    config["n_value_plot"] = project_path(config["n_value_plot"])
    config["phi_plot"] = project_path(config["phi_plot"])
    config["bedrock_plot"] = project_path(config["bedrock_plot"])
    config["bedrock_text"] = project_path(config["bedrock_text"])
    config["phi_text"] = project_path(
        config.get("phi_text", config["phi_plot"].with_suffix(".txt.gz"))
    )
    if config.get("property_mask") is not None:
        config["property_mask"] = project_path(config["property_mask"])
    config["prediction_depth_m"] = float(config["prediction_depth_m"])
    config["grid_spacing_m"] = float(config["grid_spacing_m"])
    config["display_spacing_m"] = float(config["display_spacing_m"])
    config["n_plot_min"] = float(config["n_plot_min"])
    config["n_plot_max"] = float(config["n_plot_max"])
    config["input_chunk_size"] = int(config["input_chunk_size"])
    config["property_mask_spacing_m"] = float(
        config.get("property_mask_spacing_m", config["grid_spacing_m"])
    )
    return config


def validate_inputs(config: dict) -> None:
    """Fail early when a required local input is unavailable."""
    grid_input = config["grid_input"]
    required = [
        grid_input,
        config["dem_input"],
        config["borehole_data"],
        MODEL_DATASET,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METRICS_PATH,
        BEDROCK_POINTS_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files do not exist: " + ", ".join(missing))
    if config["prediction_depth_m"] < 0:
        raise ValueError("prediction_depth_m must be zero or greater.")
    if config["grid_spacing_m"] <= 0:
        raise ValueError("grid_spacing_m must be greater than zero.")
    ratio = config["display_spacing_m"] / config["grid_spacing_m"]
    if config["display_spacing_m"] < config["grid_spacing_m"] or not np.isclose(ratio, round(ratio)):
        raise ValueError("display_spacing_m must be a multiple of grid_spacing_m.")
    if config["input_chunk_size"] <= 0:
        raise ValueError("input_chunk_size must be greater than zero.")
    if config["n_plot_min"] >= config["n_plot_max"]:
        raise ValueError("n_plot_min must be smaller than n_plot_max.")
    if config["property_mask_spacing_m"] <= 0:
        raise ValueError("property_mask_spacing_m must be greater than zero.")
    property_mask = config.get("property_mask")
    if property_mask is not None and not property_mask.is_file():
        raise FileNotFoundError(f"Property mask does not exist: {property_mask}")


def load_model_artifacts() -> tuple[xgb.XGBRegressor, object, dict]:
    """Load the final all-data model, preprocessing object, and feature manifest."""
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with METRICS_PATH.open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    if metrics.get("final_fit_rows") is None:
        raise ValueError("Saved model is not the all-data final model. Run 04_train_xgb.py.")
    return model, preprocessor, metrics


def load_geology() -> gpd.GeoDataFrame:
    """Load and combine all official GSJ polygon tiles."""
    paths = sorted(GEOLOGY_DIR.rglob("*_poly.shp"))
    if not paths:
        raise FileNotFoundError(f"No GSJ *_poly.shp file found under: {GEOLOGY_DIR}")
    parts = [gpd.read_file(path)[["symbol", "ser", "geometry"]] for path in paths]
    crs = parts[0].crs
    if crs is None or any(part.crs != crs for part in parts):
        raise ValueError("GSJ polygon CRS is missing or inconsistent.")
    geology = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=crs
    )
    LOGGER.info("Loaded GSJ polygons: %d", len(geology))
    return geology


def find_jshis_csv() -> Path:
    """Find the extracted national J-SHIS V4 CSV."""
    paths = sorted(JSHIS_DIR.rglob(JSHIS_CSV_NAME))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {JSHIS_CSV_NAME} under {JSHIS_DIR}; found {len(paths)}"
        )
    return paths[0]


def load_regional_jshis(prefixes: set[str]) -> pd.DataFrame:
    """Load only first-level meshes covering the borehole study region."""
    matches: list[pd.DataFrame] = []
    path = find_jshis_csv()
    chunks = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=JSHIS_COLUMNS,
        dtype={"CODE": "string"},
        na_values=["-"],
        skipinitialspace=True,
        chunksize=500_000,
        encoding="ascii",
    )
    for chunk in chunks:
        chunk["CODE"] = chunk["CODE"].str.strip()
        selected = chunk.loc[chunk["CODE"].str[:4].isin(prefixes)]
        if not selected.empty:
            matches.append(selected.copy())
    if not matches:
        raise ValueError(f"No J-SHIS rows found for first-level mesh prefixes: {prefixes}")
    result = pd.concat(matches, ignore_index=True)
    for column in JSHIS_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result["CODE"].duplicated().any():
        raise ValueError("J-SHIS regional data contains duplicate mesh codes.")
    LOGGER.info("Loaded regional J-SHIS meshes: %d", len(result))
    return result.rename(
        columns={
            "CODE": "jshis_meshcode",
            "JCODE": "jshis_jcode",
            "AVS": "jshis_avs30",
            "ARV": "jshis_arv",
            "AVS_EB": "jshis_avs_eb",
            "AVS_REF": "jshis_avs_ref",
        }
    )


def meshcodes_250m(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Vectorize conversion of WGS84 positions to 10-digit 250 m mesh codes."""
    if longitude.size == 0:
        return np.empty(0, dtype="<U10")
    lat_scaled = latitude * 1.5
    p = np.floor(lat_scaled).astype(int)
    lat_fraction = lat_scaled - p
    lon_scaled = longitude - 100.0
    u = np.floor(lon_scaled).astype(int)
    lon_fraction = lon_scaled - u
    q = np.minimum(np.floor(lat_fraction * 8).astype(int), 7)
    r = np.minimum(np.floor(lon_fraction * 8).astype(int), 7)
    lat_fraction = lat_fraction * 8 - q
    lon_fraction = lon_fraction * 8 - r
    s = np.minimum(np.floor(lat_fraction * 10).astype(int), 9)
    t = np.minimum(np.floor(lon_fraction * 10).astype(int), 9)
    lat_fraction = lat_fraction * 10 - s
    lon_fraction = lon_fraction * 10 - t
    fourth = 1 + (lon_fraction >= 0.5).astype(int) + 2 * (lat_fraction >= 0.5).astype(int)
    lon_fraction = np.mod(lon_fraction * 2, 1)
    lat_fraction = np.mod(lat_fraction * 2, 1)
    fifth = 1 + (lon_fraction >= 0.5).astype(int) + 2 * (lat_fraction >= 0.5).astype(int)
    code = np.char.zfill(p.astype(str), 2)
    for part in [np.char.zfill(u.astype(str), 2), q, r, s, t, fourth, fifth]:
        code = np.char.add(code, np.asarray(part).astype(str))
    return code


def attach_spatial_attributes(
    chunk: pd.DataFrame,
    geology: gpd.GeoDataFrame,
    jshis: pd.DataFrame,
    grid_crs: str,
) -> pd.DataFrame:
    """Attach GSJ polygons and J-SHIS mesh attributes to one grid chunk."""
    chunk = chunk.reset_index(drop=True).copy()
    chunk["_row"] = np.arange(len(chunk))
    points = gpd.GeoDataFrame(
        chunk,
        geometry=gpd.points_from_xy(chunk["x"], chunk["y"]),
        crs=grid_crs,
    ).to_crs(geology.crs)
    joined = gpd.sjoin(points, geology, how="left", predicate="within")
    duplicate_count = int(joined["_row"].duplicated(keep=False).sum())
    if duplicate_count:
        LOGGER.warning("Multiple GSJ polygons in chunk; selecting first: %d rows", duplicate_count)
        joined = joined.sort_values(["_row", "index_right"], kind="stable")
    joined = joined.drop_duplicates("_row", keep="first").sort_values("_row")
    result = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))

    wgs84 = points.to_crs("EPSG:4326").sort_values("_row")
    result["jshis_meshcode"] = meshcodes_250m(
        wgs84.geometry.x.to_numpy(), wgs84.geometry.y.to_numpy()
    )
    result = result.merge(jshis, on="jshis_meshcode", how="left", validate="many_to_one")
    result["geology_missing"] = result[["symbol", "ser"]].isna().all(axis=1)
    invalid_jshis = (
        result["jshis_jcode"].isna()
        | result["jshis_jcode"].eq(0)
        | result["jshis_avs30"].fillna(0).le(0)
        | result["jshis_arv"].fillna(0).le(0)
    )
    result["jshis_missing"] = invalid_jshis
    result.loc[
        invalid_jshis,
        ["jshis_jcode", "jshis_avs30", "jshis_arv", "jshis_avs_eb", "jshis_avs_ref"],
    ] = pd.NA
    return result


def prepare_features(
    rows: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]
) -> pd.DataFrame:
    """Match stage-05 preprocessing input conventions."""
    features = rows[[*numeric, *categorical]].copy()
    for column in categorical:
        features[column] = features[column].astype("string").fillna("UNKNOWN")
    return features


def predict_grid(
    config: dict,
    max_chunks: int | None,
    resume: bool,
) -> int:
    """Stream every grid row through spatial enrichment and XGBoost."""
    grid_input = config["grid_input"]
    output_dir = config["output_chunk_dir"]
    prediction_depth = config["prediction_depth_m"]
    model, preprocessor, metrics = load_model_artifacts()
    bedrock_sources = pd.read_csv(BEDROCK_POINTS_PATH, encoding="utf-8-sig")
    geology = load_geology()
    model_data = pd.read_csv(MODEL_DATASET, usecols=["jshis_meshcode"])
    prefixes = set(model_data["jshis_meshcode"].astype(str).str[:4])
    jshis = load_regional_jshis(prefixes)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = config["property_mask_bounds"]
    dem_shape = (
        int(round((bounds["ymax"] - bounds["ymin"]) / config["grid_spacing_m"])) + 1,
        int(round((bounds["xmax"] - bounds["xmin"]) / config["grid_spacing_m"])) + 1,
    )
    dem = load_or_create_dem_cache(config["dem_input"], config["dem_cache"], dem_shape)
    total = 0
    if not resume and any(output_dir.glob("N_value_10m_*.csv.gz")):
        raise FileExistsError(
            f"Prediction chunks already exist under {output_dir}. "
            "Use --resume or choose an empty --output-dir."
        )
    existing_chunks = set(output_dir.glob("N_value_10m_*.csv.gz")) if resume else set()
    if existing_chunks:
        LOGGER.info("Resuming with %d completed output chunks", len(existing_chunks))
    chunks = pd.read_csv(
        grid_input,
        sep=r"\s+",
        header=None,
        names=["x", "y", "legacy_surface_z", "legacy_bedrock_z"],
        chunksize=config["input_chunk_size"],
    )
    for chunk_number, chunk in enumerate(chunks, start=1):
        if max_chunks is not None and chunk_number > max_chunks:
            break
        chunk_path = output_dir / f"N_value_10m_{chunk_number:06d}.csv.gz"
        if chunk_path in existing_chunks:
            total += len(chunk)
            continue
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        if chunk.isna().any(axis=None):
            raise ValueError(f"Non-numeric grid values found in chunk {chunk_number}.")
        chunk = chunk.drop(columns=["legacy_surface_z", "legacy_bedrock_z"])
        terrain = sample_terrain_features(
            chunk,
            dem,
            xmin=float(bounds["xmin"]),
            ymin=float(bounds["ymin"]),
            spacing=config["grid_spacing_m"],
        )
        chunk[["surface_z", "slope", "curvature", "dem_missing"]] = terrain
        missing_terrain = chunk[["surface_z", "slope", "curvature"]].isna().any(axis=1)
        if missing_terrain.any():
            # Keep every analysis-grid point. The saved preprocessing pipeline
            # imputes missing numeric predictors with training-data medians.
            LOGGER.info(
                "Chunk %d has %s points with incomplete DEM features; "
                "retaining them for median imputation",
                chunk_number,
                f"{int(missing_terrain.sum()):,}",
            )
        chunk["n_value_elevation"] = chunk["surface_z"] - prediction_depth
        enriched = attach_spatial_attributes(chunk, geology, jshis, config["grid_crs"])
        rbf = metrics.get("bedrock_rbf", {})
        enriched[BEDROCK_RBF_FEATURE] = interpolate_bedrock_rbf(
            bedrock_sources,
            enriched,
            neighbors=int(rbf.get("neighbors", 50)),
            smoothing=float(rbf.get("smoothing", 0.0)),
            kernel=str(rbf.get("kernel", "thin_plate_spline")),
        )
        features = prepare_features(
            enriched, metrics["numeric_features"], metrics["categorical_features"]
        )
        transformed = preprocessor.transform(features)
        enriched["predicted_n_value"] = np.clip(model.predict(transformed), N_MIN, N_MAX)
        # Store only coordinates and predictions. Repeating every predictor for
        # tens of millions of 10 m cells would exhaust the available disk space.
        output_columns = ["x", "y", "predicted_n_value"]
        output_columns.append(BEDROCK_RBF_FEATURE)
        temporary_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
        enriched[output_columns].to_csv(
            temporary_path,
            mode="w",
            header=True,
            index=False,
            compression="gzip",
        )
        os.replace(temporary_path, chunk_path)
        total += len(enriched)
        LOGGER.info("Predicted chunk %d; total rows: %s", chunk_number, f"{total:,}")
    return total


def load_display_points(output_dir: Path, display_spacing: float) -> pd.DataFrame:
    """Read a regular subset of direct predictions for memory-safe plotting."""
    parts: list[pd.DataFrame] = []
    paths = sorted(output_dir.glob("N_value_10m_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No prediction chunks found under: {output_dir}")
    for path in paths:
        chunk = pd.read_csv(path)
        mask = (
            np.isclose(np.mod(chunk["x"], display_spacing), 0)
            & np.isclose(np.mod(chunk["y"], display_spacing), 0)
        )
        columns = ["x", "y", "predicted_n_value"]
        if BEDROCK_RBF_FEATURE in chunk:
            columns.append(BEDROCK_RBF_FEATURE)
        parts.append(chunk.loc[mask, columns])
    return pd.concat(parts, ignore_index=True)


def save_phi_matrix(output_dir: Path, output_path: Path, config: dict) -> int:
    """Write phi as a headerless 2D matrix aligned with the configured DEM."""
    paths = sorted(output_dir.glob("N_value_10m_*.csv.gz"))
    if not paths:
        # Support the single-file output produced by earlier versions.
        combined_path = output_dir.parent / "N_value_10m.csv.gz"
        if combined_path.is_file():
            paths = [combined_path]
        else:
            raise FileNotFoundError(f"No prediction output found under: {output_dir}")
    bounds = config["property_mask_bounds"]
    spacing = config["grid_spacing_m"]
    xmin = float(bounds["xmin"])
    ymin = float(bounds["ymin"])
    nx = int(round((float(bounds["xmax"]) - xmin) / spacing)) + 1
    ny = int(round((float(bounds["ymax"]) - ymin) / spacing)) + 1
    grid = np.zeros((ny, nx), dtype=np.float32)
    populated = np.zeros((ny, nx), dtype=bool)
    outside = 0
    duplicate = 0
    for path in paths:
        # The legacy combined file can contain tens of millions of rows.
        batches = pd.read_csv(
            path,
            usecols=["x", "y", "predicted_n_value"],
            chunksize=500_000,
        )
        for values in batches:
            phi = np.minimum(
                np.sqrt(15.0 * values["predicted_n_value"].clip(lower=N_MIN)) + 15.0,
                PHI_MAX_DEG,
            ).to_numpy(dtype=np.float32)
            x = values["x"].to_numpy()
            y = values["y"].to_numpy()
            xi = np.rint((x - xmin) / spacing).astype(int)
            yi = np.rint((y - ymin) / spacing).astype(int)
            aligned = np.isclose(x, xmin + xi * spacing) & np.isclose(y, ymin + yi * spacing)
            inside = aligned & (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
            duplicate += int(populated[yi[inside], xi[inside]].sum())
            grid[yi[inside], xi[inside]] = phi[inside]
            populated[yi[inside], xi[inside]] = True
            outside += int((~inside).sum())
    if duplicate:
        raise ValueError(f"Duplicate phi grid coordinates found: {duplicate}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.suffix == ".gz":
        stream = gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=1)
    else:
        stream = temporary_path.open("w", encoding="utf-8")
    with stream:
        np.savetxt(stream, grid, fmt="%.6f")
    os.replace(temporary_path, output_path)
    count = int(populated.sum())
    LOGGER.info(
        "Saved phi matrix %d x %d (%s populated, %s zero-filled, %s outside): %s",
        ny, nx, f"{count:,}", f"{grid.size - count:,}", f"{outside:,}", output_path,
    )
    return count


def save_bedrock_text(output_dir: Path, output_path: Path) -> int:
    """Write the RBF assumed-bedrock distribution for every predicted point."""
    paths = sorted(output_dir.glob("N_value_10m_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No prediction chunks found under: {output_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    compression = {"method": "gzip", "compresslevel": 1} if output_path.suffix == ".gz" else None
    total = 0
    write_index = 0
    for path in paths:
        for values in pd.read_csv(path, usecols=["x", "y", BEDROCK_RBF_FEATURE], chunksize=500_000):
            values.to_csv(
                temporary_path,
                sep=" ",
                mode="w" if write_index == 0 else "a",
                header=write_index == 0,
                index=False,
                compression=compression,
            )
            total += len(values)
            write_index += 1
    os.replace(temporary_path, output_path)
    LOGGER.info("Saved assumed-bedrock text values for %s points: %s", f"{total:,}", output_path)
    return total


def make_display_grid(
    points: pd.DataFrame, spacing: float
) -> tuple[np.ndarray, list[float]]:
    """Place directly predicted, regularly sampled points into a display raster."""
    x_min, x_max = points["x"].min(), points["x"].max()
    y_min, y_max = points["y"].min(), points["y"].max()
    nx = int(round((x_max - x_min) / spacing)) + 1
    ny = int(round((y_max - y_min) / spacing)) + 1
    grid = np.full((ny, nx), np.nan)
    xi = np.rint((points["x"] - x_min) / spacing).astype(int)
    yi = np.rint((points["y"] - y_min) / spacing).astype(int)
    grid[yi, xi] = points["predicted_n_value"]
    half_cell = spacing / 2.0
    return grid, [
        x_min - half_cell,
        x_max + half_cell,
        y_min - half_cell,
        y_max + half_cell,
    ]


def apply_property_mask_to_points(points: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only display points whose prop cell is valid, matching stage-03 masking."""
    prop_path = config.get("property_mask")
    if prop_path is None:
        return points

    bounds = config.get("property_mask_bounds")
    if not isinstance(bounds, dict):
        raise ValueError("property_mask_bounds must be set when property_mask is used.")
    required = {"xmin", "xmax", "ymin", "ymax"}
    missing = sorted(required - set(bounds))
    if missing:
        raise ValueError(
            "property_mask_bounds is missing keys: " + ", ".join(missing)
        )

    spacing = config["property_mask_spacing_m"]
    prop = np.loadtxt(prop_path)
    xmin = float(bounds["xmin"])
    xmax = float(bounds["xmax"])
    ymin = float(bounds["ymin"])
    ymax = float(bounds["ymax"])
    expected_shape = (
        int(round((ymax - ymin) / spacing)) + 1,
        int(round((xmax - xmin) / spacing)) + 1,
    )
    if prop.shape != expected_shape:
        raise ValueError(
            f"property_mask shape {prop.shape} does not match configured bounds "
            f"{expected_shape}."
        )

    xi = np.rint((points["x"].to_numpy() - xmin) / spacing).astype(int)
    yi = np.rint((points["y"].to_numpy() - ymin) / spacing).astype(int)
    inside = (
        (0 <= xi)
        & (xi < prop.shape[1])
        & (0 <= yi)
        & (yi < prop.shape[0])
    )
    valid = np.zeros(len(points), dtype=bool)
    valid[inside] = np.isin(prop[yi[inside], xi[inside]], np.arange(1, 10))
    masked = int((~valid).sum())
    LOGGER.info("Property mask removed %s display points", f"{masked:,}")
    return points.loc[valid].copy()


def load_shallowest_borehole_n(path: Path) -> pd.DataFrame:
    """Return the valid N-value at the minimum measured depth of each borehole."""
    source = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"坑口座標X", "坑口座標Y"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(
            "Borehole data is missing coordinate columns: " + ", ".join(missing)
        )

    measurement_numbers = [
        number
        for number in range(1, 81)
        if f"深度{number}" in source.columns and f"N値{number}" in source.columns
    ]
    if not measurement_numbers:
        raise ValueError("No matching 深度/N値 measurement columns were found.")

    depth_columns = [f"深度{number}" for number in measurement_numbers]
    n_columns = [f"N値{number}" for number in measurement_numbers]
    depths = source[depth_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    n_values = source[n_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid = np.isfinite(depths) & np.isfinite(n_values)
    selected_index = np.argmin(np.where(valid, depths, np.inf), axis=1)
    row_index = np.arange(len(source))
    has_value = valid.any(axis=1)

    boreholes = pd.DataFrame(
        {
            "x": pd.to_numeric(source["坑口座標X"], errors="coerce"),
            "y": pd.to_numeric(source["坑口座標Y"], errors="coerce"),
            "depth": depths[row_index, selected_index],
            # Match soil_parameter_setting: retain the observed upper value for
            # plot normalization, while replacing only negative N-values by zero.
            "n_value": np.maximum(n_values[row_index, selected_index], N_MIN),
            "has_value": has_value,
        }
    )
    boreholes = boreholes.loc[
        boreholes["has_value"]
        & boreholes[["x", "y", "depth", "n_value"]].notna().all(axis=1)
    ]
    return (
        boreholes.sort_values(["depth", "n_value"], ascending=[True, False])
        .drop_duplicates(["x", "y"])
        .reset_index(drop=True)
    )


def data_color_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Return finite data limits, expanding a constant range if necessary."""
    finite_parts = [array[np.isfinite(array)] for array in arrays]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        raise ValueError("No finite values are available to determine color limits.")
    combined = np.concatenate(finite_parts)
    lower = float(combined.min())
    upper = float(combined.max())
    if np.isclose(lower, upper):
        margin = max(abs(lower) * 0.01, 0.5)
        lower -= margin
        upper += margin
    return lower, upper


def save_plots(display_points: pd.DataFrame, config: dict) -> None:
    """Save direct-prediction maps using soil_parameter_setting plot conventions."""
    display_points = apply_property_mask_to_points(display_points, config)
    if display_points.empty:
        raise ValueError("No display points remain after applying the property mask.")
    grid, extent = make_display_grid(display_points, config["display_spacing_m"])
    phi = np.minimum(np.sqrt(15.0 * grid) + 15.0, PHI_MAX_DEG)
    bedrock_grid = None
    if BEDROCK_RBF_FEATURE in display_points:
        bedrock_points = display_points[["x", "y", BEDROCK_RBF_FEATURE]].rename(
            columns={BEDROCK_RBF_FEATURE: "predicted_n_value"}
        )
        bedrock_grid, _ = make_display_grid(bedrock_points, config["display_spacing_m"])
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="white")

    boreholes = load_shallowest_borehole_n(config["borehole_data"])
    boreholes = boreholes.loc[
        boreholes["x"].between(extent[0], extent[1])
        & boreholes["y"].between(extent[2], extent[3])
    ].copy()
    n_limits = (config["n_plot_min"], config["n_plot_max"])
    # observed_n = boreholes["n_value"].to_numpy(dtype=float)
    # n_limits = data_color_limits(observed_n)
    phi_limits = data_color_limits(phi)
    bedrock_limits = data_color_limits(bedrock_grid) if bedrock_grid is not None else None
    # observed_phi = np.minimum(np.sqrt(15.0 * observed_n) + 15.0, PHI_MAX_DEG,)
    # phi_limits = data_color_limits(observed_phi)

    plot_specs = [
        (
            grid,
            config["n_value_plot"],
            f"Direct XGBoost N-value prediction at depth {config['prediction_depth_m']:g} m",
            "N-value",
            n_limits,
            True,
        ),
        (
            phi,
            config["phi_plot"],
            f"Internal friction angle at depth {config['prediction_depth_m']:g} m",
            "Internal friction angle phi (deg)",
            phi_limits,
            False,
        ),
    ]
    if bedrock_grid is not None and bedrock_limits is not None:
        plot_specs.append(
            (
                bedrock_grid,
                config["bedrock_plot"],
                "RBF-interpolated assumed bedrock elevation",
                "Assumed bedrock elevation (m)",
                bedrock_limits,
                False,
            )
        )
    for values, path, title, label, limits, show_boreholes in plot_specs:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(
            np.ma.masked_invalid(values), origin="lower", extent=extent, cmap=cmap,
            aspect="equal", vmin=limits[0], vmax=limits[1],
        )
        if show_boreholes and not boreholes.empty:
            ax.scatter(
                boreholes["x"],
                boreholes["y"],
                c=boreholes["n_value"],
                cmap=cmap,
                norm=image.norm,
                s=35,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
                label="Shallowest borehole N-value",
                zorder=5,
            )
            ax.legend(loc="best")
        fig.colorbar(image, ax=ax).set_label(label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    LOGGER.info(
        "N-value plot range: %.4f to %.4f; shallowest borehole points: %d",
        n_limits[0],
        n_limits[1],
        len(boreholes),
    )
    LOGGER.info("Phi plot range: %.4f to %.4f", phi_limits[0], phi_limits[1])


def main(argv=None) -> None:
    """Run configured chunked direct prediction and optional visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--grid-input", type=Path, help="Override config grid_input.")
    parser.add_argument("--output-dir", type=Path, help="Override config output_chunk_dir.")
    parser.add_argument("--max-chunks", type=int, help="Testing only: stop after N chunks.")
    parser.add_argument("--resume", action="store_true", help="Resume a complete-chunk partial output.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate plots from existing chunks.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_prediction_config(args.config)
    if args.grid_input is not None:
        config["grid_input"] = args.grid_input
    if args.output_dir is not None:
        config["output_chunk_dir"] = args.output_dir
    validate_inputs(config)
    LOGGER.info("Prediction configuration: %s", args.config)
    LOGGER.info("Grid CRS: %s", config["grid_crs"])
    LOGGER.info("Grid spacing: %g m", config["grid_spacing_m"])
    total = None if args.plot_only else predict_grid(config, args.max_chunks, args.resume)
    phi_total = save_phi_matrix(config["output_chunk_dir"], config["phi_text"], config)
    bedrock_total = save_bedrock_text(config["output_chunk_dir"], config["bedrock_text"])
    if not args.no_plot:
        display_points = load_display_points(
            config["output_chunk_dir"], config["display_spacing_m"]
        )
        save_plots(display_points, config)
    print("\n10 m direct-prediction summary")
    print(f"Prediction depth: {config['prediction_depth_m']:g} m")
    if total is not None:
        print(f"Directly predicted 10 m points: {total:,}")
    print(f"Prediction chunk directory: {config['output_chunk_dir']}")
    print(f"Phi matrix ({phi_total:,} populated cells): {config['phi_text']}")
    print(f"Assumed-bedrock text ({bedrock_total:,} points): {config['bedrock_text']}")
    if not args.no_plot:
        print(f"Display sampling interval: {config['display_spacing_m']:g} m (no interpolation)")
        print(f"N-value plot: {config['n_value_plot']}")
        print(f"Phi plot: {config['phi_plot']}")



