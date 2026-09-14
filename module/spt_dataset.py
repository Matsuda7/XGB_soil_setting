#!/usr/bin/env python3
"""Build a measurement-level SPT dataset from the enriched borehole data."""

from __future__ import annotations

import logging
from pathlib import Path
from module.paths import project_path
from typing import Sequence

import geopandas as gpd
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BORING_CSV = project_path("data/raw/boring/BorToCsv.csv")
SPATIAL_INPUT = project_path("data/interim/boring_geology_jshis.gpkg")
SPATIAL_LAYER = "boring_geology_jshis"
OUTPUT_CSV = project_path("data/training/phi/model_dataset.csv")
BEDROCK_POINTS_OUTPUT = project_path("data/training/phi/assumed_bedrock_points.csv")
MAX_MEASUREMENTS = 80
N_VALUE_CAP = 50.0

ID_COLUMN = "ファイル名"
DEPTH_PREFIX = "深度"
N_VALUE_PREFIX = "N値"
LOGGER = logging.getLogger(__name__)


def expected_measurement_columns(max_measurements: int) -> list[str]:
    """Return all expected alternating depth and N-value source columns."""
    return [
        column
        for number in range(1, max_measurements + 1)
        for column in (f"{DEPTH_PREFIX}{number}", f"{N_VALUE_PREFIX}{number}")
    ]


def load_boring_measurements(
    csv_path: Path, max_measurements: int
) -> pd.DataFrame:
    """Load the wide KuniJiban CSV and validate measurement columns."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Borehole CSV does not exist: {csv_path}")
    source = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    required = [ID_COLUMN, *expected_measurement_columns(max_measurements)]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError("Borehole CSV is missing required columns: " + ", ".join(missing))
    if source[ID_COLUMN].isna().any():
        raise ValueError(f"Missing {ID_COLUMN} values found: {int(source[ID_COLUMN].isna().sum())}")
    if source[ID_COLUMN].duplicated().any():
        duplicates = source.loc[source[ID_COLUMN].duplicated(keep=False), ID_COLUMN]
        raise ValueError("Duplicate boring IDs found: " + ", ".join(map(str, duplicates.head(20))))
    return source[required].copy()


def reshape_spt_measurements(
    source: pd.DataFrame, max_measurements: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert depth1/N1...depth80/N80 from wide to measurement-level rows."""
    frames: list[pd.DataFrame] = []
    for number in range(1, max_measurements + 1):
        frame = source[
            [ID_COLUMN, f"{DEPTH_PREFIX}{number}", f"{N_VALUE_PREFIX}{number}"]
        ].copy()
        frame.columns = ["boring_id", "depth", "n_value"]
        frame["measurement_no"] = number
        frames.append(frame)

    long = pd.concat(frames, ignore_index=True)
    long["depth"] = pd.to_numeric(long["depth"], errors="coerce")
    long["n_value"] = pd.to_numeric(long["n_value"], errors="coerce")

    depth_only = long["depth"].notna() & long["n_value"].isna()
    n_value_only = long["depth"].isna() & long["n_value"].notna()
    both_missing = long["depth"].isna() & long["n_value"].isna()
    invalid_depth = long["depth"].notna() & long["depth"].le(0)
    invalid_n_value = long["n_value"].notna() & long["n_value"].lt(0)
    valid = ~(depth_only | n_value_only | both_missing | invalid_depth | invalid_n_value)

    statistics = {
        "wide_slots": len(long),
        "both_missing": int(both_missing.sum()),
        "depth_without_n_value": int(depth_only.sum()),
        "n_value_without_depth": int(n_value_only.sum()),
        "nonpositive_depth": int(invalid_depth.sum()),
        "negative_n_value": int(invalid_n_value.sum()),
        "valid_measurements": int(valid.sum()),
        "n_values_capped_at_50": int((long.loc[valid, "n_value"] > N_VALUE_CAP).sum()),
    }
    result = long.loc[valid].copy()
    # SPT refusal values above 50 are treated as the agreed upper-bound target.
    result["n_value"] = result["n_value"].clip(upper=N_VALUE_CAP)
    result["measurement_no"] = result["measurement_no"].astype("int16")
    return result, statistics


def load_spatial_attributes(path: Path, layer: str) -> pd.DataFrame:
    """Load hole-level geology and J-SHIS attributes created by stage 02."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Spatial input does not exist: {path}. "
            "Run 02_add_spatial_attributes.py first."
        )
    spatial = gpd.read_file(path, layer=layer)
    required = {
        "boring_id",
        "x",
        "y",
        "surface_z",
        "borehole_surface_z",
        "slope",
        "curvature",
        "dem_missing",
        "symbol",
        "ser",
        "jshis_meshcode",
        "jshis_jcode",
        "jshis_avs30",
        "jshis_arv",
        "jshis_avs_eb",
        "jshis_avs_ref",
        "jshis_matched",
    }
    missing = sorted(required - set(spatial.columns))
    if missing:
        raise ValueError("Spatial input is missing required columns: " + ", ".join(missing))
    if spatial["boring_id"].duplicated().any():
        raise ValueError("Spatial input contains duplicate boring_id values.")
    return pd.DataFrame(spatial.drop(columns="geometry"))


def merge_measurements_and_attributes(
    measurements: pd.DataFrame, spatial: pd.DataFrame
) -> pd.DataFrame:
    """Join measurement rows to hole-level attributes and derive elevation."""
    result = measurements.merge(
        spatial, on="boring_id", how="left", validate="many_to_one", indicator=True
    )
    unmatched = result["_merge"].ne("both")
    if unmatched.any():
        ids = result.loc[unmatched, "boring_id"].drop_duplicates()
        LOGGER.warning(
            "Excluding %d measurement boreholes absent from the spatial input: %s",
            len(ids),
            ", ".join(map(str, ids.head(20))),
        )
        result = result.loc[~unmatched].copy()
    result = result.drop(columns="_merge")
    result["n_value_elevation"] = result["surface_z"] - result["depth"]
    result["lowest_spt_elevation"] = result.groupby("boring_id")["n_value_elevation"].transform(
        "min"
    )
    result["distance_to_lowest_spt"] = (
        result["n_value_elevation"] - result["lowest_spt_elevation"]
    )
    n50 = (
        result.loc[result["n_value"].ge(N_VALUE_CAP)]
        .sort_values(["boring_id", "depth", "measurement_no"], kind="stable")
        .drop_duplicates("boring_id")
        .set_index("boring_id")["n_value_elevation"]
    )
    result["assumed_bedrock_elevation"] = result["boring_id"].map(n50)
    result["assumed_bedrock_from_n50"] = result["assumed_bedrock_elevation"].notna()
    result["assumed_bedrock_elevation"] = result["assumed_bedrock_elevation"].fillna(
        result["lowest_spt_elevation"]
    )
    result["geology_missing"] = result[["symbol", "ser"]].isna().all(axis=1)
    result["jshis_missing"] = ~result["jshis_matched"].astype("boolean").fillna(False).astype(bool)
    # Stage 03 preserves J-SHIS sentinel zeros for auditing. They must not be
    # interpreted as physical values in the model-ready dataset.
    jshis_value_columns = [
        "jshis_jcode",
        "jshis_avs30",
        "jshis_arv",
        "jshis_avs_eb",
        "jshis_avs_ref",
    ]
    result.loc[result["jshis_missing"], jshis_value_columns] = pd.NA

    preferred_order = [
        "boring_id",
        "measurement_no",
        "x",
        "y",
        "surface_z",
        "depth",
        "n_value_elevation",
        "lowest_spt_elevation",
        "distance_to_lowest_spt",
        "assumed_bedrock_elevation",
        "assumed_bedrock_from_n50",
        "n_value",
        "symbol",
        "ser",
        "geology_missing",
        "jshis_meshcode",
        "jshis_jcode",
        "jshis_avs30",
        "jshis_arv",
        "jshis_avs_eb",
        "jshis_avs_ref",
        "jshis_matched",
        "jshis_missing",
    ]
    remaining = [column for column in result.columns if column not in preferred_order]
    return result[[*preferred_order, *remaining]].sort_values(
        ["boring_id", "measurement_no"], kind="stable"
    )


def check_dataset_quality(dataset: pd.DataFrame) -> None:
    """Validate target rows and report missing predictor attributes."""
    key = ["boring_id", "measurement_no"]
    if dataset.duplicated(key).any():
        raise ValueError("Duplicate borehole measurement slots remain in the output.")
    if dataset[
        [
            "depth",
            "n_value",
            "x",
            "y",
            "surface_z",
            "slope",
        "curvature",
        "dem_missing",
        ]
    ].isna().any(axis=None):
        raise ValueError("Required numeric fields contain missing values in the output.")
    duplicate_depth = dataset.duplicated(["boring_id", "depth"], keep=False)
    if duplicate_depth.any():
        LOGGER.warning(
            "Repeated depths within boreholes: %d measurement rows",
            int(duplicate_depth.sum()),
        )
    if dataset["n_value"].gt(N_VALUE_CAP).any():
        raise ValueError(f"N-values above the configured cap remain: {N_VALUE_CAP}")


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned measurement-level dataset as UTF-8 CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    bedrock_points = dataset[
        ["boring_id", "x", "y", "assumed_bedrock_elevation", "assumed_bedrock_from_n50"]
    ].drop_duplicates("boring_id")
    bedrock_points.to_csv(BEDROCK_POINTS_OUTPUT, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved model dataset: %s", output_path)
    LOGGER.info("Saved assumed-bedrock RBF sources: %s", BEDROCK_POINTS_OUTPUT)


def print_summary(
    source: pd.DataFrame,
    dataset: pd.DataFrame,
    reshape_statistics: dict[str, int],
    categorical_columns: Sequence[str] = ("symbol", "ser", "jshis_jcode"),
) -> None:
    """Print cleaning counts, missingness, and categorical distributions."""
    print("\nDataset cleaning summary")
    print(f"Input boreholes: {len(source)}")
    for name, count in reshape_statistics.items():
        print(f"{name}: {count}")
    print(f"Output measurement rows: {len(dataset)}")
    print(f"Output boreholes: {dataset['boring_id'].nunique()}")
    print(f"Geology-missing rows: {int(dataset['geology_missing'].sum())}")
    print(f"J-SHIS-missing rows: {int(dataset['jshis_missing'].sum())}")
    print("\nNumeric summary:")
    print(
        dataset[
            [
                "depth",
                "surface_z",
                "slope",
                "curvature",
                "n_value_elevation",
                "lowest_spt_elevation",
                "distance_to_lowest_spt",
                "assumed_bedrock_elevation",
                "n_value",
                "jshis_avs30",
                "jshis_arv",
            ]
        ].describe().to_string()
    )
    for column in categorical_columns:
        print(f"\n{column} value counts:")
        print(dataset[column].value_counts(dropna=False).head(30).to_string())


def main() -> None:
    """Run measurement reshaping, attribute merging, and quality checks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    source = load_boring_measurements(BORING_CSV, MAX_MEASUREMENTS)
    measurements, reshape_statistics = reshape_spt_measurements(
        source, MAX_MEASUREMENTS
    )
    spatial = load_spatial_attributes(SPATIAL_INPUT, SPATIAL_LAYER)
    dataset = merge_measurements_and_attributes(measurements, spatial)
    check_dataset_quality(dataset)
    save_dataset(dataset, OUTPUT_CSV)
    print_summary(source, dataset, reshape_statistics)



