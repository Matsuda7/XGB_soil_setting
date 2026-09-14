#!/usr/bin/env python3
"""Validate the KuniJiban CSV and create borehole point data."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from module.paths import project_path

import geopandas as gpd
import pandas as pd

from module.terrain import load_or_create_dem_cache, sample_terrain_features


# Configuration
BORING_CSV = project_path("data/raw/boring/BorToCsv.csv")
BORING_CRS = "EPSG:6675"
OUTPUT_GPKG = project_path("data/interim/boring_points.gpkg")
OUTPUT_LAYER = "boring_points"
DEM_TEXT = project_path("data/raw/dem/z.txt")
DEM_CACHE = project_path("data/cache/dem/z.npy")
DEM_SHAPE = (6581, 6481)
DEM_XMIN = -45800.0
DEM_YMIN = 105400.0
DEM_SPACING = 10.0
DEM_NODATA = -200.0
CONFIG_PATH = project_path("config/prediction_phi.json")

REQUIRED_COLUMNS = ["ファイル名", "坑口座標X", "坑口座標Y", "坑口座標Z"]
LOGGER = logging.getLogger(__name__)


def load_boring_data(csv_path: Path) -> pd.DataFrame:
    """Read the borehole CSV, validate its schema, and coerce coordinates."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Borehole CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Borehole CSV is missing required columns: " + ", ".join(missing))

    result = df[REQUIRED_COLUMNS].copy()
    for column in ["坑口座標X", "坑口座標Y", "坑口座標Z"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    duplicate_count = int(result["ファイル名"].duplicated(keep=False).sum())
    if duplicate_count:
        LOGGER.warning("Duplicate boring_id rows found: %d", duplicate_count)
    return result


def create_boring_geodataframe(
    df: pd.DataFrame, boring_crs: str
) -> tuple[gpd.GeoDataFrame, int]:
    """Remove rows with invalid XY and create point geometries."""
    valid_xy = df["坑口座標X"].notna() & df["坑口座標Y"].notna()
    missing_count = int((~valid_xy).sum())
    valid = df.loc[valid_xy].copy().reset_index(drop=True).rename(
        columns={
            "ファイル名": "boring_id",
            "坑口座標X": "x",
            "坑口座標Y": "y",
            "坑口座標Z": "surface_z",
        }
    )
    valid["_point_id"] = valid.index
    points = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["x"], valid["y"]),
        crs=boring_crs,
    )
    if points.crs is None:
        raise ValueError(f"Unknown or invalid borehole CRS: {boring_crs}")
    if points.geometry.isna().any() or points.geometry.is_empty.any():
        raise ValueError("Null or empty borehole geometry was created.")
    return points, missing_count


def save_boring_points(points: gpd.GeoDataFrame, output_path: Path) -> None:
    """Save validated borehole points for subsequent spatial processing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_file(output_path, layer=OUTPUT_LAYER, driver="GPKG", index=False)
    LOGGER.info("Saved prepared borehole points: %s", output_path)


def load_dem_policy(path: Path) -> tuple[bool, float]:
    """Read the switch controlling whether out-of-DEM boreholes are retained."""
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    include = config.get("include_outside_dem_boreholes", False)
    fill = config.get("outside_dem_fill_value", 0.0)
    if not isinstance(include, bool):
        raise ValueError("include_outside_dem_boreholes must be true or false.")
    if not isinstance(fill, (int, float)):
        raise ValueError("outside_dem_fill_value must be numeric.")
    return include, float(fill)


def add_dem_terrain_attributes(points: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Replace borehole elevation with DEM elevation and add slope/curvature."""
    dem = load_or_create_dem_cache(DEM_TEXT, DEM_CACHE, DEM_SHAPE)
    terrain = sample_terrain_features(
        points,
        dem,
        xmin=DEM_XMIN,
        ymin=DEM_YMIN,
        spacing=DEM_SPACING,
        nodata=DEM_NODATA,
    )
    result = points.rename(columns={"surface_z": "borehole_surface_z"}).copy()
    result[["surface_z", "slope", "curvature", "dem_missing"]] = terrain
    valid = result[["surface_z", "slope", "curvature"]].notna().all(axis=1)
    invalid_count = int((~valid).sum())
    if invalid_count:
        LOGGER.warning(
            "Boreholes outside the valid DEM neighborhood: %d of %d",
            invalid_count,
            len(result),
        )
    include_outside, fill_value = load_dem_policy(CONFIG_PATH)
    if include_outside:
        result.loc[~valid, ["surface_z", "slope", "curvature"]] = fill_value
        result.loc[~valid, "dem_missing"] = True
        return result, invalid_count
    return result.loc[valid].copy(), invalid_count


def main() -> None:
    """Run borehole preparation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    boreholes = load_boring_data(BORING_CSV)
    points, missing_count = create_boring_geodataframe(boreholes, BORING_CRS)
    coordinate_valid_count = len(points)
    points, invalid_dem_count = add_dem_terrain_attributes(points)
    save_boring_points(points, OUTPUT_GPKG)
    print(f"Original boreholes: {len(boreholes)}")
    print(f"Valid coordinate boreholes: {coordinate_valid_count}")
    print(f"Missing coordinate boreholes: {missing_count}")
    print(f"Invalid DEM-neighborhood boreholes: {invalid_dem_count}")
    print(f"Prepared boreholes: {len(points)}")



