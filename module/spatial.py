#!/usr/bin/env python3
"""Attach GSJ geology and J-SHIS surface-soil attributes to boreholes."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from module.paths import project_path
from typing import Iterator, Sequence

import geopandas as gpd
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BORING_CSV = project_path("data/raw/boring/BorToCsv.csv")
BORING_POINTS = project_path("data/interim/boring_points.gpkg")
GEOLOGY_DIR = project_path("data/raw/geology/seamlessV2/")
JSHIS_DIR = project_path("data/raw/jshis/")
JSHIS_FILENAME = "Z-V4-JAPAN-AMP-VS400_M250.csv"
JSHIS_COLUMNS = ["CODE", "JCODE", "AVS", "ARV", "AVS_EB", "AVS_REF"]
JSHIS_CHUNK_SIZE = 500_000
OUTPUT_CSV = project_path("data/interim/boring_geology_jshis.csv")
OUTPUT_GPKG = project_path("data/interim/boring_geology_jshis.gpkg")
OUTPUT_LAYER = "boring_geology_jshis"

LOGGER = logging.getLogger(__name__)


def load_prepared_boreholes(path: Path) -> gpd.GeoDataFrame:
    """Load point data produced by 01_prepare_boring.py."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared borehole data does not exist: {path}. "
            "Run 01_prepare_boring.py first."
        )
    points = gpd.read_file(path, layer="boring_points")
    required = {
        "boring_id", "x", "y", "surface_z", "borehole_surface_z",
        "slope", "curvature", "dem_missing", "_point_id", "geometry",
    }
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError("Prepared borehole data is missing columns: " + ", ".join(missing))
    if points.crs is None:
        raise ValueError(f"Prepared borehole CRS is unknown: {path}")
    if not bool(points.geom_type.eq("Point").all()):
        raise ValueError("Prepared borehole data must contain only Point geometry.")
    return points


def _polygon_geometry_types(path: Path) -> set[str]:
    """Read enough of a shapefile to identify its non-empty geometry types."""
    sample = gpd.read_file(path, rows=100)
    return set(sample.geometry.dropna().geom_type.unique())


def find_geology_shapefile(geology_dir: Path) -> list[Path]:
    """Select one or more GSJ polygon shapefiles safely."""
    if not geology_dir.is_dir():
        raise FileNotFoundError(f"Geology directory does not exist: {geology_dir}")

    shapefiles = sorted(geology_dir.rglob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefile found under: {geology_dir}")

    polygon_candidates: list[Path] = []
    inspection_errors: list[str] = []
    for path in shapefiles:
        try:
            geometry_types = _polygon_geometry_types(path)
            if geometry_types and geometry_types <= {"Polygon", "MultiPolygon"}:
                polygon_candidates.append(path)
        except Exception as exc:  # Continue so all available files can be reported.
            inspection_errors.append(f"{path}: {exc}")

    if len(polygon_candidates) == 1:
        LOGGER.info("Selected geology shapefile: %s", polygon_candidates[0])
        return polygon_candidates

    # Official tiled downloads contain one *_poly.shp and one *_line.shp per map
    # sheet. Combining all polygon tiles is safe when every polygon candidate uses
    # the explicit GSJ *_poly naming convention.
    if polygon_candidates and all(path.stem.lower().endswith("_poly") for path in polygon_candidates):
        LOGGER.info("Selected %d tiled geology polygon shapefiles", len(polygon_candidates))
        for path in polygon_candidates:
            LOGGER.info("  %s", path)
        return polygon_candidates

    available = "\n".join(f"  - {path}" for path in shapefiles)
    if not polygon_candidates:
        details = "\n".join(inspection_errors)
        raise ValueError(
            "No polygon geology shapefile could be identified.\n"
            f"Available shapefiles:\n{available}\nInspection errors:\n{details or '  None'}"
        )

    candidates = "\n".join(f"  - {path}" for path in polygon_candidates)
    raise ValueError(
        "Multiple polygon shapefiles were found and cannot be selected safely. "
        "Keep only the GSJ geology polygon dataset in GEOLOGY_DIR or set "
        "GEOLOGY_DIR to a more specific directory.\n"
        f"Polygon candidates:\n{candidates}\nAll shapefiles:\n{available}"
    )


def load_geology(shapefile_paths: Sequence[Path]) -> gpd.GeoDataFrame:
    """Load, combine, and validate one or more geology polygon datasets."""
    parts = [gpd.read_file(path) for path in shapefile_paths]
    source_crs = parts[0].crs
    inconsistent = [str(path) for path, part in zip(shapefile_paths, parts) if part.crs != source_crs]
    if inconsistent:
        raise ValueError("Geology shapefiles have inconsistent CRS: " + ", ".join(inconsistent))
    geology = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=source_crs
    )
    LOGGER.info("Geology CRS: %s", geology.crs)
    LOGGER.info("Geology columns: %s", list(geology.columns))
    if geology.crs is None:
        raise ValueError("Geology CRS is unknown: " + ", ".join(map(str, shapefile_paths)))
    if geology.empty:
        raise ValueError("Geology shapefiles contain no features.")

    empty_count = int(geology.geometry.isna().sum() + geology.geometry.is_empty.sum())
    if empty_count:
        LOGGER.warning("Empty/null geology geometries found and removed: %d", empty_count)
        geology = geology.loc[geology.geometry.notna() & ~geology.geometry.is_empty].copy()

    invalid_types = sorted(set(geology.geom_type) - {"Polygon", "MultiPolygon"})
    if invalid_types:
        raise ValueError(
            "Geology data must contain only Polygon/MultiPolygon geometry; found: "
            + ", ".join(invalid_types)
        )
    if geology.empty:
        raise ValueError("No non-empty polygon geometry remains in geology data.")

    # Avoid GeoPandas join suffixes changing the stable borehole output schema.
    reserved = {"boring_id", "x", "y", "surface_z", "_point_id", "_match_method", "index_right"}
    collisions = sorted((set(geology.columns) - {"geometry"}) & reserved)
    if collisions:
        rename_map = {column: f"geology_{column}" for column in collisions}
        LOGGER.warning("Renaming conflicting geology columns: %s", rename_map)
        geology = geology.rename(columns=rename_map)
    return geology


def _warn_multiple_matches(matches: gpd.GeoDataFrame, method: str) -> None:
    """Log point coordinates and counts for points matching multiple polygons."""
    counts = matches.groupby("_point_id", sort=False).size()
    for point_id, count in counts[counts > 1].items():
        point = matches.loc[matches["_point_id"] == point_id].iloc[0]
        LOGGER.warning(
            "Multiple geology polygons matched by %s: boring_id=%s, X=%s, Y=%s, hits=%d",
            method,
            point["boring_id"],
            point["x"],
            point["y"],
            count,
        )


def _select_one_match(matches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select one deterministic polygon after all multiple matches were reported."""
    # A point exactly on a boundary can intersect more than one polygon. There is no
    # objective geological winner, so the smallest source feature index is used.
    ordered = matches.sort_values(["_point_id", "index_right"], kind="stable")
    return ordered.drop_duplicates("_point_id", keep="first")


def spatial_join_geology(
    boring_gdf: gpd.GeoDataFrame, geology_gdf: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, int]:
    """Transform points to the geology CRS and perform the primary within join."""
    transformed = boring_gdf.to_crs(geology_gdf.crs)
    if transformed.crs != geology_gdf.crs:
        raise ValueError(
            f"CRS mismatch before spatial join: {transformed.crs} != {geology_gdf.crs}"
        )
    joined = gpd.sjoin(transformed, geology_gdf, how="left", predicate="within")
    matched = joined.loc[joined["index_right"].notna()].copy()
    _warn_multiple_matches(matched, "within")
    matched = _select_one_match(matched)
    matched["_match_method"] = "within"
    return matched, len(matched)


def resolve_unmatched_points(
    boring_gdf: gpd.GeoDataFrame,
    geology_gdf: gpd.GeoDataFrame,
    within_matches: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, int, int]:
    """Retry only unmatched points with intersects and return exactly one row per point."""
    matched_ids = set(within_matches["_point_id"])
    unmatched = boring_gdf.loc[~boring_gdf["_point_id"].isin(matched_ids)]
    fallback = gpd.sjoin(unmatched, geology_gdf, how="left", predicate="intersects")
    intersect_matches = fallback.loc[fallback["index_right"].notna()].copy()
    _warn_multiple_matches(intersect_matches, "intersects")
    intersect_matches = _select_one_match(intersect_matches)
    intersect_matches["_match_method"] = "intersects"

    intersect_ids = set(intersect_matches["_point_id"])
    not_found = fallback.loc[
        ~fallback["_point_id"].isin(intersect_ids)
    ].drop_duplicates("_point_id")
    not_found["_match_method"] = pd.NA

    result = gpd.GeoDataFrame(
        pd.concat([within_matches, intersect_matches, not_found], ignore_index=True),
        geometry="geometry",
        crs=boring_gdf.crs,
    ).sort_values("_point_id", kind="stable")
    if result["_point_id"].duplicated().any():
        raise RuntimeError("A borehole remains assigned to multiple output rows.")
    return result, len(intersect_matches), len(not_found)


def save_results(result: gpd.GeoDataFrame, csv_path: Path, gpkg_path: Path) -> None:
    """Save a flat CSV and a point GeoPackage."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    internal_columns = ["geometry", "index_right", "_point_id", "_match_method"]
    result.drop(columns=internal_columns, errors="ignore").to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    result.drop(columns=["index_right", "_point_id"], errors="ignore").to_file(
        gpkg_path, layer=OUTPUT_LAYER, driver="GPKG", index=False
    )
    LOGGER.info("Saved CSV: %s", csv_path)
    LOGGER.info("Saved GeoPackage: %s", gpkg_path)


def print_summary(
    result: gpd.GeoDataFrame,
    geology_columns: Sequence[str],
    original_count: int,
    valid_count: int,
    missing_coordinate_count: int,
    within_count: int,
    intersects_count: int,
    not_found_count: int,
) -> None:
    """Print processing counts and likely categorical geology fields."""
    print("\nProcessing summary")
    print(f"Original boreholes: {original_count}")
    print(f"Valid coordinate boreholes: {valid_count}")
    print(f"Missing coordinate boreholes: {missing_coordinate_count}")
    print(f"\nGeology matched by within: {within_count}")
    print(f"Geology matched by intersects: {intersects_count}")
    print(f"Geology not found: {not_found_count}")
    print(f"\nTotal output boreholes: {len(result)}")

    if result["boring_id"].duplicated().any():
        LOGGER.warning("Output boring_id values are not unique.")
    if result[["x", "y"]].isna().any(axis=None):
        LOGGER.warning("Output contains missing X/Y values.")
    if result.geometry.isna().any() or result.geometry.is_empty.any():
        LOGGER.warning("Output contains null or empty point geometry.")
    if not_found_count:
        LOGGER.warning("Boreholes with no geology match: %d", not_found_count)

    matched_mask = result["_match_method"].notna()
    attribute_columns = [
        column
        for column in geology_columns
        if column != "geometry" and column in result.columns
    ]
    if matched_mask.any() and attribute_columns:
        all_attributes_missing = result.loc[matched_mask, attribute_columns].isna().all(axis=1)
        if all_attributes_missing.any():
            LOGGER.warning(
                "Matched boreholes with all geology attributes missing: %d",
                int(all_attributes_missing.sum()),
            )

    candidates = [
        column
        for column in geology_columns
        if column != "geometry"
        and column in result.columns
        and (isinstance(result[column].dtype, pd.CategoricalDtype) or result[column].dtype == object)
    ]
    print("\nCandidate categorical geology columns:")
    if not candidates:
        print("  No object/category geology columns were found.")
    for column in candidates:
        print(f"\n{column} (unique non-null values: {result[column].nunique(dropna=True)})")
        print(result[column].value_counts(dropna=False).head(20).to_string())


def find_jshis_csv(directory: Path, filename: str) -> Path:
    """Find exactly one extracted J-SHIS V4 national CSV recursively."""
    if not directory.is_dir():
        raise FileNotFoundError(f"J-SHIS directory does not exist: {directory}")
    candidates = sorted(directory.rglob(filename))
    if not candidates:
        raise FileNotFoundError(
            f"Extracted J-SHIS V4 CSV was not found under {directory}."
        )
    if len(candidates) > 1:
        listing = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(f"Multiple J-SHIS V4 CSV files were found:\n{listing}")
    LOGGER.info("Selected J-SHIS CSV: %s", candidates[0])
    return candidates[0]


def coordinate_to_250m_meshcode(longitude: float, latitude: float) -> str:
    """Convert a WGS84/JGD position to a 10-digit JIS 250 m mesh code."""
    if not (100.0 <= longitude < 180.0 and 0.0 <= latitude < 66.0):
        raise ValueError(
            f"Coordinate is outside the Japanese mesh domain: {longitude}, {latitude}"
        )
    lat_scaled = latitude * 1.5
    first_lat = math.floor(lat_scaled)
    lat_fraction = lat_scaled - first_lat
    lon_scaled = longitude - 100.0
    first_lon = math.floor(lon_scaled)
    lon_fraction = lon_scaled - first_lon
    second_lat = min(math.floor(lat_fraction * 8.0), 7)
    second_lon = min(math.floor(lon_fraction * 8.0), 7)
    lat_fraction = lat_fraction * 8.0 - second_lat
    lon_fraction = lon_fraction * 8.0 - second_lon
    third_lat = min(math.floor(lat_fraction * 10.0), 9)
    third_lon = min(math.floor(lon_fraction * 10.0), 9)
    lat_fraction = lat_fraction * 10.0 - third_lat
    lon_fraction = lon_fraction * 10.0 - third_lon
    fourth = 1 + int(lon_fraction >= 0.5) + 2 * int(lat_fraction >= 0.5)
    lon_fraction = lon_fraction * 2.0 % 1.0
    lat_fraction = lat_fraction * 2.0 % 1.0
    fifth = 1 + int(lon_fraction >= 0.5) + 2 * int(lat_fraction >= 0.5)
    return (
        f"{first_lat:02d}{first_lon:02d}{second_lat}{second_lon}"
        f"{third_lat}{third_lon}{fourth}{fifth}"
    )


def add_jshis_meshcodes(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate the J-SHIS 250 m mesh code for every point."""
    wgs84 = points.to_crs("EPSG:4326")
    result = points.copy()
    result["jshis_meshcode"] = [
        coordinate_to_250m_meshcode(point.x, point.y) for point in wgs84.geometry
    ]
    LOGGER.info("Unique target J-SHIS meshes: %d", result["jshis_meshcode"].nunique())
    return result


def read_jshis_chunks(path: Path) -> Iterator[pd.DataFrame]:
    """Yield normalized chunks from the nationwide J-SHIS CSV."""
    return pd.read_csv(
        path,
        comment="#",
        header=None,
        names=JSHIS_COLUMNS,
        dtype={"CODE": "string"},
        na_values=["-"],
        skipinitialspace=True,
        chunksize=JSHIS_CHUNK_SIZE,
        encoding="ascii",
    )


def load_target_jshis_rows(path: Path, target_codes: set[str]) -> pd.DataFrame:
    """Read only the J-SHIS rows needed by the borehole points."""
    matches: list[pd.DataFrame] = []
    rows_scanned = 0
    for chunk in read_jshis_chunks(path):
        rows_scanned += len(chunk)
        chunk["CODE"] = chunk["CODE"].str.strip()
        selected = chunk.loc[chunk["CODE"].isin(target_codes)]
        if not selected.empty:
            matches.append(selected.copy())
    LOGGER.info("J-SHIS nationwide rows scanned: %d", rows_scanned)
    if not matches:
        return pd.DataFrame(columns=JSHIS_COLUMNS)
    result = pd.concat(matches, ignore_index=True)
    if result["CODE"].duplicated().any():
        raise ValueError("Duplicate mesh codes were found in the J-SHIS CSV.")
    for column in ["JCODE", "AVS", "ARV", "AVS_EB", "AVS_REF"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def join_jshis_attributes(
    points: gpd.GeoDataFrame, jshis: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Join J-SHIS surface-soil attributes by 250 m mesh code."""
    attributes = jshis.rename(
        columns={
            "CODE": "jshis_meshcode",
            "JCODE": "jshis_jcode",
            "AVS": "jshis_avs30",
            "ARV": "jshis_arv",
            "AVS_EB": "jshis_avs_eb",
            "AVS_REF": "jshis_avs_ref",
        }
    )
    result = points.merge(
        attributes, on="jshis_meshcode", how="left", validate="many_to_one"
    )
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=points.crs)
    invalid = (
        result["jshis_jcode"].isna()
        | result["jshis_jcode"].eq(0)
        | result["jshis_avs30"].isna()
        | result["jshis_avs30"].le(0)
        | result["jshis_arv"].isna()
        | result["jshis_arv"].le(0)
    )
    result["jshis_matched"] = ~invalid
    return result


def print_jshis_summary(result: gpd.GeoDataFrame) -> None:
    """Print J-SHIS matching statistics."""
    matched = result["jshis_matched"]
    print("\nJ-SHIS processing summary")
    print(f"Input boreholes: {len(result)}")
    print(f"Unique 250 m meshes: {result['jshis_meshcode'].nunique()}")
    print(f"J-SHIS matched: {int(matched.sum())}")
    print(f"J-SHIS not found/invalid: {int((~matched).sum())}")
    print("\nJCODE value counts:")
    print(result.loc[matched, "jshis_jcode"].value_counts().sort_index().to_string())
    print("\nAVS30 statistics:")
    print(result.loc[matched, "jshis_avs30"].describe().to_string())
    print("\nARV statistics:")
    print(result.loc[matched, "jshis_arv"].describe().to_string())


def main() -> None:
    """Attach both GSJ geology and J-SHIS attributes to prepared boreholes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    boring_gdf = load_prepared_boreholes(BORING_POINTS)
    if not BORING_CSV.is_file():
        raise FileNotFoundError(f"Borehole CSV does not exist: {BORING_CSV}")
    source_coordinates = pd.read_csv(
        BORING_CSV,
        encoding="utf-8-sig",
        usecols=["坑口座標X", "坑口座標Y"],
        low_memory=False,
    ).apply(pd.to_numeric, errors="coerce")
    original_count = len(source_coordinates)
    missing_coordinate_count = int(source_coordinates.isna().any(axis=1).sum())
    geology_paths = find_geology_shapefile(GEOLOGY_DIR)
    geology_gdf = load_geology(geology_paths)
    geology_columns = list(geology_gdf.columns)

    # Transform once. This also makes the output GeoPackage use the geology CRS.
    boring_gdf = boring_gdf.to_crs(geology_gdf.crs)
    within_matches, within_count = spatial_join_geology(boring_gdf, geology_gdf)
    result, intersects_count, not_found_count = resolve_unmatched_points(
        boring_gdf, geology_gdf, within_matches
    )
    result = add_jshis_meshcodes(result)
    jshis_path = find_jshis_csv(JSHIS_DIR, JSHIS_FILENAME)
    jshis = load_target_jshis_rows(jshis_path, set(result["jshis_meshcode"]))
    result = join_jshis_attributes(result, jshis)
    save_results(result, OUTPUT_CSV, OUTPUT_GPKG)
    print_summary(
        result=result,
        geology_columns=geology_columns,
        original_count=original_count,
        valid_count=len(boring_gdf),
        missing_coordinate_count=missing_coordinate_count,
        within_count=within_count,
        intersects_count=intersects_count,
        not_found_count=not_found_count,
    )
    print_jshis_summary(result)



