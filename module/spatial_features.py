"""Geographic predictors shared by phi and gamma; no target-dependent inputs."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from module.paths import RAW
GEOLOGY_DIR = RAW / 'geology/seamlessV2'
JSHIS_DIR = RAW / 'jshis'
JSHIS_CSV_NAME = 'Z-V4-JAPAN-AMP-VS400_M250.csv'
JSHIS_COLUMNS = ['CODE','JCODE','AVS','ARV','AVS_EB','AVS_REF']
LOGGER = logging.getLogger(__name__)

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

