#!/usr/bin/env python3
"""Read the 10 m text DEM and derive pointwise terrain attributes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def load_or_create_dem_cache(
    text_path: Path,
    cache_path: Path,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a memory-mapped DEM, creating a float32 NPY cache when needed."""
    if not text_path.is_file():
        raise FileNotFoundError(f"DEM text file does not exist: {text_path}")
    cache_is_current = (
        cache_path.is_file()
        and cache_path.stat().st_mtime >= text_path.stat().st_mtime
    )
    if not cache_is_current:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        LOGGER.info("Creating DEM cache from %s", text_path)
        target = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=shape
        )
        row_count = 0
        with text_path.open(encoding="utf-8") as stream:
            for row_count, line in enumerate(stream, start=1):
                if row_count > shape[0]:
                    raise ValueError(
                        f"DEM has more than the expected {shape[0]} rows."
                    )
                values = np.fromstring(line, sep=" ", dtype=np.float32)
                if values.size != shape[1]:
                    raise ValueError(
                        f"DEM row {row_count} has {values.size} columns; "
                        f"expected {shape[1]}."
                    )
                target[row_count - 1] = values
        if row_count != shape[0]:
            raise ValueError(f"DEM has {row_count} rows; expected {shape[0]}.")
        target.flush()
        del target
        temporary.replace(cache_path)
        LOGGER.info("Saved DEM cache: %s", cache_path)

    dem = np.load(cache_path, mmap_mode="r")
    if dem.shape != shape:
        raise ValueError(f"DEM cache shape is {dem.shape}; expected {shape}.")
    return dem


def sample_terrain_features(
    points: pd.DataFrame,
    dem: np.ndarray,
    *,
    xmin: float,
    ymin: float,
    spacing: float,
    nodata: float = -200.0,
) -> pd.DataFrame:
    """Interpolate elevation and derive terrain values at exact coordinates.

    DEM elevations are bilinearly interpolated at the point and at one-grid-cell
    offsets. Slope and Laplacian curvature are then calculated by centered finite
    differences. Negative and non-finite DEM cells are treated as elevation zero;
    zero and positive elevations are retained unchanged.
    """
    if spacing <= 0:
        raise ValueError("DEM spacing must be greater than zero.")
    x = pd.to_numeric(points["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(points["y"], errors="coerce").to_numpy(dtype=float)
    result = pd.DataFrame(
        {
            "surface_z": np.nan,
            "slope": np.nan,
            "curvature": np.nan,
            "dem_missing": True,
        },
        index=points.index,
    )

    def interpolate(
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        *,
        fill_outside: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return bilinear values, inside flags, and replaced-cell flags."""
        fractional_c = (sample_x - xmin) / spacing
        fractional_r = (sample_y - ymin) / spacing
        finite = np.isfinite(fractional_c) & np.isfinite(fractional_r)
        within_extent = (
            finite
            & (fractional_c >= 0)
            & (fractional_c <= dem.shape[1] - 1)
            & (fractional_r >= 0)
            & (fractional_r <= dem.shape[0] - 1)
        )
        # At the maximum coordinate, interpolate from the final cell with a
        # fractional position of exactly one.
        c0 = np.floor(np.where(finite, fractional_c, -1)).astype(np.int64)
        r0 = np.floor(np.where(finite, fractional_r, -1)).astype(np.int64)
        c0 = np.minimum(c0, dem.shape[1] - 2)
        r0 = np.minimum(r0, dem.shape[0] - 2)
        inside = within_extent
        values = np.full(len(sample_x), np.nan, dtype=float)
        replaced = np.zeros(len(sample_x), dtype=bool)
        positions = np.flatnonzero(inside)
        if positions.size:
            rows = r0[positions]
            columns = c0[positions]
            corners = np.column_stack(
                [
                    dem[rows, columns],
                    dem[rows, columns + 1],
                    dem[rows + 1, columns],
                    dem[rows + 1, columns + 1],
                ]
            ).astype(float)
            # Negative elevations (including the -200 nodata sentinel) are
            # intentionally treated as zero. Preserve zero and positive values.
            invalid = ~np.isfinite(corners) | (corners < 0)
            replaced[positions] = invalid.any(axis=1)
            corners[invalid] = 0.0
            dc = fractional_c[positions] - columns
            dr = fractional_r[positions] - rows
            values[positions] = (
                corners[:, 0] * (1 - dc) * (1 - dr)
                + corners[:, 1] * dc * (1 - dr)
                + corners[:, 2] * (1 - dc) * dr
                + corners[:, 3] * dc * dr
            )
        if fill_outside:
            values[~inside] = 0.0
            replaced[~inside] = True
        return values, inside, replaced

    samples = [
        interpolate(x, y),
        interpolate(x - spacing, y, fill_outside=True),
        interpolate(x + spacing, y, fill_outside=True),
        interpolate(x, y - spacing, fill_outside=True),
        interpolate(x, y + spacing, fill_outside=True),
    ]
    center, west, east, south, north = [sample[0] for sample in samples]
    # Classification as inside/outside depends on the borehole itself. Offset
    # samples outside the DEM are zero-filled for slope/curvature calculations.
    inside = samples[0][1]
    replaced = np.logical_or.reduce([sample[2] for sample in samples])
    positions = np.flatnonzero(inside)
    if positions.size == 0:
        return result

    dz_dx = (east - west) / (2.0 * spacing)
    dz_dy = (north - south) / (2.0 * spacing)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    curvature = (east - 2.0 * center + west) / spacing**2 + (
        north - 2.0 * center + south
    ) / spacing**2
    result.loc[result.index[positions], "surface_z"] = center[positions]
    result.loc[result.index[positions], "slope"] = slope[positions]
    result.loc[result.index[positions], "curvature"] = curvature[positions]
    result.loc[result.index[positions], "dem_missing"] = replaced[positions]
    result["dem_missing"] = result["dem_missing"].astype(bool)
    return result
