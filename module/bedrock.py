#!/usr/bin/env python3
"""Leakage-aware RBF interpolation of assumed bedrock elevation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator


SOURCE_COLUMN = "assumed_bedrock_elevation"
FEATURE_COLUMN = "assumed_bedrock_elevation_rbf"


def prepare_bedrock_sources(points: pd.DataFrame) -> pd.DataFrame:
    """Validate sources and merge duplicate XY locations by their median."""
    required = {"x", "y", SOURCE_COLUMN}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError("Bedrock source columns are missing: " + ", ".join(missing))
    clean = points[["x", "y", SOURCE_COLUMN]].apply(pd.to_numeric, errors="coerce").dropna()
    clean = clean.groupby(["x", "y"], as_index=False)[SOURCE_COLUMN].median()
    if len(clean) < 2:
        raise ValueError("At least two unique bedrock source coordinates are required for RBF.")
    return clean


def interpolate_bedrock_rbf(
    sources: pd.DataFrame,
    query: pd.DataFrame,
    *,
    neighbors: int = 50,
    smoothing: float = 0.0,
    kernel: str = "thin_plate_spline",
    clip_to_source_range: bool = True,
) -> np.ndarray:
    """Interpolate assumed bedrock elevation at query XY coordinates."""
    sources = prepare_bedrock_sources(sources)
    if neighbors < 2:
        raise ValueError("RBF neighbors must be at least 2.")
    if smoothing < 0:
        raise ValueError("RBF smoothing must be zero or greater.")
    xy = query[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError("RBF query coordinates contain missing or non-finite values.")
    source_xy = sources[["x", "y"]].to_numpy(float)
    source_z = sources[SOURCE_COLUMN].to_numpy(float)
    interpolator = RBFInterpolator(
        source_xy,
        source_z,
        neighbors=min(neighbors, len(sources)),
        smoothing=smoothing,
        kernel=kernel,
    )
    predicted = np.asarray(interpolator(xy), dtype=float)
    if clip_to_source_range:
        predicted = np.clip(predicted, source_z.min(), source_z.max())
    return predicted
