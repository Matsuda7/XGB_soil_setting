#!/usr/bin/env python3
"""Create unit-weight and cohesion distribution data from strength_data.csv.

KuniJiban density fields are treated as g/cm3 and converted to kN/m3. The
current XML set contains shear strength, not cohesion, so cohesion output is
reported as unavailable unless a real cohesion column is supplied explicitly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from module.paths import INTERIM, RESULTS

import pandas as pd

GRAVITY = 9.80665
DEFAULT_INPUT = INTERIM / "strength/strength_data.csv"
DEFAULT_OUTPUT = RESULTS / "gamma/observations"
DENSITY_COLUMNS = {
    "wet": "wet_density",
    "dry": "dry_density",
}
MODEL_FEATURE_COLUMNS = [
    "boring_id", "xml_file", "latitude", "longitude", "surface_z",
    "depth_top", "depth_bottom", "depth_mid", "sample_z", "sample_id", "sample_no",
    "soil_name", "soil_code", "test_type", "test_condition_code",
    "particle_density", "water_content", "void_ratio", "degree_of_saturation",
    "gravel_fraction", "sand_fraction", "silt_fraction", "clay_fraction",
    "maximum_particle_size", "d10", "d50", "uniformity_coefficient",
    "liquid_limit", "plastic_limit", "plasticity_index",
]


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series, coercing missing and invalid values to NaN."""
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    """Write distribution summary rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variable", "source_column", "count", "min", "max", "mean", "median", "status"])
        writer.writeheader()
        writer.writerows(rows)


def create_distributions(input_path: Path, output_dir: Path, cohesion_column: str | None = None) -> list[dict[str, object]]:
    """Create gamma columns, distribution CSVs, and optional histogram plots."""
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for label, source_column in DENSITY_COLUMNS.items():
        density = numeric_series(frame, source_column)
        gamma_column = f"gamma_{label}_kn_m3"
        result_columns = [column for column in ["boring_id", "xml_file", "sample_id", "sample_no", "depth_mid", "sample_z", "latitude", "longitude", source_column] if column in frame]
        result = frame[result_columns].copy()
        result[gamma_column] = density * GRAVITY
        result = result.loc[density.notna()].copy()
        result.to_csv(output_dir / f"{gamma_column}.csv", index=False, encoding="utf-8-sig")
        model_columns = [column for column in MODEL_FEATURE_COLUMNS if column in frame]
        model_result = frame.loc[density.notna(), model_columns].copy()
        model_result.insert(0, "target", result[gamma_column].to_numpy())
        model_result.to_csv(output_dir / f"model_dataset_{label}_gamma.csv", index=False, encoding="utf-8-sig")
        values = result[gamma_column]
        summary.append({
            "variable": gamma_column,
            "source_column": source_column,
            "count": len(values),
            "min": values.min() if not values.empty else "",
            "max": values.max() if not values.empty else "",
            "mean": values.mean() if not values.empty else "",
            "median": values.median() if not values.empty else "",
            "status": "ok" if not values.empty else "no_data",
        })
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 5))
            plt.hist(values, bins="auto", edgecolor="black")
            plt.xlabel(f"{gamma_column} [kN/m3]")
            plt.ylabel("Count")
            plt.title(f"Distribution of {gamma_column}")
            plt.tight_layout()
            plt.savefig(output_dir / f"{gamma_column}.png", dpi=150)
            plt.close()
            if {"latitude", "longitude"}.issubset(result.columns):
                coordinates = result[["latitude", "longitude", gamma_column]].dropna()
                if not coordinates.empty:
                    plt.figure(figsize=(7, 6))
                    scatter = plt.scatter(
                        coordinates["longitude"], coordinates["latitude"],
                        c=coordinates[gamma_column], cmap="viridis", s=24,
                    )
                    plt.colorbar(scatter, label=f"{gamma_column} [kN/m3]")
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.title(f"Spatial distribution of {gamma_column}")
                    plt.tight_layout()
                    plt.savefig(output_dir / f"{gamma_column}_map.png", dpi=150)
                    plt.close()
        except ImportError:
            pass

    cohesion_name = cohesion_column or "c_total"
    cohesion = numeric_series(frame, cohesion_name)
    if cohesion_column and cohesion.notna().any():
        result = frame[[column for column in ["boring_id", "xml_file", "sample_id", "sample_no", "depth_mid", "sample_z", "latitude", "longitude", cohesion_name] if column in frame]].copy()
        result = result.loc[cohesion.notna()].copy()
        result.to_csv(output_dir / "cohesion.csv", index=False, encoding="utf-8-sig")
        model_columns = [column for column in MODEL_FEATURE_COLUMNS if column in frame]
        model_result = frame.loc[cohesion.notna(), model_columns].copy()
        model_result.insert(0, "target", result[cohesion_name].to_numpy())
        model_result.to_csv(output_dir / "model_dataset_cohesion.csv", index=False, encoding="utf-8-sig")
        status = "ok"
        count = len(result)
        minimum, maximum = result[cohesion_name].min(), result[cohesion_name].max()
        mean, median = result[cohesion_name].mean(), result[cohesion_name].median()
    else:
        cohesion_columns = ["target", *MODEL_FEATURE_COLUMNS]
        pd.DataFrame(columns=cohesion_columns).to_csv(output_dir / "model_dataset_cohesion.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["boring_id", "sample_id", "depth_mid", "sample_z", cohesion_name]).to_csv(output_dir / "cohesion.csv", index=False, encoding="utf-8-sig")
        status = "unavailable: current XML provides shear strength, not cohesion"
        count = 0
        minimum = maximum = mean = median = ""
    summary.append({"variable": "cohesion", "source_column": cohesion_name, "count": count, "min": minimum, "max": maximum, "mean": mean, "median": median, "status": status})
    write_summary(summary, output_dir / "distribution_summary.csv")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohesion-column", default=None, help="Use only a verified cohesion column")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Create distribution files."""
    args = parse_args(argv)
    for row in create_distributions(args.input, args.output_dir, args.cohesion_column):
        print(f"{row['variable']}: n={row['count']} status={row['status']}")
    return 0


