#!/usr/bin/env python3
"""Train and evaluate an XGBoost model for capped SPT N-values."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from module.paths import project_path
from module.xgb_common import load_model_config
from module.validation import load_config as load_validation_config, split_data, save_split
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from module.xgb_common import (build_regressor, borehole_train_validation_test_split, build_preprocessor, prepare_feature_frame, regression_metrics)
from module.bedrock import FEATURE_COLUMN as BEDROCK_RBF_FEATURE
from module.bedrock import SOURCE_COLUMN as BEDROCK_SOURCE_COLUMN
from module.bedrock import interpolate_bedrock_rbf


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_CSV = project_path("data/training/phi/model_dataset.csv")
MODEL_CONFIG = project_path("config/phi/model.jsonc")
OUTPUT_DIR = project_path("results/phi/model/")
MODEL_OUTPUT = OUTPUT_DIR / "xgb_spt_model.json"
PREPROCESSOR_OUTPUT = OUTPUT_DIR / "xgb_preprocessor.joblib"
METRICS_OUTPUT = OUTPUT_DIR / "metrics.json"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "test_predictions.csv"
IMPORTANCE_OUTPUT = OUTPUT_DIR / "feature_importance.csv"
GROUPED_IMPORTANCE_OUTPUT = OUTPUT_DIR / "feature_importance_grouped.csv"
IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance_top30.png"
GROUPED_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance_grouped.png"
FEATURE_SET_COMPARISON_OUTPUT = OUTPUT_DIR / "feature_set_comparison.csv"
FEATURE_SET_COMPARISON_PLOT = OUTPUT_DIR / "feature_set_comparison.png"
R2_OUTPUT = OUTPUT_DIR / "test_r2.txt"
OBSERVED_PREDICTED_PLOT = OUTPUT_DIR / "observed_vs_predicted_n.png"

TARGET_COLUMN = "n_value"
ID_COLUMN = "boring_id"
TARGET_MIN = 0.0
TARGET_MAX = 50.0
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "x",
    "y",
    "surface_z",
    "jshis_avs30",
]
CATEGORICAL_FEATURES = ["symbol", "jshis_jcode"]
TERRAIN_NUMERIC_FEATURES = ["slope", "curvature", "dem_missing"]

# The prediction elevation preserves vertical position without using depth as a
# separate feature. Borehole-bottom-derived predictors are intentionally excluded.
SELECTED_ADDITIONAL_NUMERIC_FEATURES = [
    "n_value_elevation", "jshis_arv", BEDROCK_RBF_FEATURE,
]
SELECTED_ADDITIONAL_CATEGORICAL_FEATURES = ["ser"]
BEDROCK_RBF_NEIGHBORS = 50
BEDROCK_RBF_SMOOTHING = 0.0
BEDROCK_RBF_KERNEL = "thin_plate_spline"

FEATURE_LABELS = {
    "x": "X coordinate (EPSG:6675)",
    "y": "Y coordinate (EPSG:6675)",
    "surface_z": "Ground-surface elevation",
    "n_value_elevation": "SPT measurement elevation",
    "jshis_avs30": "J-SHIS average S-wave velocity to 30 m (AVS30)",
    "jshis_arv": "J-SHIS surface amplification from Vs=400 m/s (ARV)",
    "symbol": "GSJ geological class",
    "ser": "GSJ geological legend serial number",
    "jshis_jcode": "J-SHIS geomorphological class (JCODE)",
    "slope": "Terrain slope",
    "curvature": "Terrain curvature",
    "dem_missing": "DEM missing or outside flag",
    BEDROCK_RBF_FEATURE: "RBF-interpolated assumed bedrock elevation",
}

XGB_PARAMETERS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": 4,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "importance_type": "gain",
    "early_stopping_rounds": 75,
}

LOGGER = logging.getLogger(__name__)


def load_dataset(path: Path, numeric_features: Sequence[str], categorical_features: Sequence[str]) -> pd.DataFrame:
    """Load and validate the measurement-level dataset from stage 04."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Model dataset does not exist: {path}. Run 03_clean_dataset.py first."
        )
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    configured = {feature for feature in [*numeric_features, *categorical_features] if feature != BEDROCK_RBF_FEATURE}
    required = {
        ID_COLUMN,
        TARGET_COLUMN,
        *configured,
    }
    if BEDROCK_RBF_FEATURE in numeric_features:
        required.add(BEDROCK_SOURCE_COLUMN)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Model dataset is missing required columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError(f"Model dataset contains no rows: {path}")
    if data[[ID_COLUMN, TARGET_COLUMN, "x", "y"]].isna().any(axis=None):
        raise ValueError("ID, target, X, or Y contains missing values.")
    if not data[TARGET_COLUMN].between(TARGET_MIN, TARGET_MAX).all():
        raise ValueError(
            f"Target values must be within [{TARGET_MIN}, {TARGET_MAX}]. "
            "Run the current 03_clean_dataset.py first."
        )
    return data


def select_features(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return available numeric and categorical model feature lists."""
    numeric = [*NUMERIC_FEATURES, *TERRAIN_NUMERIC_FEATURES]
    categorical = list(CATEGORICAL_FEATURES)
    return numeric, categorical










def add_cross_fitted_bedrock_feature(data: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    """Create RBF values without using each borehole as its own source."""
    result = data.copy()
    result[BEDROCK_RBF_FEATURE] = np.nan
    holes = result[[ID_COLUMN, "x", "y", BEDROCK_SOURCE_COLUMN]].drop_duplicates(ID_COLUMN)
    splitter = GroupKFold(n_splits=min(folds, len(holes)))
    for source_index, query_index in splitter.split(holes, groups=holes[ID_COLUMN]):
        source_holes = holes.iloc[source_index]
        query_ids = set(holes.iloc[query_index][ID_COLUMN])
        query_mask = result[ID_COLUMN].isin(query_ids)
        result.loc[query_mask, BEDROCK_RBF_FEATURE] = interpolate_bedrock_rbf(
            source_holes,
            result.loc[query_mask],
            neighbors=BEDROCK_RBF_NEIGHBORS,
            smoothing=BEDROCK_RBF_SMOOTHING,
            kernel=BEDROCK_RBF_KERNEL,
        )
    if result[BEDROCK_RBF_FEATURE].isna().any():
        raise RuntimeError("Cross-fitted assumed-bedrock RBF contains missing values.")
    return result


def train_and_evaluate(
    data: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    validation_config=None,
) -> tuple[xgb.XGBRegressor, ColumnTransformer, pd.DataFrame, dict[str, Any]]:
    """Fit with early stopping and evaluate on untouched borehole test data."""
    validation_config = validation_config or load_validation_config()
    train, validation, test = split_data(data, validation_config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_split((train, validation, test), OUTPUT_DIR / f"split_{validation_config['mode']}.csv")
    if BEDROCK_RBF_FEATURE in numeric_features:
        train = add_cross_fitted_bedrock_feature(train)
        sources = train[
            [ID_COLUMN, "x", "y", BEDROCK_SOURCE_COLUMN]
        ].drop_duplicates(ID_COLUMN)
        for part in (validation, test):
            part[BEDROCK_RBF_FEATURE] = interpolate_bedrock_rbf(
                sources,
                part,
                neighbors=BEDROCK_RBF_NEIGHBORS,
                smoothing=BEDROCK_RBF_SMOOTHING,
                kernel=BEDROCK_RBF_KERNEL,
            )
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_train = preprocessor.fit_transform(
        prepare_feature_frame(train, numeric_features, categorical_features)
    )
    x_validation = preprocessor.transform(
        prepare_feature_frame(validation, numeric_features, categorical_features)
    )
    x_test = preprocessor.transform(
        prepare_feature_frame(test, numeric_features, categorical_features)
    )

    model = build_regressor(XGB_PARAMETERS)
    model.fit(
        x_train,
        train[TARGET_COLUMN],
        eval_set=[(x_validation, validation[TARGET_COLUMN])],
        verbose=False,
    )
    prediction = np.clip(model.predict(x_test), TARGET_MIN, TARGET_MAX)
    prediction_table = test[
        [ID_COLUMN, "measurement_no", "x", "y", "depth", TARGET_COLUMN]
    ].copy()
    prediction_table["predicted_n_value"] = prediction
    prediction_table["residual"] = (
        prediction_table[TARGET_COLUMN] - prediction_table["predicted_n_value"]
    )

    metrics: dict[str, Any] = {
        "test": regression_metrics(test[TARGET_COLUMN], prediction),
        "rows": {
            "total": len(data),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "boreholes": {
            "total": int(data[ID_COLUMN].nunique()),
            "train": int(train[ID_COLUMN].nunique()),
            "validation": int(validation[ID_COLUMN].nunique()),
            "test": int(test[ID_COLUMN].nunique()),
        },
        "split_method": validation_config["mode"],
        "validation_config": validation_config,
        "best_iteration": int(model.best_iteration),
        "target_cap": TARGET_MAX,
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "random_state": RANDOM_STATE,
        "xgb_parameters": XGB_PARAMETERS,
        "bedrock_rbf": {
            "training_values": "5-fold borehole cross-fitted",
            "validation_test_sources": "training boreholes only",
            "neighbors": BEDROCK_RBF_NEIGHBORS,
            "smoothing": BEDROCK_RBF_SMOOTHING,
            "kernel": BEDROCK_RBF_KERNEL,
        },
    }
    return model, preprocessor, prediction_table, metrics


def fit_final_model(
    data: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    n_estimators: int,
) -> tuple[xgb.XGBRegressor, ColumnTransformer]:
    """Refit preprocessing and the final model on every available borehole."""
    data = data.copy()
    if BEDROCK_RBF_FEATURE in numeric_features:
        data = add_cross_fitted_bedrock_feature(data)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_all = preprocessor.fit_transform(
        prepare_feature_frame(data, numeric_features, categorical_features)
    )
    parameters = dict(XGB_PARAMETERS)
    parameters.pop("early_stopping_rounds", None)
    parameters["n_estimators"] = n_estimators
    model = build_regressor(parameters)
    model.fit(x_all, data[TARGET_COLUMN], verbose=False)
    return model, preprocessor


def save_artifacts(
    model: xgb.XGBRegressor,
    preprocessor: ColumnTransformer,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    """Save the model, preprocessing object, metrics, predictions, and importance."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_OUTPUT)
    joblib.dump(preprocessor, PREPROCESSOR_OUTPUT)
    predictions.to_csv(PREDICTIONS_OUTPUT, index=False, encoding="utf-8-sig")
    with METRICS_OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance["description"] = importance["feature"].map(
        lambda name: _feature_display_name(
            str(name), metrics["numeric_features"], metrics["categorical_features"]
        )
    )
    importance.to_csv(IMPORTANCE_OUTPUT, index=False, encoding="utf-8-sig")
    save_feature_importance_outputs(
        importance,
        metrics["numeric_features"],
        metrics["categorical_features"],
    )
    R2_OUTPUT.write_text(
        f"Test R2: {metrics['test']['r2']:.8f}\n",
        encoding="utf-8",
    )
    save_observed_vs_predicted_plot(predictions, metrics["test"]["r2"])
    LOGGER.info("Saved model: %s", MODEL_OUTPUT)
    LOGGER.info("Saved preprocessor: %s", PREPROCESSOR_OUTPUT)
    LOGGER.info("Saved test R2: %s", R2_OUTPUT)
    LOGGER.info("Saved observed-versus-predicted plot: %s", OBSERVED_PREDICTED_PLOT)


def _original_feature_name(
    transformed_name: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> str:
    """Map a transformed/one-hot column back to its original predictor."""
    name = transformed_name.split("__", maxsplit=1)[-1]
    if name.startswith("missingindicator_"):
        name = name.removeprefix("missingindicator_")
    if name in numeric_features:
        return name
    for feature in sorted(categorical_features, key=len, reverse=True):
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def _feature_display_name(
    transformed_name: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> str:
    """Return a self-explanatory label for a transformed model feature."""
    raw_name = transformed_name.split("__", maxsplit=1)[-1]
    is_missing_indicator = raw_name.startswith("missingindicator_")
    original = _original_feature_name(
        transformed_name, numeric_features, categorical_features
    )
    meaning = FEATURE_LABELS.get(original, original)
    if is_missing_indicator:
        return f"{original} missing indicator: {meaning}"
    if original in categorical_features:
        category_prefix = f"{original}_"
        category = (
            raw_name[len(category_prefix):]
            if raw_name.startswith(category_prefix)
            else raw_name
        )
        return f"{original} = {category} ({meaning})"
    return f"{original}: {meaning}"


def _save_horizontal_bar(
    values: pd.DataFrame, path: Path, title: str, max_items: int | None = None
) -> None:
    """Save a readable horizontal feature-importance bar chart."""
    displayed = values.head(max_items).sort_values("importance", ascending=True)
    height = max(5.0, 0.32 * len(displayed) + 1.5)
    fig, ax = plt.subplots(figsize=(12, height))
    labels = displayed.get("description", displayed["feature"])
    ax.barh(labels, displayed["importance"], color="#377eb8")
    ax.set_xlabel("XGBoost feature importance (gain contribution)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_outputs(
    importance: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> None:
    """Save encoded-level and original-predictor-level importance outputs."""
    encoded = importance.copy()
    encoded["description"] = encoded["feature"].map(
        lambda name: _feature_display_name(
            str(name), numeric_features, categorical_features
        )
    )
    total = float(encoded["importance"].sum())
    if total > 0:
        encoded["importance"] = encoded["importance"] / total
    _save_horizontal_bar(
        encoded,
        IMPORTANCE_PLOT,
        "Top 30 XGBoost feature contributions",
        max_items=30,
    )

    grouped = encoded.copy()
    grouped["feature"] = grouped["feature"].map(
        lambda name: _original_feature_name(
            str(name), numeric_features, categorical_features
        )
    )
    grouped = (
        grouped.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    grouped["description"] = grouped["feature"].map(
        lambda name: FEATURE_LABELS.get(str(name), str(name))
    )
    grouped.to_csv(GROUPED_IMPORTANCE_OUTPUT, index=False, encoding="utf-8-sig")
    grouped_plot = grouped.copy()
    # The grouped chart is the primary model-level importance chart. Its labels
    # exactly match the feature column names declared for final model F.
    grouped_plot["description"] = grouped_plot["feature"]
    _save_horizontal_bar(
        grouped_plot,
        GROUPED_IMPORTANCE_PLOT,
        "XGBoost contribution by final model F feature column",
    )
    LOGGER.info("Saved feature-importance plots: %s, %s", IMPORTANCE_PLOT, GROUPED_IMPORTANCE_PLOT)


def save_feature_set_comparison(comparison: pd.DataFrame) -> None:
    """Save metrics and plots for the controlled feature-set ablation study."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        FEATURE_SET_COMPARISON_OUTPUT, index=False, encoding="utf-8-sig"
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    specifications = [
        ("r2", "Test R²", "higher is better"),
        ("mae", "Test MAE", "lower is better"),
        ("rmse", "Test RMSE", "lower is better"),
    ]
    colors = [
        "#4daf4a",
        "#377eb8",
        "#984ea3",
        "#ff7f00",
        "#e41a1c",
        "#999999",
        "#a65628",
    ]
    for ax, (column, title, direction) in zip(axes, specifications):
        bars = ax.bar(comparison["model"], comparison[column], color=colors)
        ax.set_title(f"{title} ({direction})")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, comparison[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )
    fig.suptitle("Controlled predictor-set comparison (same borehole split)")
    fig.tight_layout()
    fig.savefig(FEATURE_SET_COMPARISON_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(
        "Saved feature-set comparison: %s, %s",
        FEATURE_SET_COMPARISON_OUTPUT,
        FEATURE_SET_COMPARISON_PLOT,
    )


def save_observed_vs_predicted_plot(
    predictions: pd.DataFrame, test_r2: float
) -> None:
    """Plot held-out observed SPT N-values against model predictions."""
    observed = predictions[TARGET_COLUMN]
    predicted = predictions["predicted_n_value"]
    lower = min(TARGET_MIN, float(observed.min()), float(predicted.min()))
    upper = max(TARGET_MAX, float(observed.max()), float(predicted.max()))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(observed, predicted, s=10, alpha=0.25, edgecolors="none")
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed SPT N-value")
    ax.set_ylabel("Predicted N-value")
    ax.set_title("Observed vs. predicted N-values (configured holdout)")
    ax.text(
        0.04,
        0.96,
        f"R² = {test_r2:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OBSERVED_PREDICTED_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(metrics: dict[str, Any]) -> None:
    """Print split information and held-out test metrics."""
    print("\nXGBoost training summary")
    print(f"Rows: {metrics['rows']}")
    print(f"Boreholes: {metrics['boreholes']}")
    print(f"Split method: {metrics['split_method']}")
    print(f"Feature selection: {metrics.get('feature_selection', 'not recorded')}")
    print(f"Best iteration: {metrics['best_iteration']}")
    print(f"Test MAE: {metrics['test']['mae']:.4f}")
    print(f"Test RMSE: {metrics['test']['rmse']:.4f}")
    print(f"Test R2: {metrics['test']['r2']:.4f}")
    comparison = metrics.get("feature_set_comparison")
    if comparison:
        table = pd.DataFrame(comparison)[["model", "mae", "rmse", "r2", "best_iteration"]]
        print("\nControlled feature-set comparison:")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """Run borehole splitting, preprocessing, training, evaluation, and saving."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    numeric_features, categorical_features = load_model_config(MODEL_CONFIG)
    data = load_dataset(INPUT_CSV, numeric_features, categorical_features)
    evaluation_model, evaluation_preprocessor, predictions, metrics = train_and_evaluate(
        data, numeric_features, categorical_features
    )
    del evaluation_model, evaluation_preprocessor
    validation_config = load_validation_config()
    if validation_config['additional_spatial'] and validation_config['mode'] != 'spatial':
        spatial_config = dict(validation_config, mode='spatial', additional_spatial=False)
        spatial_model, spatial_preprocessor, spatial_predictions, spatial_metrics = train_and_evaluate(
            data, numeric_features, categorical_features, spatial_config)
        spatial_predictions.to_csv(OUTPUT_DIR / 'spatial_test_predictions.csv', index=False)
        (OUTPUT_DIR / 'spatial_metrics.json').write_text(json.dumps(spatial_metrics, indent=2) + '\n')
        metrics['additional_spatial'] = spatial_metrics
        del spatial_model, spatial_preprocessor
    metrics['additional_spatial_enabled'] = validation_config['additional_spatial']
    metrics["feature_selection"] = "model_config.jsonc"
    metrics["model_config"] = str(MODEL_CONFIG)
    metrics["excluded_dependent_features"] = [
        "depth",
        "lowest_spt_elevation",
        "distance_to_lowest_spt",
        "assumed_bedrock_elevation",
    ]
    metrics["excluded_duplicated_features"] = []

    model, preprocessor = fit_final_model(
        data,
        numeric_features,
        categorical_features,
        metrics["best_iteration"] + 1,
    )
    metrics["final_fit_rows"] = len(data)
    metrics["final_fit_boreholes"] = int(data[ID_COLUMN].nunique())
    save_artifacts(model, preprocessor, predictions, metrics)
    print_summary(metrics)



