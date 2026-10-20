#!/usr/bin/env python3
"""Train and evaluate an XGBoost model for capped SPT N-values."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from module.paths import project_path
from module.model_reporting import save_model_reports
from module.xgb_common import load_model_config
from module.configuration import load_model_settings
from module.solphyj_features import COLUMNS as SOLPHYJ_COLUMNS, attach_solphyj, build_n_preprocessor
from module.solphyj_features import SOIL_CATEGORY, attach_soil_inventory
from module.validation import load_config as load_validation_config, split_data, save_split
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
from module.model_reporting import depth_scatter
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from module.xgb_common import (build_regressor, borehole_train_validation_test_split, build_preprocessor, prepare_feature_frame, regression_metrics)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_CSV = project_path("data/training/phi/model_dataset.csv")
MODEL_CONFIG = project_path("config/phi/model.jsonc")
from contextlib import contextmanager
from contextvars import ContextVar
from module.output_layout import current_directory

_TRAINING_OUTPUT = ContextVar('phi_training_output', default=None)


def output_path(name=''):
    directory = _TRAINING_OUTPUT.get()
    if directory is None:
        directory = current_directory() / 'phi' / 'model'
    return directory / name


@contextmanager
def output_paths(directory):
    token = _TRAINING_OUTPUT.set(Path(directory))
    try:
        yield
    finally:
        _TRAINING_OUTPUT.reset(token)


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
    "depth",
    "jshis_avs30",
]
CATEGORICAL_FEATURES = ["symbol", "jshis_jcode"]
TERRAIN_NUMERIC_FEATURES = ["slope", "curvature", "dem_missing"]

# Depth and measurement elevation describe the vertical position.
# Borehole-bottom-derived predictors are intentionally excluded.
SELECTED_ADDITIONAL_NUMERIC_FEATURES = [
    "sample_elevation", "jshis_arv",
]
SELECTED_ADDITIONAL_CATEGORICAL_FEATURES = ["ser"]

FEATURE_LABELS = {
    "x": "X coordinate (EPSG:6675)",
    "y": "Y coordinate (EPSG:6675)",
    "surface_z": "Ground-surface elevation",
    "depth": "Depth below ground surface (m)",
    "sample_elevation": "Sample elevation",
    "jshis_avs30": "J-SHIS average S-wave velocity to 30 m (AVS30)",
    "jshis_arv": "J-SHIS surface amplification from Vs=400 m/s (ARV)",
    "symbol": "GSJ geological class",
    "ser": "GSJ geological legend serial number",
    "jshis_jcode": "J-SHIS geomorphological class (JCODE)",
    "slope": "Terrain slope",
    "curvature": "Terrain curvature",
    "dem_missing": "DEM missing or outside flag",
    "soil_inventory_ssg": "Soil inventory series group (SSG)",
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


def validate_training_depth(value, name):
    """Accept an inclusive depth limit in metres, or None for all depths."""
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not np.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be null or a finite nonnegative number")
    return value


def validate_training_max_depth(value):
    return validate_training_depth(value, 'training_max_depth_m')


def load_dataset(path: Path, numeric_features: Sequence[str], categorical_features: Sequence[str],
                 training_max_depth_m=None, training_min_depth_m=None, include_n50=True) -> pd.DataFrame:
    """Load and validate the measurement-level dataset from stage 04."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Model dataset does not exist: {path}. Run 03_clean_dataset.py first."
        )
    if not isinstance(include_n50, bool):
        raise ValueError('include_n50 must be true or false')
    maximum = validate_training_max_depth(training_max_depth_m)
    minimum = validate_training_depth(training_min_depth_m, 'training_min_depth_m')
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("training_min_depth_m must not exceed training_max_depth_m")
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    input_rows = len(data)
    if minimum is not None or maximum is not None:
        if 'depth' not in data:
            raise ValueError("Depth-limited training requires the depth column")
        depth = pd.to_numeric(data['depth'], errors='coerce')
        data = data.loc[np.isfinite(depth) & depth.between(minimum if minimum is not None else 0,
                                                       maximum if maximum is not None else np.inf)].copy()
        data['depth'] = depth.loc[data.index]
        if data.empty:
            raise ValueError(f"No training records remain in depth range [{minimum}, {maximum}] m")
    data.attrs['training_depth_filter'] = {
        'training_min_depth_m': minimum,
        'training_max_depth_m': maximum,
        'lower_bound_inclusive': True,
        'upper_bound_inclusive': True,
        'input_rows': input_rows,
        'retained_rows': len(data),
        'excluded_rows': input_rows - len(data),
    }
    before_n_filter = len(data)
    if not include_n50:
        data = data.loc[data[TARGET_COLUMN].ne(TARGET_MAX)].copy()
        if data.empty:
            raise ValueError('No training records remain after excluding N=50')
    data.attrs['training_n_filter'] = {
        'include_n50': include_n50,
        'input_rows': before_n_filter,
        'excluded_rows': before_n_filter - len(data),
        'retained_rows': len(data),
    }
    LOGGER.info("Training depth range: [%s, %s] m; retained %s / %s records",
                minimum, maximum, len(data), input_rows)
    if SOIL_CATEGORY in categorical_features:
        crs = json.loads(project_path('config/spatial.json').read_text())['grid_crs']
        data = attach_soil_inventory(data, crs)
    if set(numeric_features) & set(SOLPHYJ_COLUMNS):
        crs = json.loads(project_path('config/spatial.json').read_text())['grid_crs']
        data = attach_solphyj(data, crs)
    configured = {feature for feature in [*numeric_features, *categorical_features]}
    required = {
        ID_COLUMN,
        TARGET_COLUMN,
        *configured,
    }
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










def train_and_evaluate(
    data: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    validation_config=None,
):
    """Five-fold borehole CV, with early stopping isolated from each outer test fold."""
    config = dict(validation_config or load_validation_config())
    if config.get('mode') != 'borehole':
        raise ValueError('Phi 5-fold CV requires borehole mode')
    if type(config.get('n_splits', 5)) is not int or config.get('n_splits', 5) != 5:
        raise ValueError('Phi CV requires n_splits=5')
    config['n_splits'] = 5
    if data[ID_COLUMN].isna().any():
        raise ValueError('Missing borehole IDs')
    holes = np.array(sorted(data[ID_COLUMN].unique()))
    if len(holes) < 5:
        raise ValueError('Phi 5-fold CV requires at least 5 boreholes')
    root = output_path()
    root.mkdir(parents=True, exist_ok=True)
    frame = data.copy()
    frame['cv_row_id'] = np.arange(len(frame))
    folds, predictions, memberships = [], [], []
    splitter = KFold(n_splits=5, shuffle=True, random_state=config['random_state'])
    for fold, (train_holes, test_holes) in enumerate(splitter.split(holes), 1):
        outer_train = frame.loc[frame[ID_COLUMN].isin(holes[train_holes])].copy()
        test = frame.loc[frame[ID_COLUMN].isin(holes[test_holes])].copy()
        inner = GroupShuffleSplit(n_splits=1, test_size=config['validation_size'],
                                  random_state=config['random_state'] + fold)
        ti, vi = next(inner.split(outer_train, groups=outer_train[ID_COLUMN]))
        train, validation = outer_train.iloc[ti], outer_train.iloc[vi]
        if min(len(train), len(validation), len(test)) < 2:
            raise ValueError(f'Fold {fold} needs at least two rows in train, validation and test')
        directory = root / 'evaluation' / f'fold_{fold}'
        directory.mkdir(parents=True, exist_ok=True)
        parts = [part.assign(_validation_group=part[ID_COLUMN].astype(str))
                 for part in (train, validation, test)]
        save_split(parts, directory / 'split.csv')
        memberships.extend(part[['cv_row_id', ID_COLUMN]].assign(fold=fold, split=name)
                           for name, part in zip(('train', 'validation', 'test'), parts))
        pre = build_n_preprocessor(numeric_features, categorical_features)
        xt = pre.fit_transform(prepare_feature_frame(train, numeric_features, categorical_features))
        xv = pre.transform(prepare_feature_frame(validation, numeric_features, categorical_features))
        tuning = build_regressor(XGB_PARAMETERS)
        tuning.fit(xt, train[TARGET_COLUMN], eval_set=[(xv, validation[TARGET_COLUMN])], verbose=False)
        n_estimators = int(tuning.best_iteration) + 1
        save_model_reports(tuning, pre, numeric_features, categorical_features, train[TARGET_COLUMN],
                           directory / 'tuning', 'SPT N-value')
        # Refit on all four outer training folds using the independently chosen tree count.
        with output_paths(directory):
            model, pre = fit_final_model(outer_train, numeric_features, categorical_features, n_estimators,
                                         population="outer-fold training records")
        prediction = np.clip(model.predict(pre.transform(
            prepare_feature_frame(test, numeric_features, categorical_features))), TARGET_MIN, TARGET_MAX).astype(float)
        table = test[['cv_row_id', ID_COLUMN, 'measurement_no', 'x', 'y', 'depth', TARGET_COLUMN]].copy()
        table['fold'] = fold
        table['predicted_n_value'] = prediction
        table['residual'] = table[TARGET_COLUMN] - prediction
        table.to_csv(directory / 'test_predictions.csv', index=False, encoding='utf-8-sig')
        score = dict(fold=fold, **regression_metrics(test[TARGET_COLUMN], prediction),
                     train_rows=len(train), validation_rows=len(validation),
                     refit_rows=len(outer_train), test_rows=len(test),
                     train_boreholes=int(train[ID_COLUMN].nunique()),
                     validation_boreholes=int(validation[ID_COLUMN].nunique()),
                     test_boreholes=int(test[ID_COLUMN].nunique()), n_estimators=n_estimators)
        model.save_model(directory / 'xgb_spt_model.json')
        joblib.dump(pre, directory / 'xgb_preprocessor.joblib')
        with output_paths(directory):
            save_observed_vs_predicted_plot(table, score['r2'])
        folds.append(score)
        predictions.append(table)
        LOGGER.info('Fold %s/5: R2=%.4f RMSE=%.4f MAE=%.4f',
                    fold, score['r2'], score['rmse'], score['mae'])
    oof = pd.concat(predictions, ignore_index=True).sort_values('cv_row_id').reset_index(drop=True)
    if oof.cv_row_id.tolist() != list(range(len(data))):
        raise RuntimeError('CV must evaluate every retained row exactly once')
    fold_table = pd.DataFrame(folds)
    pd.concat(memberships, ignore_index=True).to_csv(root / 'cv_splits.csv', index=False)
    oof.to_csv(root / 'cv_oof_predictions.csv', index=False, encoding='utf-8-sig')
    aggregate = regression_metrics(oof[TARGET_COLUMN], oof.predicted_n_value)
    summary = {name: {'mean': float(fold_table[name].mean()),
                      'std': float(fold_table[name].std(ddof=1))}
               for name in ('r2', 'rmse', 'mae')}
    cv = dict(n_splits=5, group_column=ID_COLUMN, target='n_value',
              folds=folds, summary=summary, std_ddof=1, oof=aggregate,
              final_n_estimators=int(np.median(fold_table.n_estimators)),
              final_tree_count_method='median of fold early-stopping tree counts')
    metrics = dict(test=aggregate, cross_validation=cv,
                   rows={'total': len(data), 'test': len(data)},
                   boreholes={'total': len(holes), 'test': len(holes)},
                   split_method='borehole_5fold_cv', validation_config=config,
                   best_iteration=cv['final_n_estimators'] - 1, target_cap=TARGET_MAX,
                   numeric_features=list(numeric_features), categorical_features=list(categorical_features),
                   random_state=config['random_state'], xgb_parameters=XGB_PARAMETERS,
                   test_population='cross_validation_oof')
    from module.phi_output import save_evaluation_table
    save_evaluation_table(root, metrics)
    return None, None, oof, metrics


def fit_final_model(
    data: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    n_estimators: int,
    population="all-data final fitting records",
) -> tuple[xgb.XGBRegressor, ColumnTransformer]:
    """Refit preprocessing and the final model on every available borehole."""
    data = data.copy()
    preprocessor = build_n_preprocessor(numeric_features, categorical_features)
    x_all = preprocessor.fit_transform(
        prepare_feature_frame(data, numeric_features, categorical_features)
    )
    parameters = dict(XGB_PARAMETERS)
    parameters.pop("early_stopping_rounds", None)
    parameters["n_estimators"] = n_estimators
    model = build_regressor(parameters)
    model.fit(x_all, data[TARGET_COLUMN], verbose=False)
    save_model_reports(model, preprocessor, numeric_features, categorical_features, data[TARGET_COLUMN],
                       output_path(), "SPT N-value", population=population)
    return model, preprocessor


def save_artifacts(
    model: xgb.XGBRegressor,
    preprocessor: ColumnTransformer,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    """Save the model, preprocessing object, metrics, predictions, and importance."""
    output_path().mkdir(parents=True, exist_ok=True)
    model.save_model(output_path('xgb_spt_model.json'))
    joblib.dump(preprocessor, output_path('xgb_preprocessor.joblib'))
    predictions.to_csv(output_path('test_predictions.csv'), index=False, encoding="utf-8-sig")
    with output_path('metrics.json').open("w", encoding="utf-8") as stream:
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
    importance.to_csv(output_path('feature_importance.csv'), index=False, encoding="utf-8-sig")
    save_feature_importance_outputs(
        importance,
        metrics["numeric_features"],
        metrics["categorical_features"],
    )
    output_path('test_r2.txt').write_text(
        f"Test R2: {metrics['test']['r2']:.8f}\n",
        encoding="utf-8",
    )
    save_observed_vs_predicted_plot(predictions, metrics["test"]["r2"])
    if 'cross_validation' in metrics:
        import shutil
        shutil.copyfile(output_path('observed_vs_predicted_n_test.png'),
                        output_path('observed_vs_predicted_n_cv.png'))
    LOGGER.info("Saved model: %s", output_path('xgb_spt_model.json'))
    LOGGER.info("Saved preprocessor: %s", output_path('xgb_preprocessor.joblib'))
    LOGGER.info("Saved test R2: %s", output_path('test_r2.txt'))
    LOGGER.info("Saved observed-versus-predicted plot: %s", output_path('observed_vs_predicted_n_test.png'))


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
        output_path('feature_importance_top30.png'),
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
    grouped.to_csv(output_path('feature_importance_grouped.csv'), index=False, encoding="utf-8-sig")
    grouped_plot = grouped.copy()
    # The grouped chart is the primary model-level importance chart. Its labels
    # exactly match the feature column names declared for final model F.
    grouped_plot["description"] = grouped_plot["feature"]
    _save_horizontal_bar(
        grouped_plot,
        output_path('feature_importance_grouped.png'),
        "XGBoost contribution by final model F feature column",
    )
    LOGGER.info("Saved feature-importance plots: %s, %s", output_path('feature_importance_top30.png'), output_path('feature_importance_grouped.png'))


def save_feature_set_comparison(comparison: pd.DataFrame) -> None:
    """Save metrics and plots for the controlled feature-set ablation study."""
    output_path().mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output_path('feature_set_comparison.csv'), index=False, encoding="utf-8-sig"
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
    fig.savefig(output_path('feature_set_comparison.png'), dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(
        "Saved feature-set comparison: %s, %s",
        output_path('feature_set_comparison.csv'),
        output_path('feature_set_comparison.png'),
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
    depth_scatter(ax, observed, predicted, predictions.get("depth"), s=10, alpha=0.25, edgecolors="none")
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed SPT N-value")
    ax.set_ylabel("Predicted N-value")
    ax.set_title("Observed vs. predicted N-values (CV held-out predictions)"
                 if "fold" in predictions else "Observed vs. predicted N-values (configured holdout)")
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
    fig.savefig(output_path('observed_vs_predicted_n_test.png'), dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(metrics: dict[str, Any]) -> None:
    """Print split information and held-out test metrics."""
    print("\nXGBoost training summary")
    print(f"Rows: {metrics['rows']}")
    print(f"Boreholes: {metrics['boreholes']}")
    print(f"Split method: {metrics['split_method']}")
    print(f"Feature selection: {metrics.get('feature_selection', 'not recorded')}")
    print(f"Best iteration: {metrics['best_iteration']}")
    if "cross_validation" in metrics:
        print("5-fold CV scores (N-value target):")
        print(pd.DataFrame(metrics["cross_validation"]["folds"])[["fold", "r2", "rmse", "mae"]].to_string(index=False))
        print("The test scores below are pooled out-of-fold scores.")
    print(f"Test MAE: {metrics['test']['mae']:.4f}")
    print(f"Test RMSE: {metrics['test']['rmse']:.4f}")
    print(f"Test R2: {metrics['test']['r2']:.4f}")
    comparison = metrics.get("feature_set_comparison")
    if comparison:
        table = pd.DataFrame(comparison)[["model", "mae", "rmse", "r2", "best_iteration"]]
        print("\nControlled feature-set comparison:")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main(output_dir=None) -> None:
    with output_paths(output_dir or current_directory() / 'phi' / 'model'):
        _main()


def _main() -> None:
    """Run borehole splitting, preprocessing, training, evaluation, and saving."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model_config = load_model_settings()
    numeric_features, categorical_features = load_model_config(MODEL_CONFIG)
    maximum = model_config['training_max_depth_m']
    minimum = model_config['training_min_depth_m']
    include_n50 = model_config['include_n50']
    data = load_dataset(INPUT_CSV, numeric_features, categorical_features, maximum, minimum, include_n50)
    depth_filter = dict(data.attrs['training_depth_filter'])
    output_path().mkdir(parents=True, exist_ok=True)
    (output_path() / 'model_config.json').write_text(
        json.dumps({**model_config, 'numeric_features': numeric_features, 'categorical_features': categorical_features}, indent=2) + '\n')
    (output_path() / 'training_depth_filter.json').write_text(
        json.dumps(depth_filter, indent=2) + '\n')
    evaluation_model, evaluation_preprocessor, predictions, metrics = train_and_evaluate(
        data, numeric_features, categorical_features
    )
    del evaluation_model, evaluation_preprocessor
    metrics["training_min_depth_m"] = minimum
    metrics["training_max_depth_m"] = maximum
    metrics["training_depth_filter"] = depth_filter
    metrics["include_n50"] = include_n50
    metrics["training_n_filter"] = dict(data.attrs['training_n_filter'])
    metrics["training_dataset"] = str(INPUT_CSV)
    metrics["feature_selection"] = "config/phi/model.jsonc"
    metrics["model_config"] = str(MODEL_CONFIG)
    metrics["shared_model_config"] = "config/model.json"
    metrics["model_settings"] = {**model_config, 'numeric_features': numeric_features,
                                 'categorical_features': categorical_features}
    metrics["excluded_dependent_features"] = [
        "depth",
        "lowest_spt_elevation",
        "distance_to_lowest_spt",
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
    from module.all_observed_reporting import save_all_observed
    all_features = preprocessor.transform(prepare_feature_frame(data, numeric_features, categorical_features))
    all_prediction = np.clip(model.predict(all_features), TARGET_MIN, TARGET_MAX)
    metrics['all_observed'] = save_all_observed(
        data, all_prediction, TARGET_COLUMN, 'predicted_n_value', output_path(),
        'observed_vs_predicted_n.png', 'SPT N-value (capped at 50)')
    print_summary(metrics)
    from module.phi_output import organize_model_output
    organize_model_output(output_path())

