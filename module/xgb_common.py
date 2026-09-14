"""Common borehole splits, preprocessing, and metrics from the phi pipeline.

The target and borehole ID can be selected for future cohesion/gamma models.
No N-value caps or phi conversion rules are applied here.
"""
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def borehole_train_validation_test_split(
    data: pd.DataFrame,
    *, target_column: str = "n_value", id_column: str = "boring_id",
    test_size: float = 0.20, validation_size: float = 0.20, random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly split complete boreholes, preventing within-hole leakage."""
    groups = data[id_column]
    if groups.nunique() < 5:
        raise ValueError(f"Too few boreholes for robust splitting: {groups.nunique()}")
    outer = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_validation_index, test_index = next(
        outer.split(data, y=data[target_column], groups=groups)
    )
    train_validation = data.iloc[train_validation_index].copy()
    test = data.iloc[test_index].copy()

    train_validation_groups = train_validation[id_column]
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state + 1,
    )
    train_index, validation_index = next(
        inner.split(
            train_validation,
            y=train_validation[target_column],
            groups=train_validation_groups,
        )
    )
    train = train_validation.iloc[train_index].copy()
    validation = train_validation.iloc[validation_index].copy()

    sets = [set(part[id_column]) for part in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Borehole leakage was detected between data splits.")
    return train, validation, test


def build_preprocessor(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> ColumnTransformer:
    """Create train-only numeric imputation and categorical encoding."""
    # Missing values are imputed without adding missingness flags to the model.
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=False))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def prepare_feature_frame(
    data: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Normalize categorical types before preprocessing."""
    frame = data[[*numeric_features, *categorical_features]].copy()
    for column in categorical_features:
        frame[column] = frame[column].astype("string").fillna("UNKNOWN")
    return frame


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }



def build_regressor(parameters):
    """Construct the shared XGBoost regressor from explicit model parameters."""
    from xgboost import XGBRegressor
    return XGBRegressor(**parameters)
