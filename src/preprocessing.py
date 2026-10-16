"""
Preprocessing Module for Mobile Price Estimation
================================================
Handles data loading, validation, cleaning, feature transformation,
and train-test splitting.
"""

from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical feature definitions
FEATURE_COLUMNS: List[str] = [
    "brand_name",
    "ram_capacity",
    "internal_memory",
    "battery_capacity",
    "screen_size",
    "primary_camera_rear",
    "primary_camera_front",
    "processor_brand",
    "has_5g",
]

CATEGORICAL_FEATURES: List[str] = [
    "brand_name",
    "processor_brand",
]

NUMERICAL_FEATURES: List[str] = [
    "ram_capacity",
    "internal_memory",
    "battery_capacity",
    "screen_size",
    "primary_camera_rear",
    "primary_camera_front",
    "has_5g",
]

TARGET_COLUMN: str = "price"


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load mobile prices dataset from a CSV file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {path.resolve()}\n"
            "Please ensure 'data/mobile_prices.csv' is present."
        )
    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Validate and clean the mobile specifications dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw DataFrame.
    is_training : bool, default=True
        Whether validation is performed during training (requires target column).

    Returns
    -------
    pd.DataFrame
        Cleaned and validated DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing or numerical values are out of bounds.
    """
    df_clean = df.copy()

    # Check for required features
    missing_features = [col for col in FEATURE_COLUMNS if col not in df_clean.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    if is_training:
        if TARGET_COLUMN not in df_clean.columns:
            raise ValueError(f"Missing target column '{TARGET_COLUMN}' in training data.")

        # Remove duplicate records
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        dropped_duplicates = initial_len - len(df_clean)
        if dropped_duplicates > 0:
            print(f"Removed {dropped_duplicates} duplicate record(s).")

        # Validate target price > 0
        df_clean = df_clean[df_clean[TARGET_COLUMN].notna() & (df_clean[TARGET_COLUMN] > 0)]

    # Standardize string fields (lowercase, strip whitespace)
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df_clean.columns:
            df_clean[cat_col] = (
                df_clean[cat_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan, "none": np.nan, "": np.nan})
            )

    # Convert boolean 5G flag to integer 0/1
    if "has_5g" in df_clean.columns:
        if df_clean["has_5g"].dtype == bool:
            df_clean["has_5g"] = df_clean["has_5g"].astype(int)
        else:
            # Handle string or numeric representations like 'True', 'False', 1, 0
            df_clean["has_5g"] = (
                df_clean["has_5g"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes", "1.0"])
                .astype(int)
            )

    # Validate numerical features (must be non-negative)
    for num_col in [col for col in NUMERICAL_FEATURES if col != "has_5g"]:
        df_clean[num_col] = pd.to_numeric(df_clean[num_col], errors="coerce")
        if is_training:
            # Drop records with negative numerical specs
            df_clean = df_clean[df_clean[num_col].isna() | (df_clean[num_col] > 0)]
        else:
            invalid_mask = df_clean[num_col].notna() & (df_clean[num_col] <= 0)
            if invalid_mask.any():
                raise ValueError(f"Numerical specification '{num_col}' must be strictly positive.")

    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """Construct a Scikit-learn ColumnTransformer for categorical and numerical features.

    Returns
    -------
    ColumnTransformer
        Fitted or unfitted preprocessor pipeline.
    """
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer, NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the cleaned dataset into training and testing feature/target sets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and validated DataFrame.
    test_size : float, default=0.2
        Proportion of dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
