"""
Prediction Module for Mobile Price Estimation
=============================================
Provides the MobilePricePredictor class to load the trained regression pipeline,
validate runtime inputs, and estimate mobile phone prices in Indian Rupees (₹).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS, validate_data


def format_indian_currency(amount: Union[int, float]) -> str:
    """Format a numerical amount into Indian currency numbering style.

    Example:
    15499 -> ₹15,499
    125000 -> ₹1,25,000
    1234567 -> ₹12,34,567

    Parameters
    ----------
    amount : Union[int, float]
        Monetary amount.

    Returns
    -------
    str
        Formatted string prefixed with ₹.
    """
    if amount is None or np.isnan(amount):
        return "₹0"

    rounded_val = int(round(amount))
    sign = "-" if rounded_val < 0 else ""
    val_str = str(abs(rounded_val))

    if len(val_str) <= 3:
        return f"{sign}₹{val_str}"

    last_three = val_str[-3:]
    remaining = val_str[:-3]

    # Split remaining digits in groups of 2 from right to left
    groups = []
    while remaining:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]

    formatted_str = ",".join(groups) + "," + last_three
    return f"{sign}₹{formatted_str}"


class MobilePricePredictor:
    """Predictor class that encapsulates the fitted Linear Regression pipeline."""

    DEFAULT_MODEL_PATH = Path("models") / "linear_regression_model.pkl"

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize the predictor.

        Parameters
        ----------
        model_path : Optional[Union[str, Path]], optional
            Path to the saved joblib model file. Defaults to models/linear_regression_model.pkl.
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model pipeline from disk.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        RuntimeError
            If the model file cannot be deserialized.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path.resolve()}.\n"
                "Please train the model first by running:\n"
                "  python train.py"
            )
        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {exc}") from exc

    def validate_specifications(self, specs: Dict[str, Any]) -> pd.DataFrame:
        """Validate input dictionary and convert it to a structured DataFrame.

        Parameters
        ----------
        specs : Dict[str, Any]
            Dictionary containing mobile specifications.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame matching training features and order.

        Raises
        ------
        ValueError
            If any required specification is missing or has an invalid value.
        """
        if not isinstance(specs, dict):
            raise TypeError(f"Specifications must be provided as a dictionary, got {type(specs)}.")

        missing_fields = [f for f in FEATURE_COLUMNS if f not in specs]
        if missing_fields:
            raise ValueError(f"Missing required specifications: {missing_fields}")

        # Basic range checks for realistic mobile hardware specs
        try:
            ram = float(specs["ram_capacity"])
            storage = float(specs["internal_memory"])
            battery = float(specs["battery_capacity"])
            screen = float(specs["screen_size"])
            rear_cam = float(specs["primary_camera_rear"])
            front_cam = float(specs["primary_camera_front"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"All numerical specifications must be valid numbers: {exc}") from exc

        if ram <= 0 or ram > 64:
            raise ValueError(f"Invalid RAM ({ram} GB). Expected range: 0.5 to 64 GB.")
        if storage <= 0 or storage > 2048:
            raise ValueError(f"Invalid Internal Storage ({storage} GB). Expected range: 4 to 2048 GB.")
        if battery <= 0 or battery > 20000:
            raise ValueError(f"Invalid Battery ({battery} mAh). Expected range: 500 to 20000 mAh.")
        if screen <= 0 or screen > 12.0:
            raise ValueError(f"Invalid Screen Size ({screen} inches). Expected range: 2.0 to 12.0 inches.")
        if rear_cam < 0 or rear_cam > 500:
            raise ValueError(f"Invalid Rear Camera ({rear_cam} MP). Expected non-negative value.")
        if front_cam < 0 or front_cam > 200:
            raise ValueError(f"Invalid Front Camera ({front_cam} MP). Expected non-negative value.")

        brand = str(specs.get("brand_name", "")).strip()
        if not brand:
            raise ValueError("Brand name cannot be empty.")

        processor = str(specs.get("processor_brand", "")).strip()
        if not processor:
            raise ValueError("Processor brand/category cannot be empty.")

        # Build clean DataFrame with exact feature order
        row_dict = {col: specs[col] for col in FEATURE_COLUMNS}
        df_input = pd.DataFrame([row_dict])
        df_input = validate_data(df_input, is_training=False)
        return df_input[FEATURE_COLUMNS]

    def predict(self, specs: Union[Dict[str, Any], pd.DataFrame]) -> Tuple[float, str]:
        """Predict the mobile price in Indian Rupees.

        Parameters
        ----------
        specs : Union[Dict[str, Any], pd.DataFrame]
            Mobile specifications as a dictionary or single-row DataFrame.

        Returns
        -------
        Tuple[float, str]
            Tuple of (raw_predicted_price_in_inr, formatted_inr_string).

        Raises
        ------
        RuntimeError
            If model is not loaded.
        ValueError
            If input fails validation.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call _load_model() or initialize properly.")

        if isinstance(specs, dict):
            df_input = self.validate_specifications(specs)
        elif isinstance(specs, pd.DataFrame):
            df_input = validate_data(specs.copy(), is_training=False)[FEATURE_COLUMNS]
        else:
            raise TypeError("Input must be a dictionary or a pandas DataFrame.")

        raw_pred = float(self.model.predict(df_input)[0])

        # Linear regression extrapolation safeguard:
        # If linear combination extrapolates to an implausibly low or negative number,
        # floor to a baseline minimum market price (e.g. ₹1,500 for entry basic phone)
        final_price = max(raw_pred, 1500.0)
        formatted = format_indian_currency(final_price)

        return final_price, formatted
