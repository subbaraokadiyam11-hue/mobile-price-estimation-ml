"""
Automated Test Suite for Mobile Price Estimation
================================================
Tests model loading, inference, input validation, feature consistency,
currency formatting, and error handling using pytest.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    build_preprocessor,
    validate_data,
)
from src.prediction import MobilePricePredictor, format_indian_currency


# Sample valid smartphone specifications fixture
@pytest.fixture
def sample_valid_specs():
    return {
        "brand_name": "samsung",
        "ram_capacity": 8.0,
        "internal_memory": 128.0,
        "battery_capacity": 5000.0,
        "screen_size": 6.5,
        "primary_camera_rear": 50.0,
        "primary_camera_front": 16.0,
        "processor_brand": "snapdragon",
        "has_5g": 1,
    }


def test_model_loading():
    """Verify that the model pipeline loads correctly from the models directory."""
    model_path = Path("models") / "linear_regression_model.pkl"
    assert model_path.exists(), f"Model file must exist at {model_path} before testing loading."

    predictor = MobilePricePredictor(model_path=model_path)
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict"), "Loaded model must have a 'predict' method."


def test_missing_model_file_raises_error(tmp_path):
    """Verify that attempting to load a non-existent model file raises FileNotFoundError."""
    non_existent_file = tmp_path / "missing_model.pkl"
    with pytest.raises(FileNotFoundError) as exc_info:
        MobilePricePredictor(model_path=non_existent_file)
    assert "not found" in str(exc_info.value).lower()


def test_valid_prediction(sample_valid_specs):
    """Verify inference on valid smartphone specifications."""
    predictor = MobilePricePredictor()
    price, formatted = predictor.predict(sample_valid_specs)

    # Output checks
    assert isinstance(price, (float, int, np.floating))
    assert math.isfinite(price), "Predicted price must be a finite number."
    assert price > 0, "Predicted price must be positive."
    assert isinstance(formatted, str), "Formatted price must be a string."
    assert formatted.startswith("₹"), "Formatted price must start with Indian Rupee symbol '₹'."


def test_prediction_output_type_and_finite(sample_valid_specs):
    """Verify predicted price is standard float and finite across multiple specifications."""
    predictor = MobilePricePredictor()

    variations = [
        {"ram_capacity": 4.0, "internal_memory": 64.0, "brand_name": "xiaomi"},
        {"ram_capacity": 12.0, "internal_memory": 256.0, "brand_name": "oneplus"},
        {"ram_capacity": 6.0, "internal_memory": 128.0, "brand_name": "apple", "processor_brand": "bionic"},
    ]

    for var in variations:
        test_spec = sample_valid_specs.copy()
        test_spec.update(var)
        price, formatted = predictor.predict(test_spec)
        assert isinstance(price, float)
        assert not math.isnan(price)
        assert not math.isinf(price)
        assert price > 0


def test_invalid_negative_feature_values(sample_valid_specs):
    """Verify that negative or out-of-range numerical values raise ValueError."""
    predictor = MobilePricePredictor()

    # Negative RAM
    bad_ram = sample_valid_specs.copy()
    bad_ram["ram_capacity"] = -8.0
    with pytest.raises(ValueError):
        predictor.predict(bad_ram)

    # Zero battery
    bad_battery = sample_valid_specs.copy()
    bad_battery["battery_capacity"] = 0
    with pytest.raises(ValueError):
        predictor.predict(bad_battery)

    # Empty brand
    bad_brand = sample_valid_specs.copy()
    bad_brand["brand_name"] = "  "
    with pytest.raises(ValueError):
        predictor.predict(bad_brand)


def test_missing_required_feature_keys(sample_valid_specs):
    """Verify that omitting a mandatory specification raises ValueError."""
    predictor = MobilePricePredictor()
    incomplete_specs = sample_valid_specs.copy()
    del incomplete_specs["ram_capacity"]

    with pytest.raises(ValueError) as exc_info:
        predictor.predict(incomplete_specs)
    assert "Missing required specifications" in str(exc_info.value)


def test_consistency_between_training_and_inference_features():
    """Verify exact match of feature columns and order between training and inference."""
    expected_features = [
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
    assert FEATURE_COLUMNS == expected_features
    assert set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES) == set(FEATURE_COLUMNS)


def test_preprocessor_pipeline_structure():
    """Verify preprocessor ColumnTransformer initializes transformers properly."""
    preprocessor = build_preprocessor()
    assert hasattr(preprocessor, "transformers")
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert "cat" in transformer_names
    assert "num" in transformer_names


def test_indian_currency_formatting():
    """Verify Indian currency numbering system formatting."""
    assert format_indian_currency(500) == "₹500"
    assert format_indian_currency(15499) == "₹15,499"
    assert format_indian_currency(125000) == "₹1,25,000"
    assert format_indian_currency(1234567) == "₹12,34,567"
    assert format_indian_currency(0) == "₹0"
