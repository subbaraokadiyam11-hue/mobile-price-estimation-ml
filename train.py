"""
Training Script for Mobile Price Estimation
===========================================
Trains a Linear Regression pipeline to predict mobile phone prices in INR (₹).
Saves the fitted model, evaluation plots, and metrics.
"""

import json
from pathlib import Path
import sys

# Ensure UTF-8 output encoding for Windows command line compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    load_data,
    split_data,
    validate_data,
)


def train_model():
    """Execute end-to-end training, evaluation, and artifact generation."""
    print("=" * 70)
    print("      MOBILE PRICE ESTIMATION: MODEL TRAINING WORKFLOW")
    print("=" * 70)

    # 1. Paths configuration
    data_path = Path("data") / "mobile_prices.csv"
    models_dir = Path("models")
    outputs_dir = Path("outputs")

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_output_path = models_dir / "linear_regression_model.pkl"

    # 2. Check dataset existence
    print(f"\n[1/6] Checking dataset at: {data_path.resolve()}")
    if not data_path.exists():
        print(
            "\n[ERROR] Dataset not found! Please place 'mobile_prices.csv' inside the 'data/' folder."
        )
        print("Expected columns include:")
        print(f"  {FEATURE_COLUMNS} and '{TARGET_COLUMN}'")
        sys.exit(1)

    # 3. Load & Validate Dataset
    print("\n[2/6] Loading and validating dataset...")
    raw_df = load_data(data_path)
    print(f"  Raw dataset shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    cleaned_df = validate_data(raw_df, is_training=True)
    print(f"  Cleaned dataset shape: {cleaned_df.shape[0]} rows, {len(FEATURE_COLUMNS)} features")

    if len(cleaned_df) < 50:
        raise ValueError(
            f"Insufficient data for reliable training. Found only {len(cleaned_df)} valid rows."
        )

    # Display dataset summary
    print("\n  Target Price Summary (in INR ₹):")
    price_stats = cleaned_df[TARGET_COLUMN].describe()
    print(f"    Count: {price_stats['count']:.0f}")
    print(f"    Mean : ₹{price_stats['mean']:,.2f}")
    print(f"    Std  : ₹{price_stats['std']:,.2f}")
    print(f"    Min  : ₹{price_stats['min']:,.2f}")
    print(f"    50%  : ₹{price_stats['50%']:,.2f}")
    print(f"    Max  : ₹{price_stats['max']:,.2f}")

    # 4. Train/Test Split
    print("\n[3/6] Splitting dataset (80% train, 20% test, random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(cleaned_df, test_size=0.2, random_state=42)
    print(f"  Training set size: {X_train.shape[0]} samples")
    print(f"  Testing set size : {X_test.shape[0]} samples")

    # 5. Build and Fit Linear Regression Pipeline
    print("\n[4/6] Building preprocessing & Linear Regression pipeline...")
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    print("  Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    # 6. Model Evaluation on Test Data
    print("\n[5/6] Evaluating model on held-out test data...")
    y_pred = pipeline.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))

    print(f"  R-squared (R2) Score : {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    print(f"  Mean Absolute Error   : ₹{mae:,.2f}")
    print(f"  Mean Squared Error    : {mse:,.2f}")
    print(f"  Root Mean Sq. Error   : ₹{rmse:,.2f}")

    # Save metrics to JSON
    metrics = {
        "algorithm": "Linear Regression",
        "random_state": 42,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "r2_score": round(r2, 4),
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "target_price_min": float(price_stats["min"]),
        "target_price_mean": float(price_stats["mean"]),
        "target_price_max": float(price_stats["max"]),
    }

    metrics_file = outputs_dir / "evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Saved evaluation metrics to: {metrics_file.resolve()}")

    # 7. Generate Evaluation Visualizations
    print("\n[6/6] Generating evaluation plots...")
    sns.set_theme(style="whitegrid")

    # Actual vs Predicted Plot
    fig_avp, ax_avp = plt.subplots(figsize=(8, 6))
    ax_avp.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", color="#2563EB", label="Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax_avp.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#DC2626",
        linestyle="--",
        linewidth=2,
        label="Perfect Fit (y = x)",
    )
    ax_avp.set_title("Actual vs. Predicted Mobile Prices (Linear Regression)", fontsize=14, fontweight="bold")
    ax_avp.set_xlabel("Actual Price (₹)", fontsize=12)
    ax_avp.set_ylabel("Predicted Price (₹)", fontsize=12)
    ax_avp.legend(frameon=True)
    fig_avp.tight_layout()
    avp_plot_path = outputs_dir / "actual_vs_predicted.png"
    fig_avp.savefig(avp_plot_path, dpi=300)
    plt.close(fig_avp)
    print(f"  Saved Actual vs Predicted plot to: {avp_plot_path.resolve()}")

    # Residual Plot
    residuals = y_test - y_pred
    fig_res, ax_res = plt.subplots(figsize=(8, 6))
    ax_res.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", color="#059669")
    ax_res.axhline(0, color="#DC2626", linestyle="--", linewidth=2)
    ax_res.set_title("Residual Plot (Residuals vs. Fitted Values)", fontsize=14, fontweight="bold")
    ax_res.set_xlabel("Predicted Price (₹)", fontsize=12)
    ax_res.set_ylabel("Residuals (Actual - Predicted) (₹)", fontsize=12)
    fig_res.tight_layout()
    res_plot_path = outputs_dir / "residual_plot.png"
    fig_res.savefig(res_plot_path, dpi=300)
    plt.close(fig_res)
    print(f"  Saved Residual plot to: {res_plot_path.resolve()}")

    # Save Pipeline Model
    joblib.dump(pipeline, model_output_path)
    print(f"  Saved fitted pipeline to: {model_output_path.resolve()}")

    # Print sample comparisons
    sample_df = pd.DataFrame({
        "Actual Price (₹)": y_test.iloc[:5].values,
        "Predicted Price (₹)": y_pred[:5],
    })
    sample_df["Absolute Error (₹)"] = (sample_df["Actual Price (₹)"] - sample_df["Predicted Price (₹)"]).abs()

    print("\n" + "=" * 70)
    print("                   TRAINING COMPLETE SUMMARY")
    print("=" * 70)
    print(f"Model File     : {model_output_path.name}")
    print(f"Test R² Score  : {r2:.4f}")
    print(f"Test MAE       : ₹{mae:,.2f}")
    print(f"Test RMSE      : ₹{rmse:,.2f}")
    print("\nSample Comparisons (First 5 Test Predictions):")
    for _, row in sample_df.iterrows():
        print(
            f"  Actual: ₹{row['Actual Price (₹)']:>8,.0f} | "
            f"Predicted: ₹{row['Predicted Price (₹)']:>8,.0f} | "
            f"Abs Error: ₹{row['Absolute Error (₹)']:>8,.0f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    train_model()
