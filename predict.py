"""
Command-Line Prediction Script for Mobile Price Estimation
==========================================================
Demonstrates price prediction for a smartphone using the trained
Linear Regression pipeline. Can be run standalone or with CLI arguments.
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows command line compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.prediction import MobilePricePredictor, format_indian_currency


def run_prediction(specs: dict) -> None:
    """Load the model, validate inputs, and print predicted price."""
    print("=" * 65)
    print("         MOBILE PHONE PRICE PREDICTION (LINEAR REGRESSION)")
    print("=" * 65)
    print("\nInput Smartphone Specifications:")
    print(f"  Brand            : {specs['brand_name'].title()}")
    print(f"  RAM              : {specs['ram_capacity']} GB")
    print(f"  Internal Storage : {specs['internal_memory']} GB")
    print(f"  Battery Capacity : {specs['battery_capacity']} mAh")
    print(f"  Screen Size      : {specs['screen_size']} inches")
    print(f"  Rear Camera      : {specs['primary_camera_rear']} MP")
    print(f"  Front Camera     : {specs['primary_camera_front']} MP")
    print(f"  Processor Brand  : {specs['processor_brand'].title()}")
    print(f"  5G Network       : {'Yes' if specs['has_5g'] else 'No'}")
    print("-" * 65)

    try:
        predictor = MobilePricePredictor()
        raw_price, formatted_price = predictor.predict(specs)

        print("\nEstimation Result:")
        print(f"  Estimated Price (₹) : {formatted_price}")
        print(f"  Raw Linear Output   : ₹{raw_price:,.2f}")
        print("\nNote: This is an estimated price based on trained linear coefficients.")
        print("Market pricing may vary based on exact model, build quality, and brand premiums.")
    except FileNotFoundError as fnf_err:
        print(f"\n[ERROR] {fnf_err}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Prediction failed: {exc}")
        sys.exit(1)

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Predict mobile phone price in INR from hardware specifications."
    )
    parser.add_argument("--brand", type=str, default="samsung", help="Brand name (e.g., samsung, apple, xiaomi, oneplus)")
    parser.add_argument("--ram", type=float, default=8.0, help="RAM in GB (e.g., 6.0, 8.0, 12.0)")
    parser.add_argument("--storage", type=float, default=128.0, help="Internal storage in GB (e.g., 64.0, 128.0, 256.0)")
    parser.add_argument("--battery", type=float, default=5000.0, help="Battery capacity in mAh (e.g., 4500.0, 5000.0)")
    parser.add_argument("--screen", type=float, default=6.5, help="Screen size in inches (e.g., 6.1, 6.7)")
    parser.add_argument("--rear_camera", type=float, default=50.0, help="Primary rear camera in MP (e.g., 50.0, 64.0)")
    parser.add_argument("--front_camera", type=float, default=16.0, help="Primary front camera in MP (e.g., 12.0, 16.0, 32.0)")
    parser.add_argument("--processor", type=str, default="snapdragon", help="Processor brand/series (e.g., snapdragon, mediatek, bionic, exynos)")
    parser.add_argument("--has_5g", type=int, choices=[0, 1], default=1, help="5G support (1 for Yes, 0 for No)")

    args = parser.parse_args()

    specs = {
        "brand_name": args.brand,
        "ram_capacity": args.ram,
        "internal_memory": args.storage,
        "battery_capacity": args.battery,
        "screen_size": args.screen,
        "primary_camera_rear": args.rear_camera,
        "primary_camera_front": args.front_camera,
        "processor_brand": args.processor,
        "has_5g": args.has_5g,
    }

    run_prediction(specs)


if __name__ == "__main__":
    main()
