"""
Streamlit Web Application: Mobile Price Estimation Using Machine Learning
========================================================================
Interactive web application for estimating smartphone prices in INR (₹)
using Scikit-learn Linear Regression.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.prediction import MobilePricePredictor, format_indian_currency

# Page configuration
st.set_page_config(
    page_title="Mobile Price Estimation",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean UI styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2563EB;
    }
    .price-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
    }
    .price-title {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_metrics():
    """Load actual model evaluation metrics from outputs/evaluation_metrics.json."""
    metrics_path = Path("outputs") / "evaluation_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    # Header Banner
    st.markdown('<div class="main-header">📱 Mobile Price Estimation Using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Accurately estimate smartphone market prices in Indian Rupees (₹) based on technical hardware specifications.</div>', unsafe_allow_html=True)

    # Navigation Tabs
    tabs = st.tabs(["🏠 Home", "⚙️ Price Estimator", "📊 Model Performance", "ℹ️ Project Information"])

    # -------------------------------------------------------------
    # TAB 1: HOME
    # -------------------------------------------------------------
    with tabs[0]:
        col_intro, col_img = st.columns([3, 2])
        with col_intro:
            st.subheader("Welcome to the Mobile Price Estimator")
            st.write(
                r"""
                In today's fast-moving consumer electronics marketplace, thousands of smartphone models 
                are released every year across varied pricing tiers—from budget entry-level devices to premium 
                flagships.

                **Project Purpose:**
                This project demonstrates the application of **Supervised Machine Learning (Linear Regression)** 
                to model the quantitative relationship between key hardware specifications (such as RAM, internal storage, 
                camera sensors, battery capacity, processor family, and 5G connectivity) and the actual market price in **Indian Rupees (₹)**.

                **Why Linear Regression?**
                Linear Regression is a foundational regression algorithm that models the target variable ($y$, price) 
                as a linear combination of explanatory features ($x_1, x_2, \dots, x_n$):
                """
            )
            st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon")
            st.write(
                r"""
                - **$\beta_0$ (Intercept):** The base expected market price.
                - **$\beta_i$ (Coefficients / Weights):** The marginal price contribution of each unit increase in a specification 
                  (e.g., additional price per GB of RAM, storage, or brand premium).
                - **$\epsilon$ (Residual error):** Discrepancy between actual and predicted price.
                """
            )

        with col_img:
            st.markdown("### 🔍 Key Features Evaluated")
            st.info(
                """
                - **Brand Premium:** Apple, Samsung, OnePlus, Xiaomi, etc.
                - **Memory & Storage:** RAM (GB) and ROM (GB)
                - **Battery:** Capacity in milliampere-hours (mAh)
                - **Display:** Diagonal screen size in inches
                - **Camera Quality:** Primary Rear & Front Camera megapixels (MP)
                - **Processor Family:** Snapdragon, Dimensity/Helio, Bionic, Exynos
                - **Next-Gen Connectivity:** 5G network support
                """
            )
            st.success("Navigate to the **Price Estimator** tab to test your own custom smartphone specifications!")

    # -------------------------------------------------------------
    # TAB 2: SPECIFICATIONS & PREDICTION
    # -------------------------------------------------------------
    with tabs[1]:
        st.subheader("Configure Mobile Phone Specifications")
        st.write("Adjust the specifications below to estimate the predicted market price in Indian Rupees.")

        # Check if model exists
        model_file = Path("models") / "linear_regression_model.pkl"
        if not model_file.exists():
            st.error(
                "⚠️ Trained model file `models/linear_regression_model.pkl` was not found!\n\n"
                "Please run `python train.py` in your terminal to train and save the model before estimating prices."
            )
            return

        # Pre-fill presets for convenience
        presets = {
            "Custom Configuration": None,
            "Apple iPhone 14 (128GB)": {
                "brand_name": "apple",
                "ram_capacity": 6.0,
                "internal_memory": 128.0,
                "battery_capacity": 3279.0,
                "screen_size": 6.1,
                "primary_camera_rear": 12.0,
                "primary_camera_front": 12.0,
                "processor_brand": "bionic",
                "has_5g": 1,
            },
            "Samsung Galaxy S23 5G (256GB)": {
                "brand_name": "samsung",
                "ram_capacity": 8.0,
                "internal_memory": 256.0,
                "battery_capacity": 3900.0,
                "screen_size": 6.1,
                "primary_camera_rear": 50.0,
                "primary_camera_front": 12.0,
                "processor_brand": "snapdragon",
                "has_5g": 1,
            },
            "OnePlus 11 5G (256GB)": {
                "brand_name": "oneplus",
                "ram_capacity": 16.0,
                "internal_memory": 256.0,
                "battery_capacity": 5000.0,
                "screen_size": 6.7,
                "primary_camera_rear": 50.0,
                "primary_camera_front": 16.0,
                "processor_brand": "snapdragon",
                "has_5g": 1,
            },
            "Xiaomi Redmi Note 12 (128GB)": {
                "brand_name": "xiaomi",
                "ram_capacity": 6.0,
                "internal_memory": 128.0,
                "battery_capacity": 5000.0,
                "screen_size": 6.67,
                "primary_camera_rear": 48.0,
                "primary_camera_front": 13.0,
                "processor_brand": "snapdragon",
                "has_5g": 1,
            },
            "Realme Narzo 50i (Budget 32GB)": {
                "brand_name": "realme",
                "ram_capacity": 2.0,
                "internal_memory": 32.0,
                "battery_capacity": 5000.0,
                "screen_size": 6.5,
                "primary_camera_rear": 8.0,
                "primary_camera_front": 5.0,
                "processor_brand": "unisoc",
                "has_5g": 0,
            },
        }

        selected_preset = st.selectbox(
            "📋 Quick-Fill from a Reference Smartphone Model (Optional)",
            options=list(presets.keys()),
            help="Select a reference device to automatically populate realistic specifications, or choose 'Custom Configuration' to enter your own.",
        )

        preset_data = presets[selected_preset] if presets[selected_preset] is not None else {}

        # Brand list based on dataset
        known_brands = [
            "samsung", "apple", "xiaomi", "oneplus", "realme", "vivo",
            "oppo", "motorola", "poco", "google", "iqoo", "tecno",
            "infinix", "nokia", "sony", "huawei", "honor", "asus",
        ]
        default_brand = preset_data.get("brand_name", "samsung")
        brand_index = known_brands.index(default_brand) if default_brand in known_brands else 0

        # Processor brands
        known_processors = [
            "snapdragon", "dimensity", "helio", "bionic", "exynos",
            "google", "unisoc", "tiger", "kirin", "mediatek",
        ]
        default_proc = preset_data.get("processor_brand", "snapdragon")
        proc_index = known_processors.index(default_proc) if default_proc in known_processors else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🏢 Brand & Memory")
            brand = st.selectbox(
                "Brand Name",
                options=known_brands,
                index=brand_index,
                format_func=lambda x: x.title(),
                help="Manufacturer brand. Brand identity strongly impacts market pricing.",
            )

            ram = st.select_slider(
                "RAM Capacity (GB)",
                options=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 18.0, 24.0],
                value=preset_data.get("ram_capacity", 8.0),
                help="System operational RAM memory in Gigabytes.",
            )

            storage = st.select_slider(
                "Internal Storage (GB)",
                options=[8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
                value=preset_data.get("internal_memory", 128.0),
                help="Internal flash ROM storage in Gigabytes.",
            )

        with col2:
            st.markdown("#### ⚡ Processor & Power")
            processor = st.selectbox(
                "Processor Category / Brand",
                options=known_processors,
                index=proc_index,
                format_func=lambda x: x.title(),
                help="System-on-Chip (SoC) chipset architecture series.",
            )

            battery = st.slider(
                "Battery Capacity (mAh)",
                min_value=1500.0,
                max_value=7000.0,
                value=float(preset_data.get("battery_capacity", 5000.0)),
                step=100.0,
                help="Battery size measured in milliampere-hours.",
            )

            has_5g_choice = st.radio(
                "5G Network Connectivity",
                options=["5G Supported (Yes)", "4G / LTE Only (No)"],
                index=0 if preset_data.get("has_5g", 1) == 1 else 1,
                horizontal=True,
                help="Whether the device supports next-generation 5G cellular bands.",
            )
            has_5g = 1 if "Yes" in has_5g_choice else 0

        with col3:
            st.markdown("#### 📸 Display & Cameras")
            screen_size = st.number_input(
                "Screen Size (Inches)",
                min_value=3.5,
                max_value=8.5,
                value=float(preset_data.get("screen_size", 6.5)),
                step=0.1,
                help="Diagonal display dimension in inches.",
            )

            rear_camera = st.number_input(
                "Primary Rear Camera (MP)",
                min_value=5.0,
                max_value=200.0,
                value=float(preset_data.get("primary_camera_rear", 50.0)),
                step=1.0,
                help="Resolution of the main rear camera sensor in Megapixels.",
            )

            front_camera = st.number_input(
                "Primary Front / Selfie Camera (MP)",
                min_value=2.0,
                max_value=64.0,
                value=float(preset_data.get("primary_camera_front", 16.0)),
                step=1.0,
                help="Resolution of the front-facing selfie sensor in Megapixels.",
            )

        st.markdown("---")

        # Estimation Action Button
        estimate_col, _ = st.columns([1, 2])
        with estimate_col:
            estimate_clicked = st.button("🚀 Estimate Mobile Price", type="primary", use_container_width=True)

        if estimate_clicked:
            input_specs = {
                "brand_name": brand,
                "ram_capacity": float(ram),
                "internal_memory": float(storage),
                "battery_capacity": float(battery),
                "screen_size": float(screen_size),
                "primary_camera_rear": float(rear_camera),
                "primary_camera_front": float(front_camera),
                "processor_brand": processor,
                "has_5g": int(has_5g),
            }

            try:
                predictor = MobilePricePredictor()
                raw_pred, formatted_pred = predictor.predict(input_specs)

                # Render estimated price
                st.markdown(
                    f"""
                    <div class="price-box">
                        <div class="price-title">Estimated Market Price</div>
                        <div class="price-value">{formatted_pred}</div>
                        <div style="font-size: 0.95rem; margin-top: 0.5rem; opacity: 0.85;">
                            Raw Linear Value: ₹{raw_pred:,.2f} INR
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Spec breakdown summary
                st.markdown("##### 📝 Specification Summary Considered by Model:")
                st.write(
                    f"- **Brand:** `{brand.title()}` | **Chipset:** `{processor.title()}` | **5G:** `{'Yes' if has_5g else 'No'}`\n"
                    f"- **RAM:** `{ram:.0f} GB` | **Storage:** `{storage:.0f} GB` | **Battery:** `{battery:.0f} mAh`\n"
                    f"- **Display:** `{screen_size:.1f}\"` | **Rear Camera:** `{rear_camera:.0f} MP` | **Front Camera:** `{front_camera:.0f} MP`"
                )

                st.markdown(
                    """
                    <div class="disclaimer-box">
                        ⚠️ <b>Important Disclaimer:</b> This estimated price is generated using a statistical 
                        Linear Regression model trained on smartphone hardware configurations. It is intended for educational 
                        and demonstrative purposes and does not represent an appraisal or guaranteed retail price.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    # -------------------------------------------------------------
    # TAB 3: MODEL PERFORMANCE
    # -------------------------------------------------------------
    with tabs[2]:
        st.subheader("Model Evaluation & Performance Metrics")
        st.write("Quantitative validation results computed on the held-out 20% test dataset.")

        metrics = load_metrics()
        if metrics is not None:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("R² Score (Goodness of Fit)", f"{metrics['r2_score']:.4f}")
                st.caption(f"Explains {metrics['r2_score'] * 100:.1f}% of price variance.")
            with m2:
                st.metric("Mean Absolute Error (MAE)", f"₹{metrics['mae']:,.0f}")
                st.caption("Average absolute prediction deviation.")
            with m3:
                st.metric("Root Mean Squared Error (RMSE)", f"₹{metrics['rmse']:,.0f}")
                st.caption("Penalizes larger pricing errors.")
            with m4:
                st.metric("Test Samples Evaluated", f"{metrics['test_samples']}")
                st.caption(f"Held out from {metrics['train_samples'] + metrics['test_samples']} total.")

            st.markdown("---")

            # Explanation of metrics for beginners
            with st.expander("📖 Plain-English Guide to Model Evaluation Metrics", expanded=False):
                st.markdown(
                    """
                    - **R² Score (Coefficient of Determination):** Ranges from 0 to 1 (can be negative if worse than mean prediction). 
                      A score of ~0.79 means that **79% of the price variations** between phones are explained by our selected hardware features.
                    - **Mean Absolute Error (MAE):** The average difference between the model's prediction and the phone's true price. 
                      An MAE of ~₹10,900 means that, on average, predictions are within ~₹10,900 of actual retail prices across all price bands.
                    - **Root Mean Squared Error (RMSE):** Similar to MAE but squares the errors before averaging, making it sensitive to extreme 
                      outliers (such as ultra-luxury or niche folding phones).
                    """
                )

            # Display generated evaluation charts
            col_chart1, col_chart2 = st.columns(2)
            avp_path = Path("outputs") / "actual_vs_predicted.png"
            res_path = Path("outputs") / "residual_plot.png"

            with col_chart1:
                st.markdown("##### 🎯 Actual vs. Predicted Prices")
                if avp_path.exists():
                    st.image(str(avp_path), caption="Scatter plot of Actual Prices vs Linear Regression Predictions (Red dashed line = Perfect Fit).")
                else:
                    st.info("Run `python train.py` to generate the Actual vs Predicted chart.")

            with col_chart2:
                st.markdown("##### 📉 Residual Plot (Errors vs. Fits)")
                if res_path.exists():
                    st.image(str(res_path), caption="Residual errors (Actual - Predicted). Random scatter around y=0 indicates linear validity.")
                else:
                    st.info("Run `python train.py` to generate the Residual plot.")

        else:
            st.warning(
                "No evaluation metrics found in `outputs/evaluation_metrics.json`.\n"
                "Please run `python train.py` to evaluate the model and create output artifacts."
            )

    # -------------------------------------------------------------
    # TAB 4: PROJECT INFORMATION
    # -------------------------------------------------------------
    with tabs[3]:
        st.subheader("Project Architecture & Documentation")

        st.markdown(
            r"""
            ### 📂 Dataset Overview
            - **Source:** Indian Smartphone Market Specifications & Price Dataset (Smartprix / 91mobiles open repository).
            - **License:** Open Access / Educational & Research Use.
            - **Target Variable:** Retail Price in Indian Rupees (₹).
            - **Records:** ~980 smartphone configurations validated, cleaned, and split into 80% training / 20% testing sets.

            ### 🛠️ Technology Stack
            - **Language:** Python 3
            - **Data Manipulation:** `pandas`, `numpy`
            - **Machine Learning:** `scikit-learn` (`LinearRegression`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`)
            - **Serialization:** `joblib`
            - **Data Visualization:** `matplotlib`, `seaborn`
            - **Web Application:** `streamlit`
            - **Automated Testing:** `pytest`

            ### 🧠 Strengths & Limitations of Linear Regression
            - **Strengths:** Highly interpretable, fast inference, deterministic coefficients, and clear baseline performance without overfitting.
            - **Limitations:** Assumes strictly linear relationships between specifications and price. In reality, pricing is subject to 
              brand luxury premiums, non-linear diminishing returns (e.g. going from 12GB to 16GB RAM does not double price), and marketing tiers.
            
            ### 🔮 Future Enhancements
            1. **Non-linear Models:** Compare against Ridge, Lasso, Random Forest Regressor, and XGBoost.
            2. **Feature Engineering:** Logarithmic price transformation $\log(price)$ to stabilize residual variance across high-end devices.
            3. **Condition & Age Modeling:** Adding age in months and cosmetic condition for used/refurbished smartphone valuation.
            """
        )


if __name__ == "__main__":
    main()
