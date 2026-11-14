import os
import sys

# Ensure root workspace directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import io
from src.preprocessor import InsurancePreprocessor, MedicalFeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Price Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern medical dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .risk-low { background-color: #dcfce7; color: #166534; }
    .risk-moderate { background-color: #dbeafe; color: #1e40af; }
    .risk-high { background-color: #ffedd5; color: #9a3412; }
    .risk-critical { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    """Loads champion model and preprocessor pipeline."""
    model_path = "models/best_model.pkl"
    preprocessor_path = "models/preprocessor.pkl"
    metrics_path = "models/model_metrics.json"
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    return model, preprocessor, metrics

@st.cache_data
def load_dataset():
    """Loads raw dataset for EDA."""
    path = "data/raw/insurance.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

model, preprocessor, metrics = load_ml_assets()
df_raw = load_dataset()

# Title banner
st.markdown('<div class="main-header">🏥 Medical Insurance Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">College Main Project | Machine Learning Predictive Analytics & Actuarial Risk Engine</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Individual Predictor", 
    "📊 EDA Visualizations", 
    "🏆 Model Benchmarks", 
    "📁 Batch Prediction (CSV)", 
    "📖 Project Overview"
])

# -------------------------------------------------------------
# TAB 1: INDIVIDUAL PREDICTOR
# -------------------------------------------------------------
with tab1:
    st.markdown("### 📋 Enter Beneficiary Information")
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    
    with col1:
        age = st.slider("Beneficiary Age", min_value=18, max_value=85, value=30, step=1, help="Age of primary insured individual.")
        sex = st.selectbox("Biological Sex", options=["Male", "Female"], index=0)
        children = st.number_input("Number of Dependents / Children", min_value=0, max_value=10, value=1, step=1)

    with col2:
        use_calc = st.checkbox("Calculate BMI from Height & Weight", value=False)
        if use_calc:
            w_col, h_col = st.columns(2)
            with w_col:
                weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)
            with h_col:
                height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=175.0, step=1.0)
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
            st.info(f"Computed BMI: **{bmi}**")
        else:
            bmi = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=55.0, value=26.5, step=0.1)

        smoker = st.radio("Tobacco / Smoking Status", options=["No", "Yes"], index=0, horizontal=True)
        region = st.selectbox("US Residential Region", options=["northeast", "northwest", "southeast", "southwest"], index=2)

    with col3:
        st.markdown("#### 💡 Clinical Insights")
        bmi_status = "Underweight (< 18.5)" if bmi < 18.5 else "Normal (18.5 - 24.9)" if bmi < 25 else "Overweight (25.0 - 29.9)" if bmi < 30 else "Obese (≥ 30.0)"
        st.write(f"• **BMI Classification:** `{bmi_status}`")
        if smoker == "Yes":
            st.warning("⚠️ **Smoker Flag:** Smoking increases respiratory & cardiovascular claims, typically triggering a 200%+ premium multiplier.")
        else:
            st.success("✅ **Non-Smoker:** Qualifies for primary preventive health discount tier.")

    predict_btn = st.button("🔮 Predict Insurance Annual Charge", type="primary", use_container_width=True)

    if predict_btn:
        if model is None or preprocessor is None:
            st.error("Model assets not loaded! Please execute `python train.py` first to generate models.")
        else:
            # Prepare input
            input_df = pd.DataFrame([{
                "age": age,
                "sex": sex.lower(),
                "bmi": bmi,
                "children": children,
                "smoker": smoker.lower(),
                "region": region.lower()
            }])
            
            # Preprocess and predict
            X_proc = preprocessor.transform(input_df)
            pred_charges = float(model.predict(X_proc)[0])
            pred_charges = max(0.0, pred_charges)
            
            # Risk categorization
            if pred_charges < 5000:
                risk_tier, badge_class, risk_desc = "Low Risk Tier", "risk-low", "Low expected actuarial claims. Eligible for baseline preventive policy."
            elif pred_charges < 15000:
                risk_tier, badge_class, risk_desc = "Moderate Risk Tier", "risk-moderate", "Standard claim exposure. Regular comprehensive coverage recommended."
            elif pred_charges < 30000:
                risk_tier, badge_class, risk_desc = "High Risk Tier", "risk-high", "Elevated medical probability due to age or chronic health markers."
            else:
                risk_tier, badge_class, risk_desc = "Very High / Critical Risk", "risk-critical", "Compounded risk factors (smoker + high BMI). Premium underwriting tier applied."

            st.markdown("---")
            st.markdown("### 📊 Prediction & Underwriting Summary")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label="Estimated Annual Premium", value=f"${pred_charges:,.2f}")
            with res_col2:
                st.metric(label="Estimated Monthly Payment", value=f"${(pred_charges / 12):,.2f}")
            with res_col3:
                st.markdown(f"**Actuarial Risk Assessment:**")
                st.markdown(f'<span class="risk-badge {badge_class}">{risk_tier}</span>', unsafe_allow_html=True)
                st.caption(risk_desc)

            # Recommendations & Savings Calculator
            st.markdown("#### 💡 Cost Optimization & Preventative Advice")
            tips = []
            if smoker == "Yes":
                # Calculate non-smoker savings
                input_nosmoke = input_df.copy()
                input_nosmoke['smoker'] = 'no'
                X_nosmoke = preprocessor.transform(input_nosmoke)
                charge_nosmoke = max(0.0, float(model.predict(X_nosmoke)[0]))
                savings = pred_charges - charge_nosmoke
                tips.append(f"🚭 **Smoking Cessation:** Transitioning to non-smoking can save approximately **${savings:,.2f} per year** (a {savings/pred_charges*100:.0f}% reduction)!")

            if bmi >= 30:
                # Calculate healthy BMI savings
                input_healthy_bmi = input_df.copy()
                input_healthy_bmi['bmi'] = 24.5
                X_hbmi = preprocessor.transform(input_healthy_bmi)
                charge_hbmi = max(0.0, float(model.predict(X_hbmi)[0]))
                bmi_savings = pred_charges - charge_hbmi
                if bmi_savings > 500:
                    tips.append(f"🥗 **BMI Optimization:** Lowering BMI to normal range (<25) could reduce annual charges by approximately **${bmi_savings:,.2f}**.")
            
            if not tips:
                tips.append("🌟 **Great Profile:** The applicant demonstrates low health claim risk. Maintaining this lifestyle ensures lock-in of the lowest premium tier.")

            for tip in tips:
                st.info(tip)

# -------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
with tab2:
    st.markdown("### 📊 Dataset Exploration & Statistical Insights")
    if not df_raw.empty:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Records", f"{len(df_raw):,}")
        col_m2.metric("Average Annual Charge", f"${df_raw['charges'].mean():,.2f}")
        col_m3.metric("Smoker Percentage", f"{(df_raw['smoker']=='yes').mean()*100:.1f}%")
        col_m4.metric("Average BMI", f"{df_raw['bmi'].mean():.2f}")

        eda_row1_col1, eda_row1_col2 = st.columns(2)
        with eda_row1_col1:
            fig_hist = px.histogram(
                df_raw, 
                x="charges", 
                nbins=40, 
                color="smoker", 
                title="<b>Distribution of Insurance Charges by Smoker Status</b>",
                color_discrete_map={"yes": "#ef4444", "no": "#10b981"},
                marginal="box",
                barmode="overlay",
                opacity=0.75
            )
            fig_hist.update_layout(template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)

        with eda_row1_col2:
            fig_scatter = px.scatter(
                df_raw, 
                x="bmi", 
                y="charges", 
                color="smoker", 
                size="age",
                hover_data=["children", "region"],
                title="<b>BMI vs. Charges (Highlighting Obese Smokers)</b>",
                color_discrete_map={"yes": "#ef4444", "no": "#3b82f6"}
            )
            fig_scatter.add_vline(x=30, line_dash="dash", line_color="gray", annotation_text="Obesity Line (BMI=30)")
            fig_scatter.update_layout(template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)

        eda_row2_col1, eda_row2_col2 = st.columns(2)
        with eda_row2_col1:
            fig_box = px.box(
                df_raw, 
                x="region", 
                y="charges", 
                color="region", 
                title="<b>Medical Charges Across US Regions</b>"
            )
            fig_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)

        with eda_row2_col2:
            corr_df = df_raw[['age', 'bmi', 'children', 'charges']].copy()
            corr_df['smoker_numeric'] = (df_raw['smoker'] == 'yes').astype(int)
            corr_mat = corr_df.corr()
            fig_corr = px.imshow(
                corr_mat, 
                text_auto=".3f", 
                color_continuous_scale="Blues",
                title="<b>Correlation Heatmap (Numeric & Encoded Factors)</b>"
            )
            fig_corr.update_layout(template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Raw dataset file not found at data/raw/insurance.csv.")

# -------------------------------------------------------------
# TAB 3: MODEL BENCHMARKS & EVALUATION
# -------------------------------------------------------------
with tab3:
    st.markdown("### 🏆 Machine Learning Model Evaluation & Comparison")
    if metrics:
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
        metrics_df.rename(columns={"index": "Algorithm"}, inplace=True)
        
        # Display Table
        st.dataframe(
            metrics_df.sort_values(by="Test_R2", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            fig_r2 = px.bar(
                metrics_df.sort_values(by="Test_R2"), 
                x="Test_R2", 
                y="Algorithm", 
                orientation="h",
                text="Test_R2",
                title="<b>Test R² Score Comparison (Higher is Better)</b>",
                color="Test_R2",
                color_continuous_scale="Blues"
            )
            fig_r2.update_layout(template="plotly_white", xaxis_range=[0, 1])
            st.plotly_chart(fig_r2, use_container_width=True)

        with b_col2:
            fig_rmse = px.bar(
                metrics_df.sort_values(by="Test_RMSE", ascending=False), 
                x="Test_RMSE", 
                y="Algorithm", 
                orientation="h",
                text="Test_RMSE",
                title="<b>Root Mean Squared Error (RMSE - Lower is Better)</b>",
                color="Test_RMSE",
                color_continuous_scale="Reds_r"
            )
            fig_rmse.update_layout(template="plotly_white")
            st.plotly_chart(fig_rmse, use_container_width=True)

        # Static Saved Visualizations
        st.markdown("#### 🔬 Diagnostic Plots")
        p_col1, p_col2 = st.columns(2)
        actual_vs_pred_img = "reports/figures/actual_vs_predicted.png"
        feat_imp_img = "reports/figures/feature_importance.png"
        
        if os.path.exists(actual_vs_pred_img):
            p_col1.image(actual_vs_pred_img, caption="Actual vs. Predicted Charges ($)", use_container_width=True)
        if os.path.exists(feat_imp_img):
            p_col2.image(feat_imp_img, caption="Champion Model Feature Importance", use_container_width=True)
    else:
        st.info("Metrics not yet generated. Run `python train.py` to train all algorithms and compute evaluation scores.")

# -------------------------------------------------------------
# TAB 4: BATCH PREDICTION (CSV)
# -------------------------------------------------------------
with tab4:
    st.markdown("### 📁 Batch Prediction for Insurance Portfolios")
    st.markdown("Upload a CSV file containing multiple policy applicants to generate batch price estimates and risk scores.")

    # Sample template download
    sample_df = pd.DataFrame([
        {"age": 19, "sex": "female", "bmi": 27.9, "children": 0, "smoker": "yes", "region": "southwest"},
        {"age": 45, "sex": "male", "bmi": 22.8, "children": 2, "smoker": "no", "region": "northwest"},
        {"age": 52, "sex": "female", "bmi": 32.1, "children": 1, "smoker": "no", "region": "southeast"},
        {"age": 31, "sex": "male", "bmi": 36.3, "children": 3, "smoker": "yes", "region": "northeast"}
    ])
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Sample Batch CSV Template",
        data=csv_buffer.getvalue(),
        file_name="sample_batch_insurance.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Applicants CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} records:")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🚀 Run Batch Underwriting Prediction", type="primary"):
                from src.predict import InsurancePredictor
                predictor = InsurancePredictor()
                result_df = predictor.predict_batch(batch_df)
                
                st.success("Batch Prediction Complete!")
                st.dataframe(result_df, use_container_width=True)
                
                # Download button
                res_buffer = io.StringIO()
                result_df.to_csv(res_buffer, index=False)
                st.download_button(
                    label="💾 Download Predictions CSV",
                    data=res_buffer.getvalue(),
                    file_name="predicted_insurance_charges.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# -------------------------------------------------------------
# TAB 5: PROJECT OVERVIEW & ARCHITECTURE
# -------------------------------------------------------------
with tab5:
    st.markdown("""
    ### 🏥 Medical Insurance Price Prediction - System Documentation
    
    #### 1. Problem Statement
    Health insurance pricing has traditionally relied on rigid actuarial tables that fail to dynamically capture non-linear compounding interactions (such as the combined exponential risk of high BMI and tobacco use). This project designs, evaluates, and deploys an end-to-end Machine Learning regression system capable of predicting individual medical charges with high precision and transparency.

    #### 2. System Architecture & Methodology
    1. **Data Ingestion & Quality Control:** Standardized Kaggle Medical Cost Personal dataset with 1,338 records across demographic and clinical variables.
    2. **Domain-Specific Feature Engineering:**
       - `is_obese`: Binary threshold ($BMI \ge 30.0$).
       - `obese_smoker`: Multiplicative interaction term ($is\_obese \times smoker$), which proves to be the single highest predictor of exponential medical expenditure.
    3. **Model Selection & Benchmarking:**
       - Linear, Ridge, and Lasso Regression
       - Support Vector Regression (SVR - RBF Kernel)
       - Decision Trees, Random Forest, AdaBoost, and Gradient Boosting
       - 5-Fold Cross-Validation and GridSearchCV Hyperparameter Optimization
    4. **Dual Interface Deployment:**
       - **Interactive Streamlit Web Dashboard** for real-time visual exploration and batch underwriting.
       - **FastAPI Production REST API** with automated OpenAPI documentation and validation schemas.

    #### 3. Evaluation Metrics Formulations
    - **Mean Absolute Error (MAE):** $MAE = \\frac{1}{n} \\sum_{i=1}^n |y_i - \\hat{y}_i|$
    - **Root Mean Squared Error (RMSE):** $RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (y_i - \\hat{y}_i)^2}$
    - **Coefficient of Determination ($R^2$):** $R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$
    """)
