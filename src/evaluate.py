import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data
from src.preprocessor import InsurancePreprocessor

# Set clean aesthetic style for academic & college reports
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_eda_figures(df: pd.DataFrame):
    """Generates and saves exploratory data analysis charts."""
    print("Generating EDA visualizations...")

    # 1. Distribution of Insurance Charges
    plt.figure(figsize=(9, 5))
    sns.histplot(df['charges'], kde=True, color='#2563eb', bins=35)
    plt.title("Distribution of Medical Insurance Charges (Right-skewed)", fontsize=14, fontweight='bold')
    plt.xlabel("Insurance Charges ($)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_charges_distribution.png"), dpi=300)
    plt.close()

    # 2. Charges by Smoking Status & BMI
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='bmi', 
        y='charges', 
        hue='smoker', 
        palette={'yes': '#dc2626', 'no': '#16a34a'},
        alpha=0.8,
        s=60
    )
    plt.axvline(30, color='gray', linestyle='--', label='Obesity Threshold (BMI=30)')
    plt.title("Impact of BMI & Smoking Status on Insurance Charges", fontsize=14, fontweight='bold')
    plt.xlabel("Body Mass Index (BMI)", fontsize=12)
    plt.ylabel("Insurance Charges ($)", fontsize=12)
    plt.legend(title="Smoker", loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_bmi_vs_charges_by_smoker.png"), dpi=300)
    plt.close()

    # 3. Age vs Charges by Smoker
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='age', 
        y='charges', 
        hue='smoker', 
        palette={'yes': '#dc2626', 'no': '#2563eb'},
        alpha=0.8,
        s=60
    )
    plt.title("Age vs Insurance Charges by Smoking Status", fontsize=14, fontweight='bold')
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Insurance Charges ($)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_age_vs_charges.png"), dpi=300)
    plt.close()

    # 4. Correlation Heatmap
    num_df = df[['age', 'bmi', 'children', 'charges']].copy()
    num_df['is_smoker'] = (df['smoker'] == 'yes').astype(int)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap='Blues', fmt=".3f", linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_correlation_heatmap.png"), dpi=300)
    plt.close()
    
    print(f"EDA figures saved to {FIGURES_DIR}")

def generate_model_evaluation_figures():
    """Generates model performance comparison and champion diagnostics."""
    print("Generating Model Evaluation figures...")
    
    # 1. Load Model Comparison CSV
    csv_path = "reports/model_comparison.csv"
    if os.path.exists(csv_path):
        metrics_df = pd.read_csv(csv_path)
        
        # Sort by R2
        metrics_df_sorted = metrics_df.sort_values(by="Test_R2", ascending=True)
        
        # Bar Chart of R2 Scores
        plt.figure(figsize=(10, 6))
        colors = ['#93c5fd' if 'Tuned' not in m else '#2563eb' for m in metrics_df_sorted['Model']]
        bars = plt.barh(metrics_df_sorted['Model'], metrics_df_sorted['Test_R2'], color=colors)
        plt.title("Model Comparison: Test R² Score", fontsize=14, fontweight='bold')
        plt.xlabel("R² Score (Higher is Better)", fontsize=12)
        plt.xlim(0, 1.0)
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "model_r2_comparison.png"), dpi=300)
        plt.close()

        # Bar Chart of RMSE
        metrics_rmse_sorted = metrics_df.sort_values(by="Test_RMSE", ascending=False)
        plt.figure(figsize=(10, 6))
        colors_rmse = ['#fca5a5' if 'Tuned' not in m else '#dc2626' for m in metrics_rmse_sorted['Model']]
        bars_rmse = plt.barh(metrics_rmse_sorted['Model'], metrics_rmse_sorted['Test_RMSE'], color=colors_rmse)
        plt.title("Model Comparison: Root Mean Squared Error (RMSE)", fontsize=14, fontweight='bold')
        plt.xlabel("RMSE ($ - Lower is Better)", fontsize=12)
        for bar in bars_rmse:
            w = bar.get_width()
            plt.text(w + 100, bar.get_y() + bar.get_height()/2, f"${w:,.0f}", va='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "model_rmse_comparison.png"), dpi=300)
        plt.close()

    # 2. Champion Model Predictions & Residuals
    model_path = "models/best_model.pkl"
    preprocessor_path = "models/preprocessor.pkl"
    test_path = "data/processed/test.csv"
    
    if os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(test_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        test_df = pd.read_csv(test_path)
        
        X_test = preprocessor.transform(test_df.drop(columns=['charges']))
        y_test = test_df['charges'].values
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        # Actual vs Predicted Scatter
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, color='#2563eb', edgecolors='k', linewidths=0.5)
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label="Ideal Prediction (y = x)")
        plt.title("Actual vs. Predicted Medical Insurance Charges", fontsize=14, fontweight='bold')
        plt.xlabel("Actual Charges ($)", fontsize=12)
        plt.ylabel("Predicted Charges ($)", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=300)
        plt.close()

        # Residuals Distribution
        plt.figure(figsize=(9, 5))
        sns.histplot(residuals, kde=True, color='#059669', bins=30)
        plt.axvline(0, color='red', linestyle='--', lw=1.5)
        plt.title("Residuals Distribution (Actual - Predicted)", fontsize=14, fontweight='bold')
        plt.xlabel("Residual ($)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "residuals_distribution.png"), dpi=300)
        plt.close()

        # Feature Importance (if tree-based)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = preprocessor.feature_names
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=True)

            plt.figure(figsize=(10, 6))
            plt.barh(feat_df['Feature'], feat_df['Importance'], color='#3b82f6')
            plt.title("Champion Model: Feature Importance (Gini)", fontsize=14, fontweight='bold')
            plt.xlabel("Relative Importance Score", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=300)
            plt.close()

    print(f"All evaluation plots successfully saved to {FIGURES_DIR}")

if __name__ == "__main__":
    raw_df = load_data()
    generate_eda_figures(raw_df)
    generate_model_evaluation_figures()
