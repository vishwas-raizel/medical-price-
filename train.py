"""
End-to-End Model Training, Hyperparameter Tuning & Evaluation Script
College Main Project: Medical Insurance Price Prediction using Machine Learning
"""
import sys
import os

# Add current workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_training import run_pipeline
from src.evaluate import generate_eda_figures, generate_model_evaluation_figures
from src.data_loader import load_data

def main():
    print("=" * 80)
    print("🏥 MEDICAL INSURANCE PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Run model training pipeline
    run_pipeline()
    
    # 2. Generate EDA & Evaluation Figures
    print("\n[Figures] Generating visual analysis charts for project report...")
    raw_df = load_data("data/raw/insurance.csv")
    generate_eda_figures(raw_df)
    generate_model_evaluation_figures()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print(" - Best Model Saved:    models/best_model.pkl")
    print(" - Preprocessor Saved:  models/preprocessor.pkl")
    print(" - Metrics Report:      reports/model_comparison.csv")
    print(" - Figures Generated:   reports/figures/")
    print("=" * 80)

if __name__ == "__main__":
    main()
