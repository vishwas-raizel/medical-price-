import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional

class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering for Medical Insurance dataset:
    1. bmi_category: Categorizes BMI into Underweight, Normal, Overweight, Obese.
    2. is_obese: Flag for BMI >= 30.
    3. smoker_obese_interaction: Interaction term between smoking status and obesity.
       (In medical underwriting, obese smokers face multiplicative actuarial risk).
    4. age_category: Young (18-35), Middle (36-50), Senior (51+).
    """
    def __init__(self, add_interaction_terms: bool = True):
        self.add_interaction_terms = add_interaction_terms

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # Ensure numerical types
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['children'] = pd.to_numeric(df['children'], errors='coerce')
        
        # 1. Obesity indicator
        df['is_obese'] = (df['bmi'] >= 30.0).astype(int)
        
        # 2. Smoker numeric flag
        smoker_binary = (df['smoker'].astype(str).str.lower() == 'yes').astype(int)
        
        # 3. Domain Interaction: Obese Smoker (critical actuarial factor)
        if self.add_interaction_terms:
            df['obese_smoker'] = df['is_obese'] * smoker_binary
            
        return df

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Builds the ColumnTransformer for numerical scaling and categorical encoding.
    """
    numerical_features = ['age', 'bmi', 'children', 'is_obese', 'obese_smoker']
    categorical_features = ['sex', 'smoker', 'region']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

class InsurancePreprocessor:
    """
    High-level orchestrator for full preprocessing pipeline.
    """
    def __init__(self):
        self.feature_engineer = MedicalFeatureEngineer()
        self.column_transformer = build_preprocessing_pipeline()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        X_engineered = self.feature_engineer.transform(X)
        self.column_transformer.fit(X_engineered)
        
        # Extract feature names after transformation
        cat_encoder = self.column_transformer.named_transformers_['cat']['onehot']
        cat_features = ['sex', 'smoker', 'region']
        cat_encoded_names = cat_encoder.get_feature_names_out(cat_features).tolist()
        num_features = ['age', 'bmi', 'children', 'is_obese', 'obese_smoker']
        self.feature_names = num_features + cat_encoded_names
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
        X_engineered = self.feature_engineer.transform(X)
        return self.column_transformer.transform(X_engineered)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/preprocessor.pkl") -> "InsurancePreprocessor":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
        return joblib.load(filepath)

if __name__ == "__main__":
    from src.data_loader import load_data
    raw_df = load_data()
    X = raw_df.drop(columns=['charges'])
    y = raw_df['charges']
    
    preprocessor = InsurancePreprocessor()
    X_proc = preprocessor.fit_transform(X)
    print("Preprocessed shape:", X_proc.shape)
    print("Features extracted:", preprocessor.feature_names)
    preprocessor.save()
