import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

EXPECTED_COLUMNS = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']

def load_data(filepath: str = "data/raw/insurance.csv") -> pd.DataFrame:
    """Load the medical insurance dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Check schema
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")
        
    return df

def validate_and_summarize(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect and summarize dataset characteristics."""
    summary = {
        "total_records": len(df),
        "total_features": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    return summary

def split_and_save_data(df: pd.DataFrame, 
                        output_dir: str = "data/processed", 
                        test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test sets and persist to processed data directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean duplicates if any (keep first)
    df_clean = df.drop_duplicates()
    
    train_df, test_df = train_test_split(df_clean, test_size=test_size, random_state=random_state)
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Data successfully split:")
    print(f" - Train set: {train_df.shape[0]} rows -> {train_path}")
    print(f" - Test set:  {test_df.shape[0]} rows -> {test_path}")
    
    return train_df, test_df

if __name__ == "__main__":
    raw_path = "data/raw/insurance.csv"
    data = load_data(raw_path)
    summary = validate_and_summarize(data)
    print("Dataset Summary:", summary)
    split_and_save_data(data)
