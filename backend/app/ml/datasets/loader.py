import os
import hashlib
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

class DatasetLoader:
    """
    Authentic dataset loader and validator for Medical Cost Personal Dataset.
    Enforces data integrity, schema validation, and reproducible splitting.
    """
    EXPECTED_COLUMNS = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']

    def __init__(self, filepath: str = "backend/app/ml/datasets/insurance.csv"):
        self.filepath = filepath
        self.sha256_hash = None
        self.metadata = {}

    def compute_sha256(self) -> str:
        sha256 = hashlib.sha256()
        with open(self.filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        self.sha256_hash = sha256.hexdigest()
        return self.sha256_hash

    def load_and_validate(self) -> pd.DataFrame:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Authentic dataset not found at {self.filepath}")

        df = pd.read_csv(self.filepath)

        # Schema validation
        missing_cols = [c for c in self.EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required schema columns: {missing_cols}")

        self.compute_sha256()

        # Metadata analysis
        self.metadata = {
            "name": "Medical Cost Personal Dataset (Kaggle Benchmark)",
            "source": "US Census Bureau & National Center for Health Statistics via Kaggle/UCI",
            "sha256_hash": self.sha256_hash,
            "total_records": len(df),
            "total_features": len(df.columns) - 1,
            "missing_values_count": int(df.isnull().sum().sum()),
            "duplicates_count": int(df.duplicated().sum()),
            "limitations": (
                "Dataset captures historical billed insurance charges rather than forward-looking commercial premiums. "
                "Predictive values should be interpreted as expected annual healthcare utilization costs."
            )
        }

        # Deduplicate if exact duplicate rows exist
        df_clean = df.drop_duplicates().reset_index(drop=True)
        return df_clean

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.load_and_validate()
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df

if __name__ == "__main__":
    loader = DatasetLoader()
    df = loader.load_and_validate()
    print("Metadata:", loader.metadata)
    print("Shape:", df.shape)
