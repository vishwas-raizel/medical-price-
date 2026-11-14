import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    PROJECT_NAME: str = "Medical Insurance Price Prediction & Risk Intelligence Platform"
    PROJECT_VERSION: str = "2.0.0"
    API_V1_PREFIX: str = "/api"
    
    # Security & Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "prod-super-secret-key-medical-ai-2026-auth-jwt-token")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/insurance_platform.db")
    
    # Paths
    ML_DATASET_PATH: str = str(BASE_DIR / "app" / "ml" / "datasets" / "insurance.csv")
    SAVED_MODELS_DIR: str = str(BASE_DIR / "app" / "ml" / "saved_models")
    REPORTS_DIR: str = str(BASE_DIR.parent / "reports")
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "*"
    ]

settings = Settings()
