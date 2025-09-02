# src/data_to_decisions/common/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env in the repo root
except Exception:
    pass

class Settings(BaseSettings):
    app_name: str = "applied-data-science-for-business"
    app_env: str = "dev"
    log_level: str = "INFO"
    mlflow_tracking_uri: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
