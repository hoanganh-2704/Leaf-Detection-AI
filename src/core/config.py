import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # Model configs
    VIT_MODEL_NAME: str = "prithivMLmods/Rice-Leaf-Disease"
    VECTOR_DB_PATH: str = "./data/processed/chroma_db"
    
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()
