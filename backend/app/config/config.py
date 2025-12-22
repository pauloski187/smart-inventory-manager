from pydantic import BaseSettings

class Config(BaseSettings):
    database_url: str = "sqlite:///./inventory.db"
    secret_key: str = "your-secret-key-here-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Data loading settings
    csv_chunk_size: int = 1000
    dead_stock_days: int = 90  # Days without sales to consider dead stock

    class Config:
        env_file = ".env"

config = Config()