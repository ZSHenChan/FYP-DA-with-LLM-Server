import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]

class PathSettings(BaseSettings):
    """File system paths and storage configurations."""
    TEMP_FILEPATH: str = Field('tmp', description="Temporary storage")
    DATA_FILEPATH: str = Field('data', description="Shared session data")
    SESSION_FILEPATH: str = Field('session', description="Session artifacts")
    FIGURE_FILEPATH: str = Field('figures', description="Generated images")
    MODEL_FILEPATH: str = Field('model', description="Saved models")
    
    VISUAL_ALLOWED_EXTENSIONS: List[str] = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']

class LogSettings(BaseSettings):
    """Logging configurations."""
    SESS_LOG_NAME: str = 'sess_log'
    SESS_LOG_FILENAME: str = 'sess.log'
    CENTRAL_LOG_DIR: str = 'logs'
    CENTRAL_LOG_FILENAME: str = 'central.log'
    CENTRAL_LOG_NAME: str = 'api_gateway'

class ServerSettings(BaseSettings):
    """FastAPI Server and Security settings."""
    ENV: str = "development"
    DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    
    UUID_LEN: int = 8

    # Security
    JWT_SECRET_KEY: str = Field("fastapi8888", min_length=8)
    JWT_ALGORITHM: str = "HS256"
    
    # Timeouts & Intervals (New)
    API_TIMEOUT: int = Field(30, description="General request timeout in seconds")
    KEEPALIVE_INTERVAL: int = Field(5, description="Ping interval in seconds for websockets/streaming")

class DatabaseSettings(BaseSettings):
    """Database, Redis, and Celery connections."""
    # Using specific Pydantic types (MySQLDsn, etc.) provides automatic validation
    WRITER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi"
    READER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi"
    
    # DB Performance (New)
    DB_POOL_SIZE: int = Field(5, description="SQLAlchemy pool size")
    DB_POOL_RECYCLE: int = Field(3600, description="Connection recycle time in seconds")
    DB_CONNECT_TIMEOUT: int = Field(10, description="Database connection timeout")

    # Redis (New)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SOCKET_TIMEOUT: int = 5
    
    # Celery
    CELERY_BROKER_URL: str = "amqp://user:bitnami@localhost:5672/"
    CELERY_BACKEND_URL: str = "redis://:password123@localhost:6379/0"

class LlmConfig(BaseSettings):
    LLM_MAX_RETRIES: int = 2
    OPENAI_MODEL: str = 'gpt-5-mini-2025-08-07'
    TIMEOUT: int | None = None
    CACHE: bool = False
    TEMPERATURE: float = 0.3
    MAX_COMPLETION_TOKENS: int | None = None
    OPENAI_API_KEY: str | None = None

class AgentSettings(BaseSettings):
    """Workflow and Agent logic configuration."""
    TASK_NODE_MAX_RETRIES: int = 3
    TASK_GRAPH_MAX_RETRIES: int = 3
    
    # Sentry / Monitoring
    SENTRY_SDN: Optional[str] = None

class BaseConfig(
    ServerSettings, 
    DatabaseSettings, 
    PathSettings, 
    LogSettings, 
    AgentSettings, 
    LlmConfig, 
    BaseSettings
):
    """
    Master Config class inheriting from all Mixins.
    """
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore" # Good practice: ignore unknown env vars to prevent crashes
    )


class TestConfig(BaseConfig):
    WRITER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi_test"
    READER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi_test"


class LocalConfig(BaseConfig):
    ...


class ProductionConfig(BaseConfig):
    DEBUG: bool = False


def get_config() -> BaseConfig:
    env = os.getenv("ENV", "local")
    config_type = {
        "test": TestConfig(),
        "local": LocalConfig(),
        "prod": ProductionConfig(),
    }
    return config_type[env]


config = get_config()
