import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]

class AppConfig(BaseSettings):
    # File Paths
    TEMP_FILEPATH: str = 'tmp'
    DATA_FILEPATH: str = 'data'
    SESSION_FILEPATH: str = 'session'
    FIGURE_FILEPATH: str = 'figures'
    MODEL_FILEPATH: str = 'model'

    SESS_LOG_NAME: str = 'sess_log'
    SESS_LOG_FILENAME: str = 'sess.log'
    CENTRAL_LOG_DIR: str = 'logs'
    CENTRAL_LOG_FILENAME: str = 'central.log'
    CENTRAL_LOG_NAME: str = 'api_gateway'

    UUID_LEN: int = 8
    TASK_NODE_MAX_RETRIES: int = 3
    TASK_GRAPH_MAX_RETRIES: int = 3

    VISUAL_ALLOWED_EXTENSIONS: List[str] = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']

class LlmConfig(BaseSettings):
    LLM_MAX_RETRIES: int = 2
    OPENAI_MODEL: str = 'gpt-5-mini-2025-08-07'
    TIMEOUT: int | None = None
    CACHE: bool = False
    TEMPERATURE: float = 0.3
    MAX_COMPLETION_TOKENS: int | None = None
    OPENAI_API_KEY: str | None = None

class Config(LlmConfig, AppConfig, BaseSettings):
    ENV: str = "development"
    DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    WRITER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi"
    READER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi"
    JWT_SECRET_KEY: str = "fastapi"
    JWT_ALGORITHM: str = "HS256"
    SENTRY_SDN: str = ""
    CELERY_BROKER_URL: str = "amqp://user:bitnami@localhost:5672/"
    CELERY_BACKEND_URL: str = "redis://:password123@localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
    )


class TestConfig(Config):
    WRITER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi_test"
    READER_DB_URL: str = "mysql+aiomysql://fastapi:fastapi@localhost:3306/fastapi_test"


class LocalConfig(Config):
    ...


class ProductionConfig(Config):
    DEBUG: bool = False


def get_config():
    env = os.getenv("ENV", "local")
    config_type = {
        "test": TestConfig(),
        "local": LocalConfig(),
        "prod": ProductionConfig(),
    }
    return config_type[env]


config: Config = get_config()
