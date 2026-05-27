from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

class Settings(BaseSettings):
    main_provider: str
    main_provider_api_key: str
    main_provider_base_url: str
    main_provider_model: str

    # поля со значениями по умолчанию (можно также задать через Field)
    log_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / 'logs')
    database_url: str = 'sqlite:///data/knowledge_base.db'

    # Настройки Pydantic: читать из .env, использовать слоты (опционально)
    model_config = SettingsConfigDict(
        env_file='.env',          # загружает .env автоматически
        env_file_encoding='utf-8',
        extra='ignore',           # игнорировать лишние переменные
        slots=True,               # включает __slots__ (Pydantic >= 2.5)
    )

# значения из .env подтянутся автоматически
@lru_cache()
def get_settings() -> Settings:
    load_dotenv()
    return Settings()