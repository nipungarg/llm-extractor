from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Reads from environment / the .env file, validates types automatically.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    gemini_api_key: str = ""
    default_provider: str = "openai"  # switch the whole app's provider here
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://us.cloud.langfuse.com"
    app_api_key: str = ""


settings = Settings()
