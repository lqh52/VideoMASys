from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "VideoMASys"
    database_url: str = "postgresql+asyncpg://videomasys:videomasys@localhost:5432/videomasys"
    artifact_root: Path = Path("./artifacts")
    auto_create_schema: bool = True
    max_upload_bytes: int = 20 * 1024 * 1024 * 1024

    vlm_base_url: str = "http://localhost:8001/v1"
    vlm_api_key: str = "local"
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    vlm_timeout_seconds: float = 120.0
    vlm_max_concurrency: int = 4
    vlm_enable_thinking: bool = False

    embedding_provider: str = "openai"
    embedding_base_url: str = "http://localhost:8002/v1"
    embedding_api_key: str = "local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int = 1024

    scene_threshold: float = Field(default=0.32, ge=0.0, le=1.0)
    min_scene_seconds: float = Field(default=2.0, gt=0.0)
    max_scene_seconds: float = Field(default=20.0, gt=0.0)
    max_scenes_per_video: int = Field(default=300, ge=1)
    max_scenes_per_event: int = Field(default=8, ge=1, le=32)
    max_event_seconds: float = Field(default=120.0, gt=0.0)
    frames_per_scene: int = Field(default=3, ge=1, le=8)
    media_max_concurrency: int = Field(default=4, ge=1, le=16)
    qa_retrieval_limit: int = Field(default=6, ge=1, le=20)
    qa_frames_per_interval: int = Field(default=8, ge=3, le=16)
    qa_use_video_clip: bool = True
    qa_clip_fps: float = Field(default=4.0, ge=1.0, le=12.0)
    worker_poll_seconds: float = Field(default=1.0, ge=0.1)

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def ensure_directories(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        (self.artifact_root / "uploads").mkdir(exist_ok=True)
        (self.artifact_root / "cache").mkdir(exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
