# src/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AuthConfig:
    """Authentication configuration."""
    secret_key: str = "your-secret-key"
    google_oauth_enabled: bool = True
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    session_timeout: int = 3600  # 1 hour

@dataclass
class LLMConfig:
    """LLM configuration."""
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30

@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    model_name: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 100

@dataclass
class StorageConfig:
    """Storage configuration."""
    data_dir: Path = Path("data")
    vector_db_path: Path = Path("data/vector_db")
    upload_dir: Path = Path("data/uploads")
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = [
                '.pdf', '.docx', '.txt', '.csv', 
                '.png', '.jpg', '.jpeg'
            ]
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.vector_db_path.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)

@dataclass
class Config:
    """Main application configuration."""
    auth: AuthConfig
    llm: LLMConfig
    embeddings: EmbeddingConfig
    storage: StorageConfig
    debug: bool = False

def load_config() -> Config:
    """Load configuration from environment variables and defaults."""
    
    # Auth config
    auth_config = AuthConfig(
        secret_key=os.getenv("SECRET_KEY", "datamap-ai-secret-key"),
        google_client_id=os.getenv("GOOGLE_CLIENT_ID"),
        google_client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        google_oauth_enabled=bool(os.getenv("GOOGLE_OAUTH_ENABLED", "true").lower() == "true")
    )
    
    # LLM config
    llm_config = LLMConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000"))
    )
    
    # Embedding config
    embedding_config = EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    
    # Storage config
    storage_config = StorageConfig(
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        max_file_size=int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))
    )
    
    return Config(
        auth=auth_config,
        llm=llm_config,
        embeddings=embedding_config,
        storage=storage_config,
        debug=bool(os.getenv("DEBUG", "false").lower() == "true")
    )
