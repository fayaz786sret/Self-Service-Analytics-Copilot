import os
from dotenv import load_dotenv

load_dotenv()


def _read_secret(name: str, default: str = "") -> str:
    """Read a setting from the environment first, then Streamlit secrets if available."""
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return env_value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value in (None, ""):
            return default
        return str(secret_value)
    except Exception:
        return default

class Settings:
    """Application settings from environment variables."""
    
    # OpenAI Config
    OPENAI_API_KEY: str = _read_secret("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = _read_secret("OPENAI_MODEL", "gpt-4")
    
    # Groq Config
    GROQ_API_KEY: str = _read_secret("GROQ_API_KEY", "")
    GROQ_MODEL: str = _read_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # FastAPI Config
    API_TITLE: str = "Data Analysis Pipeline API"
    API_VERSION: str = "0.1.0"
    API_DESCRIPTION: str = "Advanced autonomous data analysis with LangChain & LangGraph"
    DEBUG: bool = _read_secret("DEBUG", "false").lower() == "true"
    API_HOST: str = _read_secret("API_HOST", "0.0.0.0")
    API_PORT: int = int(_read_secret("API_PORT", "8010"))
    
    # Database Config
    DATABASE_URL: str = _read_secret("DATABASE_URL", "sqlite:///./analysis.db")
    
    # Vector DB Config (Pinecone)
    PINECONE_API_KEY: str = _read_secret("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = _read_secret("PINECONE_INDEX", "analysis-patterns")
    
    # Data Config
    MAX_FILE_SIZE_MB: int = int(_read_secret("MAX_FILE_SIZE_MB", "1024"))
    ALLOWED_EXTENSIONS: list = ["csv", "json", "xlsx", "txt"]
    
    # Analysis Config
    ANALYSIS_TIMEOUT_SECONDS: int = 300
    CHUNK_SIZE: int = 10000
    OUTPUTS_DIR: str = _read_secret("OUTPUTS_DIR", "data/outputs")
    RUNS_DIR: str = _read_secret("RUNS_DIR", "data/runs")
    DATA_PIPELINE_API_URL: str = _read_secret("DATA_PIPELINE_API_URL", "http://localhost:8010")
    
    def validate(self):
        """Validate required settings."""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

settings = Settings()
