"""
Configuration module for model settings and output control (Colab version).

Adapted for Google Colab environment.
"""

import os


class ModelConfig:
    """Centralized model configuration and output control."""

    # --- Model Selection ---
    LLM_MODEL = "qwen2.5-vl-72b-instruct"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Output Control ---
    TEMPERATURE = 0.3
    MAX_TOKENS = 1024
    STREAMING = False

    # --- Document Processing ---
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    # --- Retrieval ---
    TOP_K_RESULTS = 5
    SEARCH_TYPE = "mmr"

    # --- Memory ---
    MEMORY_KEY = "chat_history"
    MAX_HISTORY_LENGTH = 10

    # --- Paths (Colab-friendly) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
    VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

    @classmethod
    def get_api_key(cls) -> str:
        """Retrieve the OpenAI API key from environment or Colab userdata."""
        key = os.getenv("OPENAI_API_KEY")
        if not key or key == "your-api-key-here":
            try:
                from google.colab import userdata
                key = userdata.get("OPENAI_API_KEY")
            except (ImportError, Exception):
                pass
        if not key or key == "your-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY not set. Use os.environ['OPENAI_API_KEY'] = 'your-key' "
                "or set it in Colab Secrets."
            )
        return key
