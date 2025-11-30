"""
Global Configuration
Centralized configuration for API keys, models, paths, and other settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
MANUALS_DIR = DATA_DIR / "manuals"
CSV_DIR = DATA_DIR / "data"
VECTOR_INDEX_DIR = BASE_DIR / "vector_index"


# =============================================================================
# API CONFIGURATION
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt5.1-2025-11-13")


# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "google/embeddinggemma-300m"
TIKTOKEN_MODEL = "gpt-5-mini"


# =============================================================================
# RAG CONFIGURATION
# =============================================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
DEFAULT_TOP_K = 5


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_PATH = "automotive.db"


# =============================================================================
# TOKEN PRICING (per 1M tokens)
# =============================================================================

TOKEN_PRICING = {
    "gpt5.1-2025-11-13": {"input": 1.25, "output": 10.00},
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00},
    "default": {"input": 1.25, "output": 10.00},
}


# =============================================================================
# DATA RELOAD CONFIGURATION
# =============================================================================

RELOAD_DATA = os.getenv("RELOAD_DATA", "false").lower() == "true"


# =============================================================================
# SCRAPER CONFIGURATION
# =============================================================================

TOYOTA_PORTAL_URL = "https://customerportal.tweddle-aws.eu"
DEFAULT_LANGUAGE = "en"
DEFAULT_BRAND = "toyota"

SCRAPER_TIMEOUTS = {
    "element_wait": 20,
    "spinner_wait": 180,
    "iframe_wait": 30,
    "download_wait": 120,
}

MANUAL_MODELS = {
    "RAV4": [("RAV4", "2018 - Today"), ("RAV4 HEV", "2025 - Today")],
    "Corolla": [("Corolla", "2013 - 2018")],
    "Prius": [("Prius", "2015 - 2022")],
}


# =============================================================================
# CONTRACT FILES
# =============================================================================

CONTRACT_FILES = [
    "Contract_Toyota_2023.pdf",
    "Contract_Lexus_2023.pdf",
    "Warranty_Policy_Appendix.pdf",
]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that required configuration is present."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return True


def print_config_status():
    """Print configuration status for debugging."""
    print(f"✓ OpenAI API Key: {'FOUND' if OPENAI_API_KEY else 'NOT FOUND'}")
    print(f"✓ OpenAI Model: {OPENAI_MODEL}")
    print(f"✓ Embedding Model: {EMBEDDING_MODEL}")
    print(f"✓ Reload Data: {RELOAD_DATA}")
    print(f"✓ Base Directory: {BASE_DIR}")
