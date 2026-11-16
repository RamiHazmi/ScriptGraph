"""
Configuration for GraphRAG system.
All settings in one place.
"""

# ============================================================
# PDF CONFIGURATION
# ============================================================

PDF_PATH = "the_last_call.pdf"

# ============================================================
# NEO4J CONFIGURATION
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"  # Changez ceci si vous avez utilisé un autre mot de passe

# ============================================================
# LLM CONFIGURATION - OPENROUTER + GROQ MULTI-PROVIDER FALLBACK
# ============================================================

# OpenRouter API Key
OPENROUTER_API_KEY = ""

# Groq API Key (FREE, super fast)
GROQ_API_KEY = ""  # Get from https://console.groq.com

# Multi-provider fallback chain
# Format: (provider, model_name)
# Using OpenRouter GPT-4o-mini (you have credits) + Groq fallbacks
LLM_PROVIDERS = [
    ("openrouter", "openai/gpt-4o-mini"),        # Primary - Your OpenRouter credits
    ("groq", "llama-3.3-70b-versatile"),         # Fallback 1 - Groq FREE 70B
    ("groq", "llama-3.1-8b-instant"),            # Fallback 2 - Groq FREE 8B (fast)
]

# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

MAX_CHUNK_SIZE = 1000  # Characters per chunk
MIN_CHUNK_SIZE = 300   # Minimum chunk size

# ============================================================
# EMBEDDING & VECTOR STORE CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB persistent storage
CHROMA_PERSIST_DIRECTORY = "./chroma_db"  # Where vector database is stored

# ============================================================
# VALIDATION CONFIGURATION
# ============================================================

MAX_CONCURRENT_VALIDATIONS = 4  # Concurrent API calls for validation