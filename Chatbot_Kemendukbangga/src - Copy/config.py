# src/config.py
import os
from dotenv import load_dotenv

# Tentukan BASE_DIR sebagai root project (satu level di atas folder src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path pasti ke file .env di root project
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# Paths & data
DATA_PATH = os.path.join(BASE_DIR, "data", "Data_Training_bkkbn.csv")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))

# FAISS files
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
EMBEDDINGS_NPY = os.path.join(BASE_DIR, "embeddings.npy")
METADATA_JSON = os.path.join(BASE_DIR, "metadata.json")

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))

# Gemini / Google Generative AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Default ke model Gemini modern; ganti di .env bila perlu
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Voice / Whisper
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Optional: debug helper (set via .env DEBUG=true)
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

if DEBUG:
    print("DEBUG config.py:")
    print("  BASE_DIR =", BASE_DIR)
    print("  ENV_PATH =", ENV_PATH)
    print("  GEMINI_API_KEY set?", bool(GEMINI_API_KEY))
    print("  GEMINI_MODEL =", GEMINI_MODEL)
    print("  DATA_PATH =", DATA_PATH)
