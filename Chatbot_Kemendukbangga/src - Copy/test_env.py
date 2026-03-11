import os, pathlib
from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

print("cwd:", pathlib.Path.cwd())
print("ENV_PATH:", ENV_PATH)

load_dotenv(ENV_PATH)

print("GEMINI_API_KEY loaded?", bool(os.getenv("GEMINI_API_KEY")))
print("Preview:", (os.getenv("GEMINI_API_KEY") or "")[:8])
