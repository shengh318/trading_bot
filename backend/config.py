from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"

DB_PATH = os.getenv("DB_PATH", str(Path(__file__).resolve().parent.parent / "trader.db"))

def validate_config() -> list[str]:
    errors = []
    if not ALPACA_API_KEY:
        errors.append("ALPACA_API_KEY is not set. Copy .env.example to .env and fill in your keys.")
    if not ALPACA_SECRET_KEY:
        errors.append("ALPACA_SECRET_KEY is not set.")
    return errors
