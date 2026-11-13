from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SEED_PATH = PROJECT_DIR / "data" / "seed.sql"
RUNTIME_DIR = PROJECT_DIR / ".runtime"
CATALOG_DB_PATH = RUNTIME_DIR / "catalogo.db"
TRACES_DIR = PROJECT_DIR / "traces"

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODELS = {
    "anthropic": {"supervisor": "claude-sonnet-5-5", "worker": "claude-haiku-5-5"},
    "openai": {"supervisor": "gpt-4o", "worker": "gpt-4o-mini"},
    "ollama": {"supervisor": "llama3.2", "worker": "llama3.2"},
}
API_KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
SDK_RETRIES_DISABLED = 0
REQUEST_TIMEOUT_S = 60
MAX_TOKENS = 1024

MAX_SUPERVISOR_STEPS = 8
MAX_RUNS_PER_SPECIALIST = 3
GRAPH_RECURSION_LIMIT = 40
SPECIALIST_RECURSION_LIMIT = 12
MIN_STEPS_TO_CALL_TOOLS = 2
MAX_MODEL_ATTEMPTS = 3
BACKOFF_PARAMS = {"initial": 1.0, "max": 10.0, "jitter": 1.0}

MAX_PRODUCT_MATCHES = 5
MAX_SUMMARY_CHARS = 600
POSITIVE_RATING_THRESHOLD = 4
DECIMALS = 2
