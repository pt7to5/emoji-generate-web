from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CORS_ORIGINS = [value.strip() for value in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if value.strip()]
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

FAL_KEY = os.getenv("FAL_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BAIDU_DETECT_API_KEY = os.getenv("BAIDU_DETECT_API_KEY", "")
BAIDU_DETECT_SECRET_KEY = os.getenv("BAIDU_DETECT_SECRET_KEY", "")
BAIDU_PROCESS_API_KEY = os.getenv("BAIDU_PROCESS_API_KEY", "")
BAIDU_PROCESS_SECRET_KEY = os.getenv("BAIDU_PROCESS_SECRET_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMOJI_GENERATION_MODEL = os.getenv("EMOJI_GENERATION_MODEL", "wan2.7-image-pro")
EMOJI_GENERATION_CANDIDATES = max(1, min(4, int(os.getenv("EMOJI_GENERATION_CANDIDATES", "1"))))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
