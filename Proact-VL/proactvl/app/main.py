import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from proactvl.app.core.config import get_settings
from proactvl.app.api.ws import router as ws_router
from proactvl.app.api.hello import router as hello_router
from proactvl.app.models.model import get_model

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup (optional warmup)
    model = get_model()
    # If warmup() exists, run in a background thread; otherwise skip
    try:
        await run_in_threadpool(model.warmup)
    except AttributeError:
        pass
    except Exception as e:
        logging.exception("Model warmup failed: %s", e)
    yield
    # Resource cleanup can be placed here (e.g., close sessions/thread pools)

app = FastAPI(title="ProactVL Backend", version="0.1.0", lifespan=lifespan)

# CORS only applies to HTTP; WS should validate Origin separately (see ws.py)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    # Simple readiness indicator
    try:
        get_model()  # Raises if load failed
        return {"ok": True, "model_loaded": True}
    except Exception:
        return {"ok": True, "model_loaded": False}

# WebSocket route: /ws/stream
app.include_router(ws_router)
app.include_router(hello_router)

# Serve static assets and index.html
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if INDEX_FILE.exists():
    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_FILE.read_text(encoding="utf-8")