"""
FastAPI application for real-time emotion recognition API.
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.utils.logger import setup_logger, get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

# Global model storage
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting up SER API...")
    
    # Load models here
    # app_state["model"] = load_models()
    
    yield
    
    # Shutdown
    logger.info("Shutting down SER API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Speech Emotion Recognition API",
        description="Real-time multimodal emotion recognition system",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Speech Emotion Recognition API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


async def start_api(config: dict):
    """Start the API server."""
    setup_logger("SER_API")
    
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8000)
    
    logger.info(f"Starting API server on {host}:{port}")
    
    config_uvicorn = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        reload=False
    )
    
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


if __name__ == "__main__":
    config = load_config("configs/model_config.yaml")
    asyncio.run(start_api(config))
