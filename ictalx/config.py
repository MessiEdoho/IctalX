"""Settings, read from environment variables so the same code runs locally, on AWS and at UCD."""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path        # folder with the weights and best_multiscale_params.json
    upload_dir: Path       # temporary uploads, deleted after each analysis
    samples_dir: Path      # sample recordings offered for download (may be empty)
    web_dir: Path          # the interface (index.html and assets)
    device: str            # "cpu" or "cuda"
    workers: int           # detections run at the same time; the rest wait in the queue
    batch_size: int        # segments per forward pass; does not change results
    session_hours: float   # idle time before an analysis is dropped


def load_settings():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.get
    return Settings(
        model_dir=Path(env("ICTALX_MODEL_DIR", root / "models")),
        upload_dir=Path(env("ICTALX_UPLOAD_DIR", Path(tempfile.gettempdir()) / "ictalx_uploads")),
        samples_dir=Path(env("ICTALX_SAMPLES_DIR", root / "samples")),
        web_dir=Path(env("ICTALX_WEB_DIR", root / "web")),
        device=env("ICTALX_DEVICE", "cpu"),
        workers=int(env("ICTALX_WORKERS", "1")),
        batch_size=int(env("ICTALX_BATCH_SIZE", "256")),
        session_hours=float(env("ICTALX_SESSION_HOURS", "3")),
    )
