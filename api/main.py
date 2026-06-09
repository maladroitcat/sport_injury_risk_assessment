from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.inference.model_registry import DEFAULT_MODEL_VERSION, ModelRegistry


ALLOWED_EXTENSIONS = {".mp4", ".mov"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(250 * 1024 * 1024)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelRegistry(
        root_dir=os.getenv("MODEL_ARTIFACT_ROOT", "model_artifacts"),
        default_model_version=os.getenv("DEFAULT_MODEL_VERSION", DEFAULT_MODEL_VERSION),
        device=os.getenv("MODEL_DEVICE", "cpu"),
        batch_size=int(os.getenv("MODEL_BATCH_SIZE", "4")),
    )
    yield


app = FastAPI(
    title="Sports Injury Risk API",
    version="0.1.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


Sport = Literal["hockey", "basketball", "soccer", "football", "rugby"]
ImpactType = Literal["collision", "object_hit", "fall", "twist"]
BodyRegion = Literal["head_face", "upper_body", "lower_body"]


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "default_model_version": app.state.models.default_model_version,
    }


@app.get("/models")
def list_models() -> dict:
    return {
        "default_model_version": app.state.models.default_model_version,
        "models": app.state.models.list_models(),
        "loaded_model_versions": app.state.models.loaded_model_versions(),
    }


@app.post("/predict")
async def predict(
    video: Annotated[UploadFile, File()],
    sport: Annotated[Sport, Form()],
    impact_type: Annotated[ImpactType, Form()],
    body_region: Annotated[BodyRegion, Form()],
    model_version: Annotated[str, Form()] = DEFAULT_MODEL_VERSION,
) -> dict:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload a .mp4 or .mov video clip.")

    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        size = 0
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded video is too large.")
            tmp.write(chunk)
        tmp.flush()

        try:
            predictor = app.state.models.get(model_version)
            result = predictor.predict_video(tmp.name)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown model_version: {model_version}") from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not process video: {exc}") from exc

    return {
        **result,
        "input_metadata": {
            "sport": sport,
            "impact_type": impact_type,
            "body_region": body_region,
        },
    }
