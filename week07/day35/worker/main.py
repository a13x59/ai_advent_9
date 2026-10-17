"""Тонкая обёртка над Basic Pitch (Spotify) для транскрипции аудио в MIDI.

Принимает сырые байты WAV-файла в теле POST-запроса и возвращает MIDI-байты.
Модель загружается один раз при старте процесса.

Basic Pitch используется «как есть» — ничего не переписано. Веса модели входят
в состав пакета basic-pitch (загрузка на старте не требуется).
"""

import io
import logging
import os
import tempfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("worker")

app = FastAPI(title="basic-pitch-worker")

# Модель загружается один раз при старте процесса.
logger.info("Loading Basic Pitch model from %s ...", ICASSP_2022_MODEL_PATH)
MODEL = Model(ICASSP_2022_MODEL_PATH)
logger.info("Basic Pitch model loaded.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict:
    return {"service": "basic-pitch-worker", "endpoints": ["/health", "/predict"]}


@app.post("/predict")
async def predict_endpoint(request: Request) -> Response:
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty request body")

    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)

        logger.info("Transcribing %d bytes ...", len(data))
        model_output, midi_data, note_events = predict(path, model_or_model_path=MODEL)

        buf = io.BytesIO()
        midi_data.write(buf)
        logger.info("Transcribed %d note(s).", len(note_events))
        return Response(content=buf.getvalue(), media_type="audio/midi")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
