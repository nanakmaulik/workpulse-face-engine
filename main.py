"""Private CPU face embedding service. No face images are persisted.
Model weights are supplied separately. Validate commercial rights before usage.
"""
import base64
import hmac
import os
from contextlib import asynccontextmanager
from typing import List
import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.getenv('MODEL_DIR', './models')
TOKEN = os.getenv('FACE_SERVICE_TOKEN', '')
if len(TOKEN) < 24:
    raise RuntimeError('Set FACE_SERVICE_TOKEN to a random secret of at least 24 characters')

def model_path(name: str) -> str:
    path = os.path.join(MODEL_DIR, name)
    if not os.path.isfile(path):
        raise RuntimeError(f'Missing {path}. See README for model download and licensing.')
    return path

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = cv2.FaceDetectorYN.create(model_path('face_detection_yunet_2023mar.onnx'), '', (640, 640), score_threshold=0.9, nms_threshold=0.3, top_k=5000)
    app.state.recognizer = cv2.FaceRecognizerSF.create(model_path('face_recognition_sface_2021dec.onnx'), '')
    yield

app = FastAPI(title='WorkPulse Private Face Engine', docs_url=None, redoc_url=None, lifespan=lifespan)

class ImageRequest(BaseModel):
    image: str = Field(max_length=7_000_000)

class Candidate(BaseModel):
    employee_id: str
    embedding: List[float] = Field(min_length=128, max_length=128)

class MatchRequest(ImageRequest):
    templates: List[Candidate] = Field(max_length=5000)

def authorize(authorization: str | None):
    actual = (authorization or '').removeprefix('Bearer ')
    if not hmac.compare_digest(actual, TOKEN):
        raise HTTPException(401, 'Unauthorized')

def extract(image: str):
    if not (image.startswith('data:image/jpeg;base64,') or image.startswith('data:image/png;base64,')):
        raise HTTPException(400, 'JPEG or PNG data URL required')
    try:
        b = base64.b64decode(image.split(',', 1)[1], validate=True)
        if len(b) > 5_000_000: raise ValueError('Image too large')
        frame = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(400, 'Invalid image encoding')
    if frame is None or min(frame.shape[:2]) < 120 or max(frame.shape[:2]) > 2400:
        raise HTTPException(422, 'Invalid frame size')
    h, w = frame.shape[:2]
    app.state.detector.setInputSize((w, h))
    _, faces = app.state.detector.detect(frame)
    if faces is None or len(faces) != 1:
        raise HTTPException(422, 'Exactly one face must be visible')
    face = faces[0]
    if min(face[2], face[3]) < 90:
        raise HTTPException(422, 'Move closer to the camera')
    aligned = app.state.recognizer.alignCrop(frame, face)
    vec = app.state.recognizer.feature(aligned).flatten().astype(np.float32)
    vec /= max(float(np.linalg.norm(vec)), 1e-12)
    if vec.shape[0] != 128: raise HTTPException(500, 'Unexpected model embedding dimension')
    return vec

@app.get('/health')
def health():
    return {'ok': True, 'model': 'opencv-sface-v1'}

@app.post('/embed')
def embed(request: ImageRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    vec = extract(request.image)
    return {'embedding': vec.astype(float).tolist(), 'model': 'opencv-sface-v1'}

@app.post('/match')
def match(request: MatchRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    if not request.templates: raise HTTPException(422, 'No enrolled employees found')
    vec = extract(request.image)
    best: dict[str, float] = {}
    for item in request.templates:
        other = np.array(item.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(other))
        if norm <= 1e-8: continue
        similarity = float(np.dot(vec, other / norm))
        best[item.employee_id] = max(best.get(item.employee_id, -1.0), similarity)
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    if not ranked: raise HTTPException(422, 'No valid templates')
    # SFace's published cosine example uses ~0.363 as a verification threshold.
    # This deliberately conservative starting threshold MUST be calibrated with consented local trials.
    minimum = float(os.getenv('FACE_COSINE_THRESHOLD', '0.48'))
    margin = float(os.getenv('FACE_MATCH_MARGIN', '0.08'))
    top_id, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
    if top_score < minimum or top_score - runner_up < margin:
        raise HTTPException(422, 'Face not confidently recognized; ask the manager')
    return {'employee_id': top_id, 'similarity': round(top_score, 4), 'model': 'opencv-sface-v1'}
