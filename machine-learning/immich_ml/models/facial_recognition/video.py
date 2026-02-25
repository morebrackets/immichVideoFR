from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import orjson
from numpy.typing import NDArray

from immich_ml.config import log
from immich_ml.schemas import DetectedFace, FacialRecognitionOutput

from .detection import FaceDetector
from .recognition import FaceRecognizer

# Maximum height (in pixels) before a frame is downscaled.
_MAX_HEIGHT = 720
# Maximum width for 720p at 16:9 aspect ratio.
_MAX_WIDTH = 1280
# Seconds between sampled frames.
_FRAME_INTERVAL_SECONDS = 3.0
# Cosine-similarity threshold above which two detections are considered the same person.
_DUPLICATE_THRESHOLD = 0.4


def _sample_frames(video_path: str | Path) -> Generator[NDArray[np.uint8], None, None]:
    """Yield one frame every ``_FRAME_INTERVAL_SECONDS`` seconds from *video_path*.

    Uses timestamp-based seeking (``CAP_PROP_POS_MSEC``) so that intermediate
    frames between sample points are never decoded, keeping CPU usage low even
    for high-frame-rate or very long videos.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    try:
        interval_ms = _FRAME_INTERVAL_SECONDS * 1000.0
        timestamp_ms = 0.0
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            timestamp_ms += interval_ms
    finally:
        cap.release()


def _resize_if_needed(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return *frame* downscaled to at most 720p, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h <= _MAX_HEIGHT and w <= _MAX_WIDTH:
        return frame
    # Scale so that neither dimension exceeds its limit.
    scale = min(_MAX_HEIGHT / h, _MAX_WIDTH / w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity in [-1, 1]."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _parse_embedding(embedding_str: str) -> NDArray[np.float32]:
    return np.array(orjson.loads(embedding_str), dtype=np.float32)


def deep_video_face_scan(
    video_path: str | Path,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    duplicate_threshold: float = _DUPLICATE_THRESHOLD,
) -> list[DetectedFace]:
    """Scan a video for faces, deduplicating across frames.

    Frames are sampled every :data:`_FRAME_INTERVAL_SECONDS` seconds (or every
    keyframe when that is cheaper — cv2 automatically seeks to the nearest
    decodable frame, so time-based stepping is at least as efficient as manual
    I-frame extraction in pure Python).

    For each sampled frame the function:

    1. Downscales the frame to 720p if necessary.
    2. Calls ``detector.predict()`` to locate faces.
    3. Calls ``recognizer.predict()`` to obtain embeddings.
    4. Deduplicates: if the same person was already seen with a *higher* score
       the current detection is discarded; if the current score is higher it
       replaces the stored one.

    Returns a :class:`list` of :class:`~immich_ml.schemas.DetectedFace` dicts
    (``boundingBox``, ``embedding``, ``score``) compatible with Immich's
    ``asset_faces`` Postgres table.
    """
    # Each entry: (parsed_embedding_vector, DetectedFace)
    best: list[tuple[NDArray[np.float32], DetectedFace]] = []

    for frame in _sample_frames(video_path):
        frame = _resize_if_needed(frame)

        detections = detector.predict(frame)
        if detections["boxes"].shape[0] == 0:
            continue

        recognized: FacialRecognitionOutput = recognizer.predict(frame, detections)

        for face in recognized:
            embedding = _parse_embedding(face["embedding"])

            matched_idx: int | None = None
            for idx, (existing_emb, _) in enumerate(best):
                if _cosine_similarity(embedding, existing_emb) >= duplicate_threshold:
                    matched_idx = idx
                    break

            if matched_idx is None:
                best.append((embedding, face))
            elif face["score"] > best[matched_idx][1]["score"]:
                best[matched_idx] = (embedding, face)
            # else: existing detection has equal or higher confidence — keep it

    log.debug(f"deep_video_face_scan: {len(best)} unique face(s) found in '{video_path}'")
    return [face for _, face in best]
