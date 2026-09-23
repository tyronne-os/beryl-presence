"""Is the avatar actually talking? Measured from painted pixels and the audio it was driven with -- never from
internal values. This is the MOTION-pillar fix: a stub, a frozen face, or frames behind an opaque layer all fail here.

Thresholds are UNCALIBRATED defaults. Run `--selftest` on your avatar, read the speech vs. silence values it prints,
and set these between them."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SR, FPS = 16000, 25
SPEECH_DB = -38.0      # per-frame audio RMS above this counts as speech
SPEECH_FRAC = 0.30     # window is "speech" when this share of frames is above SPEECH_DB
MOTION_MIN = 0.012     # std of mouth-aspect-ratio needed to call the mouth "moving"
SYNC_MIN = 0.25        # best Pearson r (audio envelope vs mouth opening, lag -2..+2 frames) for "in sync"
FACE_MIN = 0.80        # share of frames with a detected face

TALKING = {"TALKING_IN_SYNC", "TALKING_OUT_OF_SYNC"}


@dataclass
class ProbeResult:
    verdict: str
    frames: int
    face_ratio: float
    speech_frac: float
    mouth_motion: float
    sync_r: float
    sync_lag: int

    def as_event(self) -> dict:
        return {"type": "probe", **asdict(self)}


def audio_envelope_db(audio: np.ndarray, n_frames: int) -> np.ndarray:
    hop = SR // FPS
    a = np.zeros(n_frames * hop, dtype=np.float32)
    a[: min(len(a), len(audio))] = audio[: len(a)]
    rms = np.sqrt(np.mean(a.reshape(n_frames, hop) ** 2, axis=1) + 1e-12)
    return np.maximum(20 * np.log10(rms), -60.0)


def best_lagged_r(x: np.ndarray, y: np.ndarray, max_lag: int = 2) -> tuple[float, int]:
    best, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        a, b = (x[lag:], y[: len(y) - lag]) if lag >= 0 else (x[:lag], y[-lag:])
        if len(a) < 8 or a.std() < 1e-6 or b.std() < 1e-6:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best:
            best, best_lag = r, lag
    return best, best_lag


def evaluate(mar: np.ndarray, audio: np.ndarray) -> ProbeResult:
    """mar: mouth-aspect-ratio per frame, NaN where no face was found. audio: float32 16 kHz aligned to the frames."""
    n = len(mar)
    face = ~np.isnan(mar)
    face_ratio = float(face.mean()) if n else 0.0
    env = audio_envelope_db(audio, n) if n else np.zeros(0)
    speech_frac = float((env > SPEECH_DB).mean()) if n else 0.0
    if n == 0 or face_ratio < FACE_MIN:
        return ProbeResult("NO_FACE", n, face_ratio, speech_frac, 0.0, 0.0, 0)
    m, e = mar[face], env[face]
    motion = float(m.std())
    r, lag = best_lagged_r(e, m)
    if speech_frac >= SPEECH_FRAC:
        if motion < MOTION_MIN:
            v = "MOUTH_STILL_WITH_AUDIO"
        elif r >= SYNC_MIN:
            v = "TALKING_IN_SYNC"
        else:
            v = "TALKING_OUT_OF_SYNC"
    else:
        v = "MOUTH_MOVING_WHILE_SILENT" if motion >= 3 * MOTION_MIN else "IDLE_SILENT"
    return ProbeResult(v, n, round(face_ratio, 3), round(speech_frac, 3), round(motion, 4), round(r, 3), lag)


def telemetry_stage(verdict: str) -> str:
    if verdict in TALKING:
        return "speaking"
    if verdict in ("IDLE_SILENT", "MOUTH_MOVING_WHILE_SILENT"):
        return "idle"
    return "broken"


class MouthTracker:
    """MediaPipe FaceMesh (already a FlashHead dependency). MAR = inner-lip gap / mouth width."""
    UPPER, LOWER, LEFT, RIGHT = 13, 14, 78, 308

    def __init__(self):
        import mediapipe as mp
        self._fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    def mar(self, frames_rgb: np.ndarray) -> np.ndarray:
        out = np.full(len(frames_rgb), np.nan, dtype=np.float32)
        for i, f in enumerate(frames_rgb):
            res = self._fm.process(np.ascontiguousarray(f))
            if not res.multi_face_landmarks:
                continue
            p = res.multi_face_landmarks[0].landmark
            gap = np.hypot(p[self.UPPER].x - p[self.LOWER].x, p[self.UPPER].y - p[self.LOWER].y)
            width = np.hypot(p[self.LEFT].x - p[self.RIGHT].x, p[self.LEFT].y - p[self.RIGHT].y)
            out[i] = gap / width if width > 1e-6 else np.nan
        return out
