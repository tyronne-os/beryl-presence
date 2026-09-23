"""Qwen3-TTS CustomVoice as a local HTTP service in its own venv (qwen-tts pins differ from FlashHead's
transformers==4.57.3 and the Decider's transformers>=5, so each model lives in its own process).

POST /speak {"text", "instruct"?, "speaker"?} -> raw PCM16 mono, 16 kHz (FlashHead's input rate).
  uvicorn tts_server:app --host 127.0.0.1 --port 8010"""
from __future__ import annotations

import os
import threading
import time
from math import gcd

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from scipy.signal import resample_poly

MODEL_ID = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
SPEAKER = os.environ.get("TTS_SPEAKER", "Vivian")
LANGUAGE = os.environ.get("TTS_LANGUAGE", "English")
ATTN = os.environ.get("TTS_ATTN", "sdpa")  # "flash_attention_2" if flash-attn is installed in this venv
OUT_SR = 16000

app = FastAPI()
_lock = threading.Lock()
_model = None
_ready = {"ok": False, "load_s": None, "warm_ms": None}


class SpeakReq(BaseModel):
    text: str
    instruct: str | None = None
    speaker: str | None = None


def _synth(text: str, instruct: str | None, speaker: str | None) -> np.ndarray:
    kw = dict(text=text, language=LANGUAGE, speaker=speaker or SPEAKER)
    if instruct:
        kw["instruct"] = instruct
    with _lock:
        wavs, sr = _model.generate_custom_voice(**kw)
    a = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    if sr != OUT_SR:
        g = gcd(OUT_SR, int(sr))
        a = resample_poly(a, OUT_SR // g, int(sr) // g).astype(np.float32)
    return np.clip(a, -1.0, 1.0)


@app.on_event("startup")
def _load():
    global _model
    from qwen_tts import Qwen3TTSModel
    t = time.perf_counter()
    _model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=ATTN)
    _ready["load_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    _synth("Hi.", None, None)
    _ready["warm_ms"] = round((time.perf_counter() - t) * 1000)
    _ready["ok"] = True


@app.get("/health")
def health():
    return {**_ready, "model": MODEL_ID, "speaker": SPEAKER,
            "vram_gb": round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None}


@app.post("/speak")
def speak(r: SpeakReq):
    t = time.perf_counter()
    a = _synth(r.text, r.instruct, r.speaker)
    pcm = (a * 32767).astype("<i2").tobytes()
    return Response(pcm, media_type="application/octet-stream",
                    headers={"X-Sample-Rate": str(OUT_SR), "X-Gen-Ms": str(round((time.perf_counter() - t) * 1000)),
                             "X-Audio-Seconds": f"{len(a) / OUT_SR:.2f}"})
