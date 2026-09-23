"""Beryl live pipeline, one GPU:
  mic -> Silero VAD -> faster-whisper -> Decider route (in parallel with the brain's first tokens) -> Qwen3-TTS
  -> FlashHead Lite (continuous stream: speech when there is speech, rendered silence otherwise) -> browser,
  with every rendered chunk measured by the talking probe.

  python -m beryl.orchestrator --serve       http://127.0.0.1:7860 (viewer + /ws + /health)
  python -m beryl.orchestrator --selftest    speaks a hello into selftest_hello.mp4 and exits 0 only if she really talks
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from . import talking_probe as probe
from .brain import Brain
from .decider_client import DeciderClient
from .render_flashhead import FlashHeadRenderer

HERE = Path(__file__).resolve().parent.parent
SR = 16000
TONE_INSTRUCT = {
    "warm": "Speak warmly and naturally, like talking with a close friend.",
    "playful": "Speak in a light, playful, amused way.",
    "empathetic": "Speak softly and gently, with care.",
    "focused": "Speak clearly and calmly, confident and helpful.",
}
BACKCHANNELS = ["Mm-hm.", "Right.", "Yeah.", "I see."]


def _p50(xs) -> float | None:
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 3) if xs else None


@dataclass
class Settings:
    root: str
    avatar_image: str
    flashhead_repo: str
    flashhead_ckpt: str
    wav2vec: str
    decider_url: str
    tts_url: str
    brain_url: str
    brain_key: str
    brain_model: str
    stt_model: str
    bind: str
    port: int
    out_dir: str

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        root = e("BERYL_ROOT", "/opt/beryl")
        return cls(
            root=root,
            avatar_image=os.path.abspath(e("BERYL_AVATAR_IMAGE", f"{root}/repos/SoulX-FlashHead/examples/girl.png")),
            flashhead_repo=e("FLASHHEAD_REPO", f"{root}/repos/SoulX-FlashHead"),
            flashhead_ckpt=e("FLASHHEAD_CKPT", f"{root}/models/SoulX-FlashHead-1_3B"),
            wav2vec=e("WAV2VEC_DIR", f"{root}/models/wav2vec2-base-960h"),
            decider_url=e("DECIDER_URL", "http://127.0.0.1:8000"),
            tts_url=e("TTS_URL", "http://127.0.0.1:8010"),
            brain_url=e("BRAIN_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            brain_key=e("BRAIN_API_KEY", ""),
            brain_model=e("BRAIN_MODEL", "meta/llama-3.1-405b-instruct"),
            stt_model=e("STT_MODEL", "distil-large-v3"),
            bind=e("BERYL_BIND", "127.0.0.1"),
            port=int(e("BERYL_PORT", "7860")),
            out_dir=os.path.abspath(e("BERYL_OUT", f"{root}/out")),
        )


class Services:
    """Everything loaded once per process and shared by the (single) live session."""

    def __init__(self, s: Settings):
        self.s = s
        self.renderer = FlashHeadRenderer(s.flashhead_repo, s.flashhead_ckpt, s.wav2vec, s.avatar_image)
        self.decider = DeciderClient(s.decider_url)
        self.tts = httpx.AsyncClient(base_url=s.tts_url, timeout=httpx.Timeout(30.0, connect=2.0))
        self.brain = Brain(s.brain_url, s.brain_key, s.brain_model) if s.brain_key else None
        self.render_pool = ThreadPoolExecutor(1, thread_name_prefix="render")  # the GPU renders one chunk at a time
        self.stt_pool = ThreadPoolExecutor(1, thread_name_prefix="stt")
        self.probe_pool = ThreadPoolExecutor(1, thread_name_prefix="probe")
        self.stt = self.vad_model = self.tracker = self.idle_frames = None
        self.backchannels: dict[str, list[np.ndarray]] = {}
        self.boot: dict = {}
        self.ready = False
        self.render_rtf: deque[float] = deque(maxlen=100)
        self.turn_ms: deque[float] = deque(maxlen=50)
        self.probe_counts: dict[str, int] = {}
        self.last_probe: dict | None = None
        self.underruns = 0
        self.proprioception_mismatch = 0

    async def _wait_http(self, name: str, url: str, timeout_s: float = 900) -> float:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=3.0) as c:
            while time.perf_counter() - t0 < timeout_s:
                try:
                    r = await c.get(url)
                    if r.status_code == 200 and r.json().get("ok"):
                        return round(time.perf_counter() - t0, 1)
                except Exception:
                    pass
                await asyncio.sleep(2)
        raise RuntimeError(f"{name} not healthy at {url} after {timeout_s:.0f}s -- check its log")

    def _load_stt(self) -> float:
        from faster_whisper import WhisperModel
        t = time.perf_counter()
        self.stt = WhisperModel(self.s.stt_model, device="cuda", compute_type="int8_float16")
        return round(time.perf_counter() - t, 1)

    def transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self.stt.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False, vad_filter=False)
        return " ".join(seg.text.strip() for seg in segs).strip()

    def _load_tracker(self) -> float:
        t = time.perf_counter()
        self.tracker = probe.MouthTracker()
        return round(time.perf_counter() - t, 1)

    async def speak(self, text: str, tone: str = "warm") -> np.ndarray:
        r = await self.tts.post("/speak", json={"text": text, "instruct": TONE_INSTRUCT.get(tone)})
        r.raise_for_status()
        return np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0

    async def start(self, with_session_tools: bool = True):
        t_all = time.perf_counter()
        loop = asyncio.get_running_loop()
        render_load = loop.run_in_executor(self.render_pool, self.renderer.load)  # slowest: overlaps everything below
        tracker_load = loop.run_in_executor(self.probe_pool, self._load_tracker)
        self.boot["decider_ready_s"] = await self._wait_http("decider", self.s.decider_url + "/health")
        self.boot["tts_ready_s"] = await self._wait_http("tts", self.s.tts_url + "/health")
        if with_session_tools:
            self.boot["stt_load_s"] = await loop.run_in_executor(self.stt_pool, self._load_stt)
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad()
        self.boot["tracker_load_s"] = await tracker_load
        self.boot["render_load_s"] = round(await render_load, 1)
        self.boot["decider_warm_ms"] = round(await self.decider.warm(), 1)
        if with_session_tools:
            t = time.perf_counter()
            for tone in TONE_INSTRUCT:
                self.backchannels[tone] = [await self.speak(b, tone) for b in BACKCHANNELS]
            self.boot["backchannel_cache_s"] = round(time.perf_counter() - t, 1)
        frames, secs = await loop.run_in_executor(
            self.render_pool, self.renderer.render, np.zeros(self.renderer.slice_samples, np.float32))
        self.idle_frames = frames  # real model-rendered idle, shown the instant a viewer connects
        self.boot["first_render_s"] = round(secs, 3)
        self.boot["ready_s"] = round(time.perf_counter() - t_all, 1)
        self.ready = True

    def health(self) -> dict:
        import torch
        free, total = torch.cuda.mem_get_info()
        rtf = _p50(self.render_rtf)
        return {
            "ready": self.ready,
            "boot": self.boot,
            "gpu": {"name": torch.cuda.get_device_name(0), "used_gb_all_processes": round((total - free) / 2**30, 2),
                    "total_gb": round(total / 2**30, 2)},
            "decider": self.decider.summary(),
            "render": {"rtf_p50": rtf, "realtime": rtf is not None and rtf < 1.0, "underruns": self.underruns,
                       "chunk_s": round(self.renderer.chunk_seconds, 3) if self.renderer.pipe else None},
            "turn_latency_ms_p50": _p50(self.turn_ms),
            "probe": {"last": self.last_probe, "counts": self.probe_counts,
                      "proprioception_mismatch": self.proprioception_mismatch},
            "brain": {"configured": self.brain is not None, "model": self.s.brain_model},
        }


class Session:
    def __init__(self, svc: Services, send_bytes, send_json):
        import torch
        from silero_vad import VADIterator
        self._torch = torch
        self.svc, self.alive = svc, True
        self._send_lock = asyncio.Lock()
        self._send_bytes, self._send_json = send_bytes, send_json
        self.speech = np.zeros(0, np.float32)
        self.flush_idle = asyncio.Event()
        self.out_q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.vad = VADIterator(svc.vad_model, sampling_rate=SR, threshold=0.5, min_silence_duration_ms=450, speech_pad_ms=120)
        self._vad_rem = np.zeros(0, np.float32)
        self._preroll: deque = deque(maxlen=10)
        self.user_audio: list[np.ndarray] = []
        self.user_speaking = False
        self.user_end_t: float | None = None
        self.reply_task: asyncio.Task | None = None
        self.history_llm: list[dict] = []
        self.history_txt: deque[str] = deque(maxlen=8)
        self.window: deque = deque(maxlen=3)
        self.still_streak = 0

    async def send_bytes(self, b: bytes):
        async with self._send_lock:
            await self._send_bytes(b)

    async def send_json(self, j: dict):
        async with self._send_lock:
            await self._send_json(j)

    # ---- input side
    def on_audio(self, b: bytes):
        x = np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
        buf = np.concatenate((self._vad_rem, x))
        n = len(buf) // 512 * 512
        self._vad_rem = buf[n:]
        for f in buf[:n].reshape(-1, 512):
            ev = self.vad(self._torch.from_numpy(f.copy()))
            (self.user_audio if self.user_speaking else self._preroll).append(f)
            if ev and "start" in ev:
                self.user_speaking = True
                self.user_audio = list(self._preroll)
                self._preroll.clear()
                if len(self.speech) or (self.reply_task and not self.reply_task.done()):
                    self.barge_in()
            elif ev and "end" in ev:
                self.user_speaking = False
                audio = np.concatenate(self.user_audio) if self.user_audio else np.zeros(0, np.float32)
                self.user_audio = []
                if len(audio) > 0.3 * SR:
                    self.user_end_t = time.perf_counter()
                    self.start_turn(audio=audio)

    def barge_in(self):
        self.speech = np.zeros(0, np.float32)
        self.flush_idle.clear()
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        asyncio.create_task(self.send_json({"type": "barge_in"}))

    def start_turn(self, audio: np.ndarray | None = None, text: str | None = None):
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        if text is not None:
            self.user_end_t = time.perf_counter()
        self.reply_task = asyncio.create_task(self.handle_turn(audio, text))

    def enqueue_speech(self, pcm: np.ndarray):
        self.speech = np.concatenate((self.speech, pcm))
        self.flush_idle.set()

    # ---- thinking side
    async def _pump_brain(self, text: str, q: asyncio.Queue):
        try:
            async for s in self.svc.brain.stream_sentences(self.history_llm[-12:], text):
                q.put_nowait(s)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            q.put_nowait(("error", f"{type(e).__name__}: {e}"))
        finally:
            q.put_nowait(None)

    async def handle_turn(self, audio, text):
        svc, loop = self.svc, asyncio.get_running_loop()
        try:
            if text is None:
                t = time.perf_counter()
                text = await loop.run_in_executor(svc.stt_pool, svc.transcribe, audio)
                await self.send_json({"type": "transcript", "text": text, "stt_ms": round((time.perf_counter() - t) * 1000)})
            if not text.strip():
                return
            q: asyncio.Queue = asyncio.Queue()
            llm = asyncio.create_task(self._pump_brain(text, q)) if svc.brain else None  # speculative: starts now
            d = await svc.decider.route_turn(text, list(self.history_txt))
            await self.send_json(d.as_event())
            if d.route != "answer" and llm:
                llm.cancel()
            if d.route == "wait":
                self.user_end_t = None
                return
            self.history_txt.append(f"user: {text}")
            if d.route == "backchannel":
                self.enqueue_speech(random.choice(svc.backchannels.get(d.tone) or svc.backchannels["warm"]))
                self.history_txt.append("beryl: (acknowledges)")
                return
            if llm is None:
                q.put_nowait(f"I heard you say: {text}. My brain isn't connected yet, so that's all I've got.")
                q.put_nowait(None)
            said, first = [], True
            while (item := await q.get()) is not None:
                if isinstance(item, tuple):
                    await self.send_json({"type": "error", "where": "brain", "message": item[1]})
                    item = "Sorry, I lost my train of thought for a second."
                sentence, stop = item, False
                if first and svc.brain:
                    p_ok, src = await svc.decider.check_reply(text, sentence)
                    await self.send_json({"type": "reply_check", "p_on_topic": round(p_ok, 3), "source": src})
                    if p_ok < 0.3:
                        if llm:
                            llm.cancel()
                        sentence, stop = "Sorry, I think I misheard. Could you say that again?", True
                first = False
                self.enqueue_speech(await svc.speak(sentence, d.tone))
                said.append(sentence)
                await self.send_json({"type": "say", "text": sentence, "tone": d.tone})
                if stop:
                    break
            reply = " ".join(said)
            self.history_llm += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            self.history_txt.append(f"beryl: {reply}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.send_json({"type": "error", "where": "turn", "message": f"{type(e).__name__}: {e}"})

    # ---- output side
    async def render_loop(self):
        """Producer: always renders the next 0.96 s -- speech if any is queued, otherwise silence (real idle motion)."""
        r, loop = self.svc.renderer, asyncio.get_running_loop()
        n = r.slice_samples
        while self.alive:
            if len(self.speech):
                s, self.speech, speaking = self.speech[:n], self.speech[n:], True
                if len(s) < n:
                    s = np.pad(s, (0, n - len(s)))
            else:
                s, speaking = np.zeros(n, np.float32), False
            frames, secs = await loop.run_in_executor(self.svc.render_pool, r.render, s)
            self.svc.render_rtf.append(secs / r.chunk_seconds)
            await self.out_q.put((frames, s, speaking))

    async def pace_loop(self):
        """Consumer: 25 fps to the browser. Plays cached idle frames until the live stream starts (instant presence),
        and cuts idle short the moment speech is waiting (lower turn latency)."""
        import cv2
        svc, loop = self.svc, asyncio.get_running_loop()
        dt = 1.0 / svc.renderer.fps
        next_t, idle_i, live = loop.time(), 0, False

        async def show(frame):
            nonlocal next_t
            ok, jpg = cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 82])
            delay = next_t - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.5:
                next_t = loop.time()
            await self.send_bytes(b"\x01" + jpg.tobytes())
            next_t += dt

        while self.alive:
            try:
                frames, audio, speaking = self.out_q.get_nowait()
            except asyncio.QueueEmpty:
                if not live:
                    await show(svc.idle_frames[idle_i % len(svc.idle_frames)])
                    idle_i += 1
                    continue
                if not self.flush_idle.is_set():  # waiting on purpose for speech is not the GPU being slow
                    svc.underruns += 1
                frames, audio, speaking = await self.out_q.get()
                next_t = max(next_t, loop.time())
            live = True
            if not speaking and self.flush_idle.is_set():
                continue
            if speaking:
                self.flush_idle.clear()
                if self.user_end_t is not None:
                    svc.turn_ms.append((time.perf_counter() - self.user_end_t) * 1000)
                    self.user_end_t = None
                await self.send_bytes(b"\x02" + (audio * 32767).astype("<i2").tobytes())
            shown = 0
            for f in frames:
                if not speaking and self.flush_idle.is_set():
                    break
                await show(f)
                shown += 1
            if shown == len(frames):
                asyncio.create_task(self._probe(frames, audio, speaking))

    async def _probe(self, frames, audio, speaking):
        svc = self.svc
        mar = await asyncio.get_running_loop().run_in_executor(svc.probe_pool, svc.tracker.mar, frames)
        self.window.append((mar, audio, speaking))
        res = probe.evaluate(np.concatenate([w[0] for w in self.window]), np.concatenate([w[1] for w in self.window]))
        svc.last_probe = res.as_event()
        svc.probe_counts[res.verdict] = svc.probe_counts.get(res.verdict, 0) + 1
        stated = "speaking" if sum(w[2] for w in self.window) * 2 > len(self.window) else "idle"
        telemetry = probe.telemetry_stage(res.verdict)
        mismatch = stated != telemetry
        if mismatch:
            svc.proprioception_mismatch += 1
        self.still_streak = self.still_streak + 1 if res.verdict == "MOUTH_STILL_WITH_AUDIO" else 0
        await self.send_json({**res.as_event(), "stated_stage": stated, "telemetry_stage": telemetry, "mismatch": mismatch})
        if self.still_streak >= 2:
            svc.renderer.reset()
            self.still_streak = 0
            await self.send_json({"type": "alert", "message": "mouth frozen while audio plays for 2 windows -- renderer context reset"})


# ---------------------------------------------------------------- web app
def build_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse

    app = FastAPI()
    state = {"svc": None, "active": 0}

    @app.on_event("startup")
    async def _startup():
        state["svc"] = Services(Settings.from_env())
        await state["svc"].start()

    @app.get("/")
    def index():
        return FileResponse(HERE / "viewer.html")

    @app.get("/health")
    def health():
        return state["svc"].health()

    @app.websocket("/render")
    async def ws_render(websocket: WebSocket):
        """Thin TTS+render endpoint for ELANA's engine.
        ELANA already handles STT + brain + Beryl; this endpoint only speaks and renders.
        Client sends: {"type":"speak","text":"...","tone":"warm"}
        Client sends: {"type":"stop"} to interrupt current render
        Server streams: 0x01+JPEG frames, 0x02+PCM16@16kHz audio, then {"type":"done"}
        """
        import cv2
        await websocket.accept()
        svc = state["svc"]
        if not svc.ready:
            await websocket.send_json({"type": "error", "message": "service not ready"})
            await websocket.close()
            return
        loop = asyncio.get_running_loop()
        stopped = False

        async def stream_one(text: str, tone: str):
            nonlocal stopped
            stopped = False
            pcm_f32 = await svc.speak(text, tone)
            if not len(pcm_f32):
                await websocket.send_json({"type": "done"})
                return
            n = svc.renderer.slice_samples
            pad = (-len(pcm_f32)) % n
            chunks = np.pad(pcm_f32, (0, pad)).reshape(-1, n)
            for chunk in chunks:
                if stopped:
                    break
                frames, _ = await loop.run_in_executor(svc.render_pool, svc.renderer.render, chunk.copy())
                if stopped:
                    break
                await websocket.send_bytes(b"\x02" + (chunk * 32767).astype("<i2").tobytes())
                for frame in frames:
                    if stopped:
                        break
                    ok, jpg = cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        await websocket.send_bytes(b"\x01" + jpg.tobytes())
            await websocket.send_json({"type": "done"})

        current: asyncio.Task | None = None
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if not msg.get("text"):
                    continue
                j = json.loads(msg["text"])
                if j.get("type") == "stop":
                    stopped = True
                    if current and not current.done():
                        current.cancel()
                    continue
                if j.get("type") == "speak" and j.get("text", "").strip():
                    if current and not current.done():
                        stopped = True
                        current.cancel()
                    current = asyncio.create_task(stream_one(j["text"].strip(), j.get("tone", "warm")))
                    await current
        except WebSocketDisconnect:
            pass
        finally:
            stopped = True
            if current and not current.done():
                current.cancel()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        svc = state["svc"]
        if state["active"] >= 1:
            await websocket.send_json({"type": "error", "message": "one live session at a time on this GPU"})
            await websocket.close()
            return
        state["active"] += 1
        sess = Session(svc, websocket.send_bytes, websocket.send_json)
        tasks = [asyncio.create_task(sess.render_loop()), asyncio.create_task(sess.pace_loop())]
        try:
            await sess.send_json({"type": "hello", "boot": svc.boot})
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    sess.on_audio(msg["bytes"])
                elif msg.get("text"):
                    j = json.loads(msg["text"])
                    if j.get("type") == "say" and j.get("text", "").strip():
                        sess.start_turn(text=j["text"].strip())
        except WebSocketDisconnect:
            pass
        finally:
            sess.alive = False
            for t in tasks + ([sess.reply_task] if sess.reply_task else []):
                t.cancel()
            svc.renderer.reset()
            state["active"] -= 1

    return app


# ---------------------------------------------------------------- selftest
async def selftest(s: Settings, text: str) -> int:
    import imageio.v2 as imageio
    import imageio_ffmpeg

    svc = Services(s)
    await svc.start(with_session_tools=False)
    loop = asyncio.get_running_loop()
    fails, report = [], {"boot": svc.boot}

    decisions = []
    for u in ["hey, how's it going?", "mm-hm", "so the thing about my job is that", "I just found out my dog is sick."]:
        d = await svc.decider.route_turn(u, [])
        decisions.append({"utterance": u, "route": d.route, "p": round(d.route_confidence, 3), "tone": d.tone,
                          "ms": round(d.latency_ms, 1), "source": d.source})
    report["decider"] = {"decisions": decisions, **svc.decider.summary()}
    if svc.decider.fallbacks:
        fails.append(f"Decider fell back ({svc.decider.last_error}) -- it is not actually deciding")

    t = time.perf_counter()
    pcm = await svc.speak(text, "warm")
    report["tts"] = {"ms": round((time.perf_counter() - t) * 1000), "audio_s": round(len(pcm) / SR, 2),
                     "rms": round(float(np.sqrt(np.mean(pcm**2))), 4)}
    if report["tts"]["rms"] < 0.01:
        fails.append("TTS returned (near) silence")

    n = svc.renderer.slice_samples
    speech = np.pad(pcm, (0, (-len(pcm)) % n))
    chunks = [np.zeros(n, np.float32)] + list(speech.reshape(-1, n)) + [np.zeros(n, np.float32)]
    frames, marks, secs = [], [], []
    for c in chunks:
        f, sec = await loop.run_in_executor(svc.render_pool, svc.renderer.render, c)
        frames.append(f)
        secs.append(sec)
        marks.append(await loop.run_in_executor(svc.probe_pool, svc.tracker.mar, f))
    rtf = [x / svc.renderer.chunk_seconds for x in secs]
    report["render"] = {"chunks": len(chunks), "chunk_s": round(svc.renderer.chunk_seconds, 3),
                        "gpu_s_per_chunk_p50": _p50(secs), "rtf_p50": _p50(rtf), "fps_p50": round(svc.renderer.slice_frames / _p50(secs), 1)}

    windows = []
    for i in range(len(chunks)):
        lo = max(0, i - 2)
        r = probe.evaluate(np.concatenate(marks[lo:i + 1]), np.concatenate(chunks[lo:i + 1]))
        windows.append(r.as_event())
    speech_windows = [w for w in windows if w["speech_frac"] >= probe.SPEECH_FRAC]
    report["probe"] = {"windows": windows}
    verdicts = {w["verdict"] for w in speech_windows}
    if "NO_FACE" in {w["verdict"] for w in windows}:
        fails.append("NO_FACE: rendered frames do not contain a trackable face")
    if "MOUTH_STILL_WITH_AUDIO" in verdicts:
        fails.append("MOUTH_STILL_WITH_AUDIO: audio plays but the painted mouth does not move")
    if "TALKING_IN_SYNC" not in verdicts:
        fails.append("no speech window reached TALKING_IN_SYNC (check thresholds against the printed values)")

    os.makedirs(s.out_dir, exist_ok=True)
    silent_mp4, wav, out = (os.path.join(s.out_dir, x) for x in ("_video.mp4", "_audio.wav", "selftest_hello.mp4"))
    imageio.mimwrite(silent_mp4, np.concatenate(frames), fps=svc.renderer.fps, codec="libx264", macro_block_size=1)
    import wave
    with wave.open(wav, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.concatenate(chunks) * 32767).astype("<i2").tobytes())
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", silent_mp4, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)
    report["video"] = out

    realtime = report["render"]["rtf_p50"] is not None and report["render"]["rtf_p50"] < 1.0
    print("\n==================== BERYL SELFTEST ====================")
    print(f"boot                {svc.boot}")
    for d in decisions:
        print(f"decider  {d['source']:8s} {d['ms']:7.1f} ms  {d['route']:11s} p={d['p']:.2f} tone={d['tone']:10s} <- {d['utterance']!r}")
    print(f"tts                 {report['tts']}")
    print(f"render              {report['render']}  {'REAL-TIME' if realtime else 'NOT REAL-TIME ON THIS GPU'}")
    for i, w in enumerate(windows):
        print(f"probe window {i:2d}     {w['verdict']:26s} speech={w['speech_frac']:.2f} motion={w['mouth_motion']:.4f} "
              f"sync_r={w['sync_r']:.2f} lag={w['sync_lag']} face={w['face_ratio']:.2f}")
    print(f"video               {out}")
    print("--------------------------------------------------------")
    print("VERDICT: " + ("TALKING -- the pipeline works end to end" if not fails else "NOT TALKING"))
    for f in fails:
        print(f"  x {f}")
    if not realtime:
        print("  ! renders correctly but slower than real time on this GPU: the live stream will stall; use a faster GPU")
    Path(s.out_dir, "selftest_report.json").write_text(json.dumps(report, indent=2))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--text", default="Hi, I'm Beryl. It's really good to see you tonight. How was your day?")
    args = ap.parse_args()
    s = Settings.from_env()
    if args.selftest:
        sys.exit(asyncio.run(selftest(s, args.text)))
    import uvicorn
    uvicorn.run(build_app(), host=s.bind, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
