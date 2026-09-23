"""B.B.P Decider client (tyronne-os/decider-plugin-brain-beryl) over its real POST /v1/systemone wire format.

The Decider answers typed questions about a state with calibrated probabilities. It routes turns and judges replies;
it does not synthesize motion. Every call that fails is counted, so a bypassed Decider shows up at /health as a
fallback rate instead of silently defaulting."""
from __future__ import annotations

import json
import re
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

QUESTIONS = json.loads((Path(__file__).resolve().parent.parent / "decider_questions.json").read_text())
HESITATIONS = {"", "uh", "um", "uhm", "er", "hmm", "mm"}
ROUTES = ("answer", "backchannel", "wait")


@dataclass
class TurnDecision:
    route: str
    route_confidence: float
    tone: str
    frustration: float
    latency_ms: float
    source: str  # decider | cache | rule | fallback

    def as_event(self) -> dict:
        return {"type": "decision", **asdict(self)}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()


class DeciderClient:
    def __init__(self, base_url: str, timeout_s: float = 0.3, cache_size: int = 256, gate: float = 0.6):
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=1.0),
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=600),
        )
        self._cache: OrderedDict[str, TurnDecision] = OrderedDict()
        self._cache_size = cache_size
        self._gate = gate  # below this, never ignore the user: backchannel/wait fall through to answer
        self.calls = self.fallbacks = self.cache_hits = self.rule_hits = 0
        self.latencies: deque[float] = deque(maxlen=200)
        self.last_error = ""

    async def close(self):
        await self._http.aclose()

    async def health(self) -> dict:
        try:
            r = await self._http.get("/health", timeout=2.0)
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def ask(self, state, questions: dict) -> dict:
        r = await self._http.post("/v1/systemone", json={"state": state, "questions": questions, "independent": True})
        r.raise_for_status()
        return r.json()["answers"]

    async def warm(self, rounds: int = 3) -> float:
        """First requests capture CUDA graphs for new shapes; pay that before a user is waiting."""
        ms = 0.0
        for _ in range(rounds):
            t = time.perf_counter()
            await self.ask({"user_said": "hey, how's it going?", "recent_conversation": []}, QUESTIONS["turn"])
            await self.ask({"user_said": "hi", "beryl_reply": "Hi! Good to see you."}, QUESTIONS["reply_check"])
            ms = (time.perf_counter() - t) * 1000
        return ms

    async def route_turn(self, transcript: str, history: list[str]) -> TurnDecision:
        t0 = time.perf_counter()
        key = _norm(transcript)
        if key in HESITATIONS:
            self.rule_hits += 1
            return TurnDecision("wait", 1.0, "warm", 0.0, 0.0, "rule")
        short = len(key.split()) <= 3
        if short and key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            d = self._cache[key]
            return TurnDecision(d.route, d.route_confidence, d.tone, d.frustration, 0.0, "cache")

        self.calls += 1
        try:
            a = await self.ask({"user_said": transcript, "recent_conversation": history[-4:]}, QUESTIONS["turn"])
        except Exception as e:
            self.fallbacks += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return TurnDecision("answer", 0.0, "warm", 0.0, (time.perf_counter() - t0) * 1000, "fallback")

        ms = (time.perf_counter() - t0) * 1000
        self.latencies.append(ms)
        route, conf = a["route"]["choice"], float(a["route"]["confidence"])
        if route != "answer" and conf < self._gate:
            route = "answer"
        d = TurnDecision(route, conf, a["tone"]["choice"], float(a["frustration"]["score"]), ms, "decider")
        if short:
            self._cache[key] = d
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return d

    async def check_reply(self, user_text: str, reply: str) -> tuple[float, str]:
        """P(reply is on-topic). Fails open (1.0) so a Decider outage never mutes Beryl, but the fallback is counted."""
        self.calls += 1
        try:
            a = await self.ask({"user_said": user_text, "beryl_reply": reply}, QUESTIONS["reply_check"])
            return float(a["on_topic"]["noul"]), "decider"
        except Exception as e:
            self.fallbacks += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return 1.0, "fallback"

    def summary(self) -> dict:
        lat = sorted(self.latencies)
        p50 = lat[len(lat) // 2] if lat else None
        return {
            "calls": self.calls,
            "fallbacks": self.fallbacks,
            "fallback_rate": round(self.fallbacks / self.calls, 3) if self.calls else None,
            "cache_hits": self.cache_hits,
            "rule_hits": self.rule_hits,
            "p50_ms": round(p50, 2) if p50 is not None else None,
            "last_error": self.last_error,
        }
