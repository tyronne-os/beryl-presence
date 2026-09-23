"""Cognitive core, offloaded: any OpenAI-compatible chat endpoint (NVIDIA NIM hosted by default), streamed and cut
into sentences so TTS starts on sentence one while the model is still writing sentence two. Uses zero VRAM here."""
from __future__ import annotations

import json
import re
from typing import AsyncIterator

import httpx

SYSTEM_PROMPT = (
    "You are Beryl, talking out loud on a live video call. Speak like a warm, quick-witted friend: short spoken "
    "sentences, one to three per turn, no lists, no markdown, no emojis. Ask something back when it fits."
)
_END = re.compile(r'([.!?]+["\')\]]?)(\s+)')


def split_sentences(buffer: str, soft_limit: int = 140) -> tuple[list[str], str]:
    """Pull complete sentences off the front of buffer -> (sentences, remainder)."""
    out = []
    while True:
        m = _END.search(buffer)
        if m:
            out.append(buffer[: m.end(1)].strip())
            buffer = buffer[m.end():]
            continue
        if len(buffer) > soft_limit and "," in buffer:
            cut = buffer.rfind(",") + 1
            out.append(buffer[:cut].strip())
            buffer = buffer[cut:].lstrip()
            continue
        return [s for s in out if s], buffer


class Brain:
    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 160, temperature: float = 0.7):
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    async def close(self):
        await self._http.aclose()

    async def stream_sentences(self, history: list[dict], user_text: str) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_text}]
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "stream": True}
        buf = ""
        async with self._http.stream("POST", "/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
                sentences, buf = split_sentences(buf + delta)
                for s in sentences:
                    yield s
        if buf.strip():
            yield buf.strip()
