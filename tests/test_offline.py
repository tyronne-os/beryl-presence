"""No-GPU checks. Run from the repo root:
    DECIDER_REPO=/path/to/decider-plugin-brain-beryl python tests/test_offline.py
DECIDER_REPO lets the schema test use the fork's real question parser (decider/systemone.py) and answer format."""
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
DECIDER_REPO = os.environ.get("DECIDER_REPO")
if DECIDER_REPO:
    sys.path.insert(0, DECIDER_REPO)

from beryl import talking_probe as probe  # noqa: E402
from beryl.brain import split_sentences  # noqa: E402
from beryl.decider_client import QUESTIONS, DeciderClient  # noqa: E402

SR, FPS, HOP = 16000, 25, 640


def speech_like(n_frames, seed=0):
    """Noise amplitude-modulated at a syllable rate, with pauses -> an audio envelope a real mouth would follow."""
    rng = np.random.default_rng(seed)
    env = np.clip(np.sin(np.linspace(0, n_frames / FPS * 2 * np.pi * 3.5, n_frames)), 0, None) ** 0.7
    env[n_frames // 3: n_frames // 3 + 5] = 0
    audio = (rng.standard_normal(n_frames * HOP) * np.repeat(env, HOP) * 0.3).astype(np.float32)
    return audio, env


def test_probe_verdicts():
    n = 72
    audio, env = speech_like(n)
    rng = np.random.default_rng(1)
    cases = {
        "TALKING_IN_SYNC": 0.05 + 0.35 * env + rng.normal(0, 0.01, n),
        "MOUTH_STILL_WITH_AUDIO": np.full(n, 0.08) + rng.normal(0, 0.001, n),
        "TALKING_OUT_OF_SYNC": 0.05 + 0.35 * rng.random(n),
    }
    for want, mar in cases.items():
        got = probe.evaluate(mar.astype(np.float32), audio).verdict
        assert got == want, (want, got)
    silence = np.zeros(n * HOP, np.float32)
    assert probe.evaluate((0.06 + rng.normal(0, 0.002, n)).astype(np.float32), silence).verdict == "IDLE_SILENT"
    assert probe.evaluate(np.full(n, np.nan, np.float32), audio).verdict == "NO_FACE"  # stub/placeholder frames
    assert probe.telemetry_stage("MOUTH_STILL_WITH_AUDIO") == "broken"


def test_split_sentences():
    buf, out = "", []
    for tok in ["Hey", " there!", " It's", " good", " to see", " you.", " How was", " your day"]:
        s, buf = split_sentences(buf + tok)
        out += s
    assert out == ["Hey there!", "It's good to see you."], out
    assert buf.strip() == "How was your day"


def test_questions_parse_with_real_decider_parser():
    if not DECIDER_REPO:
        print("   (skipped: set DECIDER_REPO to validate against the fork's parser)")
        return
    import decider.systemone as S1
    for group, qs in QUESTIONS.items():
        for qid, spec in qs.items():
            S1.render_question(spec)  # raises on anything the server would reject with 422
        assert "other" not in json.dumps(qs).lower().split('"')  # README: no catch-all option


def _fake_answers(questions, picks):
    """Answers in exactly the server's format, built with the fork's own format_answer."""
    import decider.systemone as S1
    out = {}
    for qid, spec in questions.items():
        rq = S1.render_question(spec)
        p = [0.05] * len(rq["options"])
        p[picks.get(qid, 0)] = 1.0
        out[qid] = S1.format_answer(rq, p)
    return out


def test_decider_client_payload_and_gates():
    if not DECIDER_REPO:
        print("   (skipped: set DECIDER_REPO)")
        return
    import httpx
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body)
        assert set(body) == {"state", "questions", "independent"}, body.keys()  # serve.py S1Req fields
        if "route" in body["questions"]:
            text = body["state"]["user_said"]
            route = {"what's the capital of france?": 0, "totally": 1}.get(text, 0)
            return httpx.Response(200, json={"answers": _fake_answers(body["questions"], {"route": route, "tone": 3})})
        return httpx.Response(200, json={"answers": _fake_answers(body["questions"], {"on_topic": 1})})

    async def run():
        c = DeciderClient("http://decider.test")
        c._http = httpx.AsyncClient(base_url="http://decider.test", transport=httpx.MockTransport(handler))
        d = await c.route_turn("what's the capital of france?", [])
        assert (d.route, d.tone, d.source) == ("answer", "focused", "decider"), d
        d = await c.route_turn("totally", [])
        assert d.route == "backchannel" and d.source == "decider", d
        d = await c.route_turn("totally", [])
        assert d.source == "cache", d
        d = await c.route_turn("um", [])
        assert (d.route, d.source) == ("wait", "rule"), d
        p, src = await c.check_reply("hi", "Hi! Good to see you.")
        assert src == "decider" and p > 0.9, (p, src)

        bad = DeciderClient("http://decider.test")
        bad._http = httpx.AsyncClient(base_url="http://decider.test",
                                      transport=httpx.MockTransport(lambda r: httpx.Response(422, json={"detail": "x"})))
        d = await bad.route_turn("hello?", [])
        assert (d.route, d.source) == ("answer", "fallback"), d
        assert bad.summary()["fallback_rate"] == 1.0
    asyncio.run(run())


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
