# beryl-presence

Beryl's presence engine: a talking avatar from one photo, on one GPU, that has to prove it is really talking.

```
mic ─ Silero VAD ─ faster-whisper ─┬─ B.B.P Decider (route · tone · frustration) ──┐
                                   └─ brain (hosted LLM, streamed by sentence) ────┴─ Decider reply check
                                                                                     │
browser ◄─ 25 fps pacer ◄─ talking probe ◄─ FlashHead Lite (speech or rendered idle) ◄─ Qwen3-TTS
```

## The rule this repo exists to keep

Never score a placeholder as the product. `beryl/talking_probe.py` measures mouth opening from rendered pixels
against the driving audio. No face → `NO_FACE`. Audio with a still mouth → `MOUTH_STILL_WITH_AUDIO`. Only
`TALKING_IN_SYNC` passes. `run.sh` will not start the live server until the selftest passes.

## Where each model runs (L4, 24 GB)

| Model | Role | Placement |
|---|---|---|
| SoulX-FlashHead Lite 1.3B | render | GPU |
| Decider-2B (tyronne-os/decider-plugin-brain-beryl) | routing, tone, reply check | GPU |
| Qwen3-TTS 0.6B CustomVoice | voice | GPU |
| faster-whisper distil-large-v3 | speech to text | GPU |
| Silero VAD, MediaPipe FaceMesh | turn detection, talking probe | CPU |
| LLM brain | thinking | hosted API (NVIDIA NIM by default) |

Each GPU model runs in its own venv/process because their dependency pins are incompatible.

## Run

```bash
GCP_PROJECT=... GCP_ZONE=us-central1-a bash gcp_launch.sh    # laptop: create the spot GPU box, copy this repo
bash setup_gpu.sh                                             # on the box, once: weights + venvs, timed
BERYL_AVATAR_IMAGE=~/beryl.png BRAIN_API_KEY=... bash run.sh  # selftest -> live server on 127.0.0.1:7860
```

`curl localhost:7860/health` shows render real-time factor, turn latency, Decider latency and fallback rate,
and the probe's last verdict. No-GPU tests: `DECIDER_REPO=/path/to/decider python tests/test_offline.py`.

## Open questions (measure, don't assume)

- FlashHead Lite on an L4: the 96 FPS figure is from an RTX 4090. The selftest prints the real speed.
- Talking-probe thresholds are uncalibrated defaults; set them from the selftest's printed values.
- End-of-speech to voice: estimated 2–3.5 s for answers, 1–1.5 s for backchannels. `/health` has the real numbers.
