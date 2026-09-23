#!/usr/bin/env bash
# One-time setup on the GPU box. Idempotent: re-running skips finished steps. Prints how long each step took.
#   bash setup_gpu.sh            (run it over SSH so you can watch it -- not as a hidden startup-script)
#
# Three venvs on purpose -- the models' pins are incompatible:
#   render  (py3.10)  FlashHead (transformers==4.57.3, mediapipe 0.10.9) + faster-whisper + silero-vad + orchestrator
#   decider (py3.12)  your Decider fork (transformers>=5, flash-linear-attention)
#   tts     (py3.12)  Qwen3-TTS
# Weights pulled (~16.5 GB): FlashHead Lite 8.3 GB (Pro's 6 GB skipped), decider-2b 3.8 GB, Qwen3-TTS-0.6B 2.5 GB,
# faster-distil-whisper-large-v3 1.5 GB, wav2vec2-base-960h 0.4 GB (safetensors only).
set -euo pipefail

ROOT="${BERYL_ROOT:-/opt/beryl}"
APP="$(cd "$(dirname "$0")" && pwd)"
T0=$(date +%s)
step() { printf '\n== [%5ss] %s\n' "$(( $(date +%s) - T0 ))" "$*"; }

sudo mkdir -p "$ROOT" && sudo chown "$(id -u):$(id -g)" "$ROOT"
mkdir -p "$ROOT"/{repos,models,venvs,logs,out}

step "GPU"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader \
  || { echo "No working NVIDIA driver. On a GCP Deep Learning VM: sudo /opt/deeplearning/install-driver.sh"; exit 1; }

step "system packages + uv"
sudo apt-get update -qq && sudo apt-get install -y -qq git curl ca-certificates >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
HF() { uvx --from "huggingface_hub[hf_xet]" hf download "$@"; }
# cu126 wheels run on any 12.x-era driver (the Deep Learning VM images ship 550+); PyPI's default torch may need a newer one
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
cuda_ok() { "$1" -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('$2 venv OK:', torch.__version__, torch.cuda.get_device_name(0))"; }

step "code"
[ -d "$ROOT/repos/SoulX-FlashHead/.git" ] || git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead "$ROOT/repos/SoulX-FlashHead"
[ -d "$ROOT/repos/decider/.git" ] || git clone --depth 1 https://github.com/tyronne-os/decider-plugin-brain-beryl "$ROOT/repos/decider"

step "weights (skips files already present)"
HF Soul-AILab/SoulX-FlashHead-1_3B --local-dir "$ROOT/models/SoulX-FlashHead-1_3B" \
   --include "Model_Lite/*" "VAE_LTX/*" "VAE_Wan/*" "config.json" "model_index.json"
HF facebook/wav2vec2-base-960h --local-dir "$ROOT/models/wav2vec2-base-960h" --include "*.json" "model.safetensors"
# local dir, not the HF id: serve.py reads decider_config.json from this path (fitted temperature 1.3, isolated levels)
HF Mapika/decider-2b --local-dir "$ROOT/models/decider-2b"
HF Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir "$ROOT/models/qwen3-tts-0.6b"
HF Systran/faster-distil-whisper-large-v3 --local-dir "$ROOT/models/faster-distil-whisper-large-v3"
du -sh "$ROOT/models"/*

step "venv: render (Python 3.10)"
[ -x "$ROOT/venvs/render/bin/python" ] || uv venv --seed --python 3.10 "$ROOT/venvs/render"
RP="$ROOT/venvs/render/bin/python"
uv pip install -p "$RP" torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$TORCH_INDEX"
# FlashHead's own pinned set, installed with pip (not uv) exactly as its README does -- uv refuses its nccl pin
"$RP" -m pip install -q -r "$ROOT/repos/SoulX-FlashHead/requirements.txt"
"$RP" -m pip install -q ninja packaging psutil
MAX_JOBS=4 "$RP" -m pip install -q flash_attn==2.8.0.post2 --no-build-isolation \
  || echo "!! flash_attn did not install -- FlashHead may still run on xformers; the selftest will tell you"
uv pip install -p "$RP" faster-whisper silero-vad fastapi "uvicorn[standard]" httpx
if ! "$RP" -c "import mediapipe, cv2, numpy" 2>/dev/null; then
  echo "!! mediapipe 0.10.9 is built for NumPy 1.x -- pinning numpy<2 and opencv<4.12 in the render venv"
  "$RP" -m pip install -q "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12"
fi
"$RP" -c "import mediapipe, cv2, faster_whisper, silero_vad"
cuda_ok "$RP" render

step "venv: decider (Python 3.12)"
[ -x "$ROOT/venvs/decider/bin/python" ] || uv venv --seed --python 3.12 "$ROOT/venvs/decider"
uv pip install -p "$ROOT/venvs/decider/bin/python" torch --index-url "$TORCH_INDEX"
uv pip install -p "$ROOT/venvs/decider/bin/python" -e "$ROOT/repos/decider[serve]"
"$ROOT/venvs/decider/bin/python" -c "import decider"
cuda_ok "$ROOT/venvs/decider/bin/python" decider

step "venv: tts (Python 3.12)"
[ -x "$ROOT/venvs/tts/bin/python" ] || uv venv --seed --python 3.12 "$ROOT/venvs/tts"
uv pip install -p "$ROOT/venvs/tts/bin/python" torch --index-url "$TORCH_INDEX"
uv pip install -p "$ROOT/venvs/tts/bin/python" qwen-tts fastapi "uvicorn[standard]" scipy
"$ROOT/venvs/tts/bin/python" -c "import qwen_tts"
cuda_ok "$ROOT/venvs/tts/bin/python" tts

step "done -- setup took $(( $(date +%s) - T0 ))s. Next: bash run.sh"
