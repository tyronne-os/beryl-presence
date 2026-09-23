#!/usr/bin/env bash
# Start Decider + TTS, prove she talks (selftest -> $ROOT/out/selftest_hello.mp4), then bring up the live server.
# Every stage prints its measured seconds -- together with setup_gpu.sh that is your real download-to-hello time.
#
#   BERYL_AVATAR_IMAGE=/path/to/beryl.png BRAIN_API_KEY=nvapi-... bash run.sh
#
# BERYL_AVATAR_IMAGE  square, face-centred portrait (default: FlashHead's example image)
# BRAIN_API_KEY       key for the offloaded brain (default endpoint: NVIDIA NIM). Use a key scoped to this box,
#                     not the director's personal nvapi- key. Without it Beryl only echoes what she heard.
# BRAIN_MODEL         any model id your endpoint serves (default meta/llama-3.1-405b-instruct; a Nemotron id
#                     answers faster) -- check the id exists on build.nvidia.com before relying on it
# FORCE_LIVE=1        start the live server even if the selftest says she is not talking
set -euo pipefail

ROOT="${BERYL_ROOT:-/opt/beryl}"
APP="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs"
RP="$ROOT/venvs/render/bin/python"
T0=$(date +%s)
step() { printf '\n== [%5ss] %s\n' "$(( $(date +%s) - T0 ))" "$*"; }
mkdir -p "$LOG" "$ROOT/out"

wait_ok() {  # name url timeout_s logfile pattern
  local t=$(date +%s)
  until curl -sf "$2" 2>/dev/null | grep -q "$5"; do
    if (( $(date +%s) - t > $3 )); then echo "!! $1 not healthy after $3s -- last log lines:"; tail -n 30 "$4"; exit 1; fi
    sleep 2
  done
  echo "   $1 healthy after $(( $(date +%s) - t ))s"
}

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
FP8=$(awk -v c="$CC" 'BEGIN { print (c >= 8.9) ? 1 : 0 }')   # FP8 linears need Ada/Hopper: on for L4, off for A100
pkill -f "decider.serve:app" 2>/dev/null || true
pkill -f "tts_server:app" 2>/dev/null || true
pkill -f "beryl.orchestrator" 2>/dev/null || true
sleep 1

step "decider (compute capability $CC, FP8=$FP8)"
( cd "$ROOT/repos/decider" && DECIDER_MODEL="$ROOT/models/decider-2b" DECIDER_FP8=$FP8 \
    DECIDER_COMPILE="${DECIDER_COMPILE:-1}" DECIDER_WARM_BATCHES=1,2,4 \
    nohup "$ROOT/venvs/decider/bin/uvicorn" decider.serve:app --host 127.0.0.1 --port 8000 > "$LOG/decider.log" 2>&1 & )

step "tts (Qwen3-TTS 0.6B)"
( cd "$APP" && TTS_MODEL="$ROOT/models/qwen3-tts-0.6b" \
    nohup "$ROOT/venvs/tts/bin/uvicorn" tts_server:app --host 127.0.0.1 --port 8010 > "$LOG/tts.log" 2>&1 & )

wait_ok decider http://127.0.0.1:8000/health 900 "$LOG/decider.log" '"ok":true'
wait_ok tts     http://127.0.0.1:8010/health 600 "$LOG/tts.log" '"ok":true'

# faster-whisper (CTranslate2) needs the cuBLAS/cuDNN that pip installed inside the render venv
NVLIB=$("$RP" -c 'import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; print(a.__path__[0] + ":" + b.__path__[0])' 2>/dev/null || true)
export LD_LIBRARY_PATH="${NVLIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BERYL_ROOT="$ROOT" BERYL_OUT="$ROOT/out" STT_MODEL="$ROOT/models/faster-distil-whisper-large-v3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step "selftest: does she actually talk?"
cd "$APP"
set +e
"$RP" -m beryl.orchestrator --selftest 2>&1 | tee "$LOG/selftest.log"
rc=${PIPESTATUS[0]}
set -e
if (( rc != 0 )) && [ "${FORCE_LIVE:-0}" != "1" ]; then
  echo "!! Selftest says NOT TALKING. Watch $ROOT/out/selftest_hello.mp4, read the reasons above, fix, re-run."
  echo "   (FORCE_LIVE=1 bash run.sh starts the live server anyway.)"
  exit "$rc"
fi

step "live server"
nohup "$RP" -m beryl.orchestrator --serve > "$LOG/orchestrator.log" 2>&1 &
wait_ok orchestrator http://127.0.0.1:7860/health 900 "$LOG/orchestrator.log" '"ready":true'

step "READY -- total $(( $(date +%s) - T0 ))s from start to live"
cat <<EOF

  Watch her from your laptop (tunnel -- no firewall rule, nothing exposed publicly):
    gcloud compute ssh beryl-tier-small --zone=\$GCP_ZONE -- -L 7860:127.0.0.1:7860
  then open http://localhost:7860 in Chrome, click Connect, put headphones on, turn the mic on.

  Live numbers:   curl -s localhost:7860/health | python3 -m json.tool
  Logs:           $LOG/{decider,tts,orchestrator,selftest}.log
  Hello video:    $ROOT/out/selftest_hello.mp4
EOF
