#!/usr/bin/env bash
# Run on your laptop or in Cloud Shell, from the repo root. Creates (or restarts) the spot GPU box, copies this
# service onto it, and prints the two commands to run over SSH. Replaces 02_launch_small.sh, whose hidden startup
# script pulled a model id that does not exist (SoulX-FlashHead-1.3B; the real repo is SoulX-FlashHead-1_3B).
#
#   GCP_PROJECT=my-proj GCP_ZONE=us-central1-a bash gcp_launch.sh             L4 24GB  (tonight's tier)
#   TIER=a100 GCP_PROJECT=my-proj GCP_ZONE=us-central1-a bash gcp_launch.sh   A100 40GB, if the selftest says the
#                                                                              L4 renders slower than real time
# Image families change over time; list current ones with:
#   gcloud compute images list --project deeplearning-platform-release --format="value(family)" | grep common-cu12 | sort -u
set -euo pipefail
: "${GCP_PROJECT:?export GCP_PROJECT first}"
: "${GCP_ZONE:?export GCP_ZONE first}"
TIER="${TIER:-l4}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu124-debian-11}"
case "$TIER" in
  l4)   NAME="${NAME:-beryl-tier-small}"; MACHINE=(--machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1) ;;
  a100) NAME="${NAME:-beryl-tier-mid}";   MACHINE=(--machine-type=a2-highgpu-1g) ;;
  *)    echo "TIER must be l4 or a100"; exit 1 ;;
esac
G=(--project "$GCP_PROJECT" --zone "$GCP_ZONE")

status=$(gcloud compute instances describe "$NAME" "${G[@]}" --format="value(status)" 2>/dev/null || echo NONE)
if [ "$status" = NONE ]; then
  gcloud compute instances create "$NAME" "${G[@]}" "${MACHINE[@]}" \
    --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release \
    --boot-disk-size=150GB --boot-disk-type=pd-ssd \
    --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
    --metadata=install-nvidia-driver=True
elif [ "$status" != RUNNING ]; then
  gcloud compute instances start "$NAME" "${G[@]}"
fi

echo "== waiting for SSH on $NAME (first boot installs the NVIDIA driver, ~2-5 min)"
until gcloud compute ssh "$NAME" "${G[@]}" --command "nvidia-smi -L" --quiet 2>/dev/null; do sleep 15; done

echo "== copying beryl-live"
gcloud compute ssh "$NAME" "${G[@]}" --command "rm -rf ~/beryl-live" --quiet
gcloud compute scp --recurse "$(cd "$(dirname "$0")" && pwd)" "$NAME":~/beryl-live "${G[@]}" --quiet

cat <<EOF

== $NAME is up. Now:
  gcloud compute scp /path/to/beryl.png $NAME:~/beryl.png --project $GCP_PROJECT --zone $GCP_ZONE
  gcloud compute ssh $NAME --project $GCP_PROJECT --zone $GCP_ZONE -- -L 7860:127.0.0.1:7860
  cd ~/beryl-live && bash setup_gpu.sh          # once: ~16.5 GB of weights + 3 venvs, prints its own timing
  BERYL_AVATAR_IMAGE=~/beryl.png BRAIN_API_KEY=... bash run.sh
Then http://localhost:7860 in Chrome. Spot VMs can be preempted -- they stop, not delete; start it again and re-run run.sh.
When you're done tonight:  gcloud compute instances stop $NAME --project $GCP_PROJECT --zone $GCP_ZONE
EOF
