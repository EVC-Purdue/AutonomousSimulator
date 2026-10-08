#!/bin/bash
# Runs the AutonomousSimulator web image on the mini PC: bash deploy.sh <image_ref>
set -euo pipefail
IMAGE_REF=$1

NAME=autonomoussim
URL=http://127.0.0.1:5060/
RESULTS_PATH="$HOME/autonomoussim/results"
SECRETS_PATH="$HOME/.autonomoussim-secrets"
AK_PATH="$HOME/AutonomousKart"

echo "Checking host paths..."
mkdir -p "$RESULTS_PATH"
for p in "$SECRETS_PATH/env" "$SECRETS_PATH/ssh" "$AK_PATH"; do
  if [ ! -e "$p" ]; then
    echo "Missing $p (see the Deploy section of README.md)"
    exit 1
  fi
done
if ! grep -q '^AUTONOMOUS_SIM_PASSWORD=.' "$SECRETS_PATH/env"; then
  echo "AUTONOMOUS_SIM_PASSWORD is not set in $SECRETS_PATH/env"
  exit 1
fi
HF_UPLOAD=1
grep -q '^HF_TOKEN=.' "$SECRETS_PATH/env" || HF_UPLOAD=0
if [ "$(git -C "$AK_PATH" config --get remote.origin.mirror || true)" != "true" ]; then
  echo "Warning: $AK_PATH is not a mirror clone, so its local branches stay stale after a fetch"
fi

echo "Pulling Docker image: $IMAGE_REF"
docker pull "$IMAGE_REF"

echo "Stopping existing container..."
docker stop "$NAME" || true
docker rm "$NAME" || true

echo "Starting new container..."
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --network host \
  --pid host \
  --env-file "$SECRETS_PATH/env" \
  -e AUTONOMOUS_SIM_RESULTS=/data/results \
  -e AUTONOMOUS_KART_REPO=/repos/AutonomousKart \
  -e AUTONOMOUS_SIM_HF_UPLOAD="$HF_UPLOAD" \
  -v "$RESULTS_PATH:/data/results" \
  -v "$AK_PATH:/repos/AutonomousKart" \
  -v "$SECRETS_PATH/ssh:/root/.ssh" \
  "$IMAGE_REF"

echo "Waiting for $URL..."
for _ in $(seq 30); do
  if curl -fsS -o /dev/null "$URL"; then
    echo "Deployment successful!"
    docker ps --filter name="$NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
    exit 0
  fi
  sleep 2
done
echo "Container did not answer on $URL"
docker logs --tail 200 "$NAME" || true
exit 1
