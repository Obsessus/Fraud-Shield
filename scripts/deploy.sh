#!/usr/bin/env bash
# Deploy the Fraud Intelligence API on a single VM (Stage 16).
#
# Prerequisites: docker + docker compose installed; the champion model must be present
# under ./data/artifacts (run `dvc pull`, or copy from a trained run, before deploying).
#
# This rebuilds the image from the current checkout and (re)starts the service in the
# background. The compose file pins resource limits, a healthcheck, and a read-only rootfs.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Building image..."
docker compose build

echo "Starting service..."
docker compose up -d

echo "API is starting. Health check: http://localhost:8000/health"
echo "Tail logs with: docker compose logs -f api"
