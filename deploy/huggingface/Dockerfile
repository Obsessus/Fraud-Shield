# Fraud Shield — Hugging Face Spaces deployment
#
# The application image already has the trained model + UI demo data baked in.
# We simply inherit that image; the container honors the PORT env var that
# Hugging Face Spaces sets (7860), so no model files need to live in this repo.
#
# Before creating the Space:
#   1. Build/tag the image locally (or reuse the existing fraud-intel-api:latest)
#   2. Push it to a PUBLIC registry, e.g. Docker Hub:
#        docker tag fraud-intel-api:latest obsessus/fraud-intel-api:latest
#        docker push obsessus/fraud-intel-api:latest
#      (or GHCR: ghcr.io/obsessus/fraud-intel-api:latest)
#   3. Create a new HF Space (Docker SDK) and upload THIS folder (Dockerfile + README.md).
#
# If you rename the image/registry, update the FROM line below.
FROM obsessus/fraud-intel-api:latest
