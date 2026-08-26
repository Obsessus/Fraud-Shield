# Render deployment
#
# Inherits the prebuilt application image, which already contains the trained
# model + UI demo data. Nothing is copied at build time, so no dataset or
# large artifacts ever live in this repo.
#
# Prerequisite: the base image must be public on Docker Hub:
#   docker tag  fraud-intel-api:latest 20mahfuz05/fraud-intel-api:latest
#   docker push 20mahfuz05/fraud-intel-api:latest   # set repo PUBLIC on hub.docker.com
#
# Render sets the PORT env var; the container honors it (see app.py / Dockerfile).
FROM docker.io/20mahfuz05/fraud-intel-api:latest
