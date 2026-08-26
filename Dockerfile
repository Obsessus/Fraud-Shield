# Fraud Intelligence Platform — inference service image
# Lean runtime: code + the champion model artifact. Heavy train-time tooling (mlflow,
# dvc, evidently) is intentionally NOT installed here (see D23).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIP_DATA_DIR=/app/data

WORKDIR /app

# Copy project metadata + source, then install runtime extras.
# (Editable install keeps __file__ under /app/src so data.paths resolves PROJECT_ROOT.)
COPY pyproject.toml ./
COPY src ./src
COPY data/artifacts ./data/artifacts

RUN pip install --upgrade pip && \
    pip install -e ".[base,ml,serving]"

# Drop root privileges.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "fraudintel.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
