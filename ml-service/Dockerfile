# Runtime image for the ML service and the Kafka consumer (CPU ONNX Runtime). Training runs outside Docker on a GPU host.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 10001 app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=app:app app.py consumer.py sensor_sim.py ./
COPY --chown=app:app data ./data
COPY --chown=app:app sigproc ./sigproc
COPY --chown=app:app fusion ./fusion
COPY --chown=app:app benchmark ./benchmark
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
