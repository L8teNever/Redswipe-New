FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir nudenet fastapi "uvicorn[standard]"
WORKDIR /app
COPY app.py .
ENV ORT_THREADS=3 DETECTORS=2 MIN_SCORE=0.25
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
