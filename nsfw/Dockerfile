FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir nudenet mediapipe fastapi "uvicorn[standard]"
WORKDIR /app
# Pose Landmarker (lite | full | heavy)
ARG POSE_MODEL=full
RUN curl -fsSL -o /app/pose_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${POSE_MODEL}/float16/latest/pose_landmarker_${POSE_MODEL}.task"
COPY app.py .
ENV ORT_THREADS=3 DETECTORS=2 MIN_SCORE=0.25 POSE_MAX=3 POSE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
