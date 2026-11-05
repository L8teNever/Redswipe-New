FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl libegl1 libgles2 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir nudenet mediapipe ultralytics fastapi "uvicorn[standard]"
WORKDIR /app
# Pose Landmarker (lite | full | heavy)
ARG POSE_MODEL=full
RUN curl -fsSL -o /app/pose_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${POSE_MODEL}/float16/latest/pose_landmarker_${POSE_MODEL}.task"
RUN curl -fsSL -o /app/hand_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
RUN curl -fsSL -o /app/holistic_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
RUN curl -fsSL -o /app/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
RUN curl -fsSL -o /app/face_parsing.onnx https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx
RUN curl -fsSL -o /tmp/b.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
 && python -c "import zipfile;zipfile.ZipFile('/tmp/b.zip').extract('genderage.onnx','/app')" && rm /tmp/b.zip
RUN curl -fsSL -o /app/yolo11s-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt
COPY app.py .
ENV ORT_THREADS=3 DETECTORS=2 MIN_SCORE=0.25 POSE_MAX=3 POSE=1 YOLO=1 HANDS=1 HANDS_MAX=4 HOLISTIC=0 COLORS=1 COLORS_EVERY=12
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
