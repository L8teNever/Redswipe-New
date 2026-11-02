"""NudeNet-Scanner: Bilder und Videos (jeder einzelne Frame) auf freigelegte Körperteile prüfen.
Bekommt nur Pfade im gemeinsamen Cache-Volume, gibt eine zusammengefasste JSON-Antwort zurück."""
import json
import os
import queue
import subprocess
import threading

import numpy as np
import onnxruntime
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from nudenet import NudeDetector
from pydantic import BaseModel

MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.25"))
N_DET = int(os.environ.get("DETECTORS", "2"))
ORT_THREADS = int(os.environ.get("ORT_THREADS", "3"))
MAX_WIDTH = 640  # Frames vor der Erkennung verkleinern (das Modell rechnet ohnehin mit 320 px)
BATCH = 8
ROOT = "/cache"

# Mehrere Detektoren, damit zwei Scans parallel laufen können.
_orig = onnxruntime.InferenceSession


def _session(*a, **kw):
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    kw["sess_options"] = so
    return _orig(*a, **kw)


onnxruntime.InferenceSession = _session
pool: "queue.Queue[NudeDetector]" = queue.Queue()
for _ in range(N_DET):
    pool.put(NudeDetector())

app = FastAPI()


class Req(BaseModel):
    path: str
    kind: str  # "image" | "video"


@app.get("/healthz")
def healthz():
    return "ok"


def norm(det, w, h):
    x, y, bw, bh = det["box"]
    return {
        "label": det["class"],
        "score": round(float(det["score"]), 3),
        "box": [round(max(0, x) / w, 4), round(max(0, y) / h, 4), round(bw / w, 4), round(bh / h, 4)],
    }


def detect_many(det, frames):
    """Frames (BGR-Arrays) stapelweise erkennen; fällt bei Fehlern auf Einzelbilder zurück."""
    try:
        return det.detect_batch(frames, batch_size=BATCH)
    except Exception:
        return [det.detect(f) for f in frames]


def scan_image(det, path):
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise ValueError("Bild nicht lesbar")
    h, w = img.shape[:2]
    res = det.detect(img)
    items = [norm(d, w, h) for d in res if d["score"] >= MIN_SCORE]
    items.sort(key=lambda d: -d["score"])
    return {"kind": "image", "width": w, "height": h, "detections": items}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames:format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=60,
    )
    j = json.loads(out.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return int(s["width"]), int(s["height"]), fps or 30.0, float((j.get("format") or {}).get("duration") or 0)


def scan_video(det, path):
    w, h, fps, duration = probe(path)
    if w > MAX_WIDTH:
        h = int(h * MAX_WIDTH / w) // 2 * 2
        w = MAX_WIDTH
    w, h = max(2, w // 2 * 2), max(2, h // 2 * 2)
    size = w * h * 3
    # Kein fps-Filter: wirklich jeder Frame wird geprüft.
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    labels = {}
    timeline = []  # pro Frame mit Treffern: [Zeit, [[Label-Nr, Score, x, y, b, h], ...]]
    names = []
    total = 0
    batch, idx = [], []

    def flush():
        nonlocal batch, idx
        if not batch:
            return
        for i, dets in zip(idx, detect_many(det, batch)):
            t = i / fps
            seen = set()
            row = []
            for d in dets:
                if d["score"] < MIN_SCORE:
                    continue
                n = norm(d, w, h)
                if n["label"] not in names:
                    names.append(n["label"])
                row.append([names.index(n["label"]), n["score"]] + [round(v, 3) for v in n["box"]])
                e = labels.setdefault(n["label"], {"label": n["label"], "max": 0.0, "frames": 0, "first": t, "last": t, "box": n["box"], "at": t})
                if n["score"] > e["max"]:
                    e["max"], e["box"], e["at"] = n["score"], n["box"], t
                if n["label"] not in seen:
                    e["frames"] += 1
                    seen.add(n["label"])
                e["last"] = t
            if row:
                timeline.append([round(t, 3), row])
        batch, idx = [], []

    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            batch.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy())
            idx.append(total)
            total += 1
            if len(batch) >= BATCH:
                flush()
        flush()
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if total == 0:
        raise ValueError("Keine Frames im Video")
    out = []
    for e in labels.values():
        out.append({"label": e["label"], "score": round(e["max"], 3), "frames": e["frames"], "share": round(e["frames"] / total, 4),
                    "first": round(e["first"], 2), "last": round(e["last"], 2), "at": round(e["at"], 2), "box": e["box"]})
    out.sort(key=lambda d: -d["score"])
    return {"kind": "video", "width": w, "height": h, "frames": total, "fps": round(fps, 3),
            "duration": round(duration or total / fps, 2), "detections": out,
            "names": names, "timeline": timeline}


def work(req: Req):
    path = os.path.realpath(req.path)
    if not path.startswith(ROOT + "/"):
        raise HTTPException(400, "Pfad nicht erlaubt")
    det = pool.get()
    try:
        return scan_image(det, path) if req.kind == "image" else scan_video(det, path)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Scan fehlgeschlagen: {e}")
    finally:
        pool.put(det)


@app.post("/scan")
async def scan(req: Req):
    if req.kind not in ("image", "video"):
        raise HTTPException(400, "Unbekannte Art")
    return await run_in_threadpool(work, req)
