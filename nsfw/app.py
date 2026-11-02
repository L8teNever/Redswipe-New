"""NudeNet-Scanner: Bilder und Videos (jeder einzelne Frame) auf freigelegte Körperteile prüfen.
Bekommt nur Pfade im gemeinsamen Cache-Volume, gibt eine zusammengefasste JSON-Antwort zurück."""
import asyncio
import json
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from nudenet import NudeDetector
from pydantic import BaseModel

MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.25"))
N_DET = int(os.environ.get("DETECTORS", "2"))
ORT_THREADS = int(os.environ.get("ORT_THREADS", "3"))
YOLO_ON = os.environ.get("YOLO", "1") != "0"
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.35"))
YOLO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("YOLO_FILE", "yolo11n.pt"))
POSE_ON = os.environ.get("POSE", "1") != "0"
POSE_CONF = float(os.environ.get("POSE_CONF", "0.65"))
POSE_MAX = int(os.environ.get("POSE_MAX", "3"))
HANDS_ON = os.environ.get("HANDS", "1") != "0"
HANDS_MAX = int(os.environ.get("HANDS_MAX", "4"))
HANDS_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
POSE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")
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


class Pose:
    """MediaPipe Pose Landmarker. Für Videos im VIDEO-Modus (verfolgt Personen von Frame zu Frame),
    für Bilder im IMAGE-Modus. Pro Auftrag eine eigene Instanz."""

    def __init__(self, video):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.video = video
        self.last_ts = -1
        opts = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=POSE_MAX,
            min_pose_detection_confidence=POSE_CONF,
            min_pose_presence_confidence=POSE_CONF,
            min_tracking_confidence=POSE_CONF,
        )
        self.lm = vision.PoseLandmarker.create_from_options(opts)

    def run(self, bgr, t=0.0):
        """Liefert pro erkannter Person 33 Punkte [x, y, Sichtbarkeit] (0..1)."""
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ts = max(int(t * 1000), self.last_ts + 1)
            self.last_ts = ts
            res = self.lm.detect_for_video(img, ts)
        else:
            res = self.lm.detect(img)
        return [[[round(p.x, 3), round(p.y, 3), round(p.visibility or 0.0, 2)] for p in person] for person in res.pose_landmarks]

    def close(self):
        try:
            self.lm.close()
        except Exception:  # noqa: BLE001
            pass


class Hands:
    """MediaPipe Hand Landmarker (21 Punkte pro Hand), gleiche Betriebsarten wie Pose."""

    def __init__(self, video):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.video = video
        self.last_ts = -1
        opts = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=HANDS_MODEL),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=HANDS_MAX,
            min_hand_detection_confidence=POSE_CONF,
            min_hand_presence_confidence=POSE_CONF,
            min_tracking_confidence=POSE_CONF,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)

    def run(self, bgr, t=0.0):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ts = max(int(t * 1000), self.last_ts + 1)
            self.last_ts = ts
            res = self.lm.detect_for_video(img, ts)
        else:
            res = self.lm.detect(img)
        return [[[round(p.x, 3), round(p.y, 3), 1] for p in hand] for hand in res.hand_landmarks]

    def close(self):
        try:
            self.lm.close()
        except Exception:  # noqa: BLE001
            pass


_yolo = None
_yolo_lock = threading.Lock()


def yolo_persons(frames):
    """YOLO (Ultralytics): Personen-Boxen pro Frame, normalisiert [(x1, y1, x2, y2), ...]. None, wenn YOLO aus ist."""
    global _yolo
    if not YOLO_ON:
        return None
    with _yolo_lock:
        if _yolo is None:
            import torch
            from ultralytics import YOLO

            torch.set_num_threads(max(1, ORT_THREADS - 1))
            _yolo = YOLO(YOLO_MODEL)
        res = _yolo.predict(frames, classes=[0], conf=YOLO_CONF, imgsz=640, verbose=False)
    return [[tuple(float(v) for v in b) for b in r.boxes.xyxyn.tolist()] for r in res]


def _in_boxes(pts, boxes, margin):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    for x1, y1, x2, y2 in boxes:
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        if x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my:
            return True
    return False


TORSO_KEYS = ("BREAST", "BELLY", "BUTTOCKS", "GENITALIA", "ANUS")


def anchors_from(labels_boxes):
    """Körperteil-Anker aus NudeNet: (Art, Mitte x, Mitte y) für Rumpf/Füße/Achseln. Eingabe: [(label, (x, y, w, h))]."""
    out = []
    for label, (x, y, w, h) in labels_boxes:
        kind = "torso" if any(k in label for k in TORSO_KEYS) else "feet" if label.startswith("FEET") else "arm" if label.startswith("ARMPITS") else None
        if kind:
            out.append((kind, x + w / 2, y + h / 2))
    return out


def pose_support(pose, anchors):
    """Wie viele NudeNet-Anker passen zum Skelett: Rumpf-Anker liegen im (erweiterten) Rechteck aus
    Schultern/Hüften, Fuß-Anker nahe den Knöcheln/Füßen, Achsel-Anker nahe den Schultern."""

    def region(ids, m):
        pts = [pose[i] for i in ids if i < len(pose) and pose[i][2] >= 0.3]
        if not pts:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        d = max(w, h, 0.05) * m
        return min(xs) - d, min(ys) - d, max(xs) + d, max(ys) + d

    regs = {"torso": region((11, 12, 23, 24), 0.35), "feet": region((27, 28, 29, 30, 31, 32), 0.6), "arm": region((11, 12), 0.6)}
    n = 0
    for kind, cx, cy in anchors:
        r = regs[kind]
        if r and r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
            n += 1
    return n


def limit_by_persons(boxes, poses, hands, anchors=()):
    """Anzahl Personen (YOLO) begrenzt Pose und Hände: Pose nur in Personen-Boxen und höchstens so viele wie Personen,
    Hände nur in/an Personen-Boxen und höchstens 2 pro Person. Ohne Person bleibt nichts übrig."""
    if boxes is None:
        return poses, hands
    n = len(boxes)
    if n == 0:
        return [], []
    poses = [p for p in poses if _in_boxes(p, boxes, 0.1)]
    # Erst Skelette, die zu NudeNet-Körperteilen (Brust, Bauch, Füße ...) passen, dann nach Sichtbarkeit
    poses.sort(key=lambda p: (-pose_support(p, anchors), -sum(q[2] for q in p)))
    hands = [h for h in hands if _in_boxes(h, boxes, 0.25)]
    return poses[:n], hands[: 2 * n]


def face_boxes(items):
    """Gesichts-Boxen (x, y, w, h normalisiert) aus den NudeNet-Treffern."""
    return [tuple(d["box"]) for d in items if str(d["label"]).startswith("FACE_")]


def refine_faces(poses, faces):
    """NudeNet-Gesichter korrigieren die Pose: Gesichtspunkte (0-10) bleiben nur, wenn sie in/an einer
    erkannten Gesichtsbox liegen. Sonst (kein Gesicht erkannt oder Punkte woanders) werden sie ausgeblendet
    (Sichtbarkeit 0), das Frontend zeichnet sie dann nicht."""
    out = []
    for person in poses:
        pts = [list(p) for p in person]
        for i in range(min(11, len(pts))):
            x, y = pts[i][0], pts[i][1]
            ok = any(fx - fw * 0.4 <= x <= fx + fw * 1.4 and fy - fh * 0.4 <= y <= fy + fh * 1.4 for fx, fy, fw, fh in faces)
            if not ok:
                pts[i][2] = 0.0
        out.append(pts)
    return out


class ScanCancelled(Exception):
    """Der Aufrufer hat die Verbindung geschlossen (Beitrag zu weit zurück): Scan abbrechen."""


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
    out = {"kind": "image", "width": w, "height": h, "detections": items}
    boxes = (yolo_persons([img]) or [None])[0]
    poses, hs = [], []
    if POSE_ON:
        pose = Pose(False)
        try:
            poses = pose.run(img)
        finally:
            pose.close()
    if HANDS_ON:
        hands = Hands(False)
        try:
            hs = hands.run(img)
        finally:
            hands.close()
    poses, hs = limit_by_persons(boxes, poses, hs, anchors_from((d["label"], d["box"]) for d in items))
    poses = refine_faces(poses, face_boxes(items))
    if POSE_ON:
        out["poses"] = poses
    if HANDS_ON:
        out["hands"] = hs
    if boxes is not None:
        out["persons"] = len(boxes)
    return out


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


def scan_video(det, path, cancel=None):
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
    pose_tl = []  # pro Frame mit Personen: [Zeit, [Person(33 x [x, y, Sichtbarkeit]), ...]]
    pose = Pose(True) if POSE_ON else None
    hand_tl = []
    hands = Hands(True) if HANDS_ON else None
    pose_ex = ThreadPoolExecutor(1) if (pose or hands or YOLO_ON) else None
    person_max, person_frames = 0, 0  # läuft parallel zu NudeNet, bleibt in Frame-Reihenfolge
    total = 0
    batch, idx = [], []

    def pose_batch(frames, ids):
        boxes = yolo_persons(frames)
        out = []
        for k, (f, i) in enumerate(zip(frames, ids)):
            b = boxes[k] if boxes else None
            out.append((i / fps, b, pose.run(f, i / fps) if pose else [], hands.run(f, i / fps) if hands else []))
        return out

    def flush():
        nonlocal batch, idx, person_max, person_frames
        if not batch:
            return
        fut = pose_ex.submit(pose_batch, batch, idx) if pose_ex else None
        frame_faces, frame_anchors = [], []
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
            frame_faces.append([tuple(r[2:6]) for r in row if names[r[0]].startswith("FACE_")])
            frame_anchors.append(anchors_from((names[r[0]], tuple(r[2:6])) for r in row))
        if fut:
            for (t, b, people, hs), ff, fa in zip(fut.result(), frame_faces, frame_anchors):
                people, hs = limit_by_persons(b, people, hs, fa)
                npers = len(b) if b is not None else None
                people = refine_faces(people, ff)
                if npers:
                    person_frames += 1
                    person_max = max(person_max, npers)
                if people:
                    pose_tl.append([round(t, 3), people])
                if hs:
                    hand_tl.append([round(t, 3), hs])
        batch, idx = [], []

    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise ScanCancelled()
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
        if pose_ex:
            pose_ex.shutdown(wait=True)
        if pose:
            pose.close()
        if hands:
            hands.close()
    if total == 0:
        raise ValueError("Keine Frames im Video")
    out = []
    for e in labels.values():
        out.append({"label": e["label"], "score": round(e["max"], 3), "frames": e["frames"], "share": round(e["frames"] / total, 4),
                    "first": round(e["first"], 2), "last": round(e["last"], 2), "at": round(e["at"], 2), "box": e["box"]})
    out.sort(key=lambda d: -d["score"])
    res = {"kind": "video", "width": w, "height": h, "frames": total, "fps": round(fps, 3),
            "duration": round(duration or total / fps, 2), "detections": out,
            "names": names, "timeline": timeline}
    if pose:
        res["pose_timeline"] = pose_tl
        res["pose_frames"] = len(pose_tl)
    if YOLO_ON:
        res["persons_max"] = person_max
        res["person_frames"] = person_frames
    if hands:
        res["hand_timeline"] = hand_tl
        res["hand_frames"] = len(hand_tl)
    return res


def work(req: Req, cancel=None):
    path = os.path.realpath(req.path)
    if not path.startswith(ROOT + "/"):
        raise HTTPException(400, "Pfad nicht erlaubt")
    det = pool.get()
    try:
        return scan_image(det, path) if req.kind == "image" else scan_video(det, path, cancel)
    except HTTPException:
        raise
    except ScanCancelled:
        raise HTTPException(499, "Abgebrochen")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Scan fehlgeschlagen: {e}")
    finally:
        pool.put(det)


@app.post("/scan")
async def scan(req: Req, request: Request):
    if req.kind not in ("image", "video"):
        raise HTTPException(400, "Unbekannte Art")
    cancel = threading.Event()

    async def watch():
        while True:
            await asyncio.sleep(0.5)
            if await request.is_disconnected():
                cancel.set()
                return

    task = asyncio.create_task(watch())
    try:
        return await run_in_threadpool(work, req, cancel)
    finally:
        task.cancel()
