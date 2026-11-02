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
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "960"))  # Bilder; Video-Frames bleiben bei 640
NIP_PREC = int(os.environ.get("NIP_PREC", "2"))  # Brustwarzen-Suche: 1 = schnell, 2 = genau
YOLO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("YOLO_FILE", "yolo11n.pt"))
POSE_ON = os.environ.get("POSE", "1") != "0"
POSE_CONF = float(os.environ.get("POSE_CONF", "0.65"))
POSE_MAX = int(os.environ.get("POSE_MAX", "6"))
POSE_CROP_CONF = float(os.environ.get("POSE_CROP_CONF", "0.35"))
HANDS_ON = os.environ.get("HANDS", "1") != "0"
HANDS_MAX = int(os.environ.get("HANDS_MAX", "4"))
HANDS_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
POSE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")
HOLISTIC_ON = os.environ.get("HOLISTIC", "0") != "0"
HOLISTIC_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holistic_landmarker.task")
COLORS_ON = os.environ.get("COLORS", "1") != "0"
COLORS_EVERY = int(os.environ.get("COLORS_EVERY", "12"))  # Video: nur jeden n-ten Frame auf Haar-/Hautfarbe prüfen
PARSE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_parsing.onnx")
AGE_ON = os.environ.get("AGE", "1") != "0"
AGE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genderage.onnx")
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

    def __init__(self, video, conf=None, num=None):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.video = video
        self.last_ts = -1
        conf = POSE_CONF if conf is None else conf
        opts = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=POSE_MAX if num is None else num,
            min_pose_detection_confidence=conf,
            min_pose_presence_confidence=conf,
            min_tracking_confidence=conf,
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


class Holistic:
    """MediaPipe Holistic Landmarker: Ermittelt Pose (33 Punkte), Gesicht und Hände in einem Modell."""

    def __init__(self, video):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.video = video
        self.last_ts = -1
        opts = vision.HolisticLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=HOLISTIC_MODEL),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            min_pose_detection_confidence=POSE_CONF,
            min_pose_landmarks_confidence=POSE_CONF,
            min_hand_landmarks_confidence=POSE_CONF,
            min_face_detection_confidence=POSE_CONF,
            min_face_landmarks_confidence=POSE_CONF,
        )
        self.lm = vision.HolisticLandmarker.create_from_options(opts)

    def run(self, bgr, t=0.0):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ts = max(int(t * 1000), self.last_ts + 1)
            self.last_ts = ts
            res = self.lm.detect_for_video(img, ts)
        else:
            res = self.lm.detect(img)
        poses = []
        if res.pose_landmarks:
            poses.append([[round(p.x, 3), round(p.y, 3), round(p.visibility or 0.0, 2)] for p in res.pose_landmarks])
        hands = []
        if res.left_hand_landmarks:
            hands.append([[round(p.x, 3), round(p.y, 3), 1.0] for p in res.left_hand_landmarks])
        if res.right_hand_landmarks:
            hands.append([[round(p.x, 3), round(p.y, 3), 1.0] for p in res.right_hand_landmarks])
        return poses, hands

    def close(self):
        try:
            self.lm.close()
        except Exception:  # noqa: BLE001
            pass


_yolo = None
_yolo_lock = threading.Lock()


def yolo_persons(frames, imgsz=640):
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
        res = _yolo.predict(frames, classes=[0], conf=YOLO_CONF, iou=0.6, imgsz=imgsz, verbose=False)
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


CROP_PAD = 0.18
CROP_MIN = 72  # kleinere Ausschnitte (Pixel) bringen nichts


def _crop_rect(box, W, H, pad=CROP_PAD):
    """Pixel-Ausschnitt (x0, y0, x1, y1) um eine normalisierte Personen-Box (x1, y1, x2, y2) mit Rand."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0, int((x1 - dx) * W)), max(0, int((y1 - dy) * H)), min(W, int((x2 + dx) * W)), min(H, int((y2 + dy) * H)))


def _pose_center(p):
    pts = [q for q in p if q[2] >= 0.3] or p
    return sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts)


def uncovered_boxes(boxes, poses):
    """Personen-Boxen (größte zuerst), denen noch kein Skelett zugeordnet ist.
    Jedes Skelett gehört zur kleinsten Box, die seinen Mittelpunkt enthält."""
    free = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    for p in poses:
        cx, cy = _pose_center(p)
        hit = [b for b in free if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
        if hit:
            free.remove(min(hit, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
    return free


def crop_poses(pose_crop, bgr, boxes, poses):
    """Mehrere Personen: Für jede Person ohne Skelett wird die Pose einzeln auf dem Personen-Ausschnitt gesucht
    (das Modell sieht sonst oft nur die größte Person). Liefert die ergänzten Skelette (Gesamtbild-Koordinaten)."""
    if pose_crop is None or not boxes or len(boxes) < 2:
        return poses
    H, W = bgr.shape[:2]
    out = list(poses)
    for box in uncovered_boxes(boxes, out):
        x0, y0, x1, y1 = _crop_rect(box, W, H)
        cw, ch = x1 - x0, y1 - y0
        if cw < CROP_MIN or ch < CROP_MIN:
            continue
        found = pose_crop.run(np.ascontiguousarray(bgr[y0:y1, x0:x1]))
        best, best_s = None, 0.0
        for p in found:
            q = [[round((x0 + a * cw) / W, 3), round((y0 + b * ch) / H, 3), c] for a, b, c in p]
            cx, cy = _pose_center(q)
            s = sum(v[2] for v in q)
            if box[0] <= cx <= box[2] and box[1] <= cy <= box[3] and s > best_s:
                best, best_s = q, s
        if best is None:
            continue
        # Doppelte vermeiden: Skelett mit fast gleicher Mitte wie ein vorhandenes
        bx, by = _pose_center(best)
        tol = max(box[2] - box[0], box[3] - box[1]) * 0.12
        if any(abs(bx - _pose_center(p)[0]) < tol and abs(by - _pose_center(p)[1]) < tol for p in out):
            continue
        out.append(best)
    return out


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def detect_person_crops(det, bgr, boxes, items, W, H):
    """Mehrere Personen: NudeNet zusätzlich auf jedem Personen-Ausschnitt. Kleine Körperteile der Nebenpersonen
    werden so größer und damit erkannt. Neue Treffer (anderes Label oder < 40 % Überlappung) kommen dazu."""
    if not boxes or len(boxes) < 2:
        return items
    out = list(items)
    for box in boxes:
        x0, y0, x1, y1 = _crop_rect(box, W, H)
        cw, ch = x1 - x0, y1 - y0
        if cw < CROP_MIN or ch < CROP_MIN:
            continue
        try:
            res = det.detect(np.ascontiguousarray(bgr[y0:y1, x0:x1]))
        except Exception:  # noqa: BLE001
            continue
        for d in res:
            if d["score"] < MIN_SCORE:
                continue
            bx, by, bw, bh = d["box"]
            full = norm({"class": d["class"], "score": d["score"], "box": [bx + x0, by + y0, bw, bh]}, W, H)
            if any(o["label"] == full["label"] and _iou(o["box"], full["box"]) > 0.4 for o in out):
                continue
            out.append(full)
    out.sort(key=lambda d: -d["score"])
    return out


def face_boxes(items):
    """Gesichts-Boxen (x, y, w, h normalisiert) aus den NudeNet-Treffern."""
    return [tuple(d["box"]) for d in items if str(d["label"]).startswith("FACE_")]


def torso_boxes(items):
    """Rumpf-Boxen (Brust, Bauch, Po, Genitalien) aus den NudeNet-Treffern als (x, y, w, h)."""
    return [tuple(d["box"]) for d in items if any(k in str(d["label"]) for k in TORSO_KEYS)]


def feet_boxes(items):
    """Fuß-Boxen aus den NudeNet-Treffern als (x, y, w, h)."""
    return [tuple(d["box"]) for d in items if str(d["label"]).startswith("FEET_")]


def _hit(x, y, boxes, m=0.0):
    return any(bx - bw * m <= x <= bx + bw * (1 + m) and by - bh * m <= y <= by + bh * (1 + m) for bx, by, bw, bh in boxes)


def reconcile(poses, hands, items):
    """Gegenseitiges Abgleichen aller Detektoren (NudeNet, Pose, Hände, YOLO):
    1. Gesichtspunkte (0-10):
       - Dürfen niemals auf Rumpf- (Brust/Bauch/Po/Genitalien) oder Fuß-Boxen liegen!
       - Liegen bei aufrechter Haltung immer über den Schultern.
       - Wenn NudeNet Gesichter erkannt hat: Gesichtspunkte müssen in einer Gesichtsbox liegen
         und werden dann bestätigt (erhöhte Sichtbarkeit); Punkte woanders werden verworfen.
       - Wenn NudeNet kein Gesicht erkannt hat: Liegt der Rumpf weit oben (Kopf abgeschnitten)
         oder sind die Punkte uneindeutig, werden verdächtige Gesichtspunkte ausgeblendet.
    2. Hände & Handgelenke:
       - Das präzise Hand-Tracking von MediaPipe Hands richtet Handgelenke (15, 16) und
         Fingerspitzen (17-22) der Pose exakt aus.
    3. Rumpf & Skelette:
       - Skelette, die den erkannten NudeNet-Körperteilen widersprechen oder keine Stütze haben,
         werden aussortiert."""
    fb = face_boxes(items)
    tb = torso_boxes(items)
    ftb = feet_boxes(items)
    out = []
    for person in poses:
        pts = [list(p) for p in person]

        # 1. Gesichtspunkte anatomisch und gegen NudeNet abgleichen
        sh = [pts[i] for i in (11, 12) if i < len(pts) and pts[i][2] >= 0.3]
        hips = [pts[i] for i in (23, 24) if i < len(pts) and pts[i][2] >= 0.3]
        sy = sum(q[1] for q in sh) / len(sh) if sh else None
        hy = sum(q[1] for q in hips) / len(hips) if hips else None

        for i in range(min(11, len(pts))):
            x, y = pts[i][0], pts[i][1]

            # Ein Gesichtspunkt kann NIE auf Brust/Bauch/Po/Genitalien oder Füßen liegen
            if _hit(x, y, tb, m=0.03) or _hit(x, y, ftb, m=0.05):
                pts[i][2] = 0.0
                continue

            # Aufrechte Haltung: Schultern über Hüften -> Kopf muss über den Schultern sein
            if sy is not None:
                if hy is None or sy < hy:  # Aufrecht: y nimmt nach unten zu
                    if y > sy + 0.02:  # Liegt tiefer als die Schultern -> kein Gesicht
                        pts[i][2] = 0.0
                        continue
                elif sy > hy:  # Kopfstand / invertiert
                    if y < sy - 0.02:
                        pts[i][2] = 0.0
                        continue

            # NudeNet Gesichts-Abgleich
            if fb:
                in_face = any(fx - fw * 0.35 <= x <= fx + fw * 1.35 and fy - fh * 0.35 <= y <= fy + fh * 1.35 for fx, fy, fw, fh in fb)
                if not in_face:
                    pts[i][2] = 0.0
                else:
                    pts[i][2] = max(pts[i][2], 0.9)
            elif tb:
                min_torso_y = min(b[1] for b in tb)
                if min_torso_y < 0.15 and y > min_torso_y - 0.05:
                    pts[i][2] = 0.0

        # 2. Handgelenke und Finger an die präzise Hand-Erkennung anbinden
        # 15: links (17, 19, 21), 16: rechts (18, 20, 22)
        for wi, fingers in ((15, (17, 19, 21)), (16, (18, 20, 22))):
            if wi >= len(pts):
                continue
            best, bd = None, 0.12
            for hd in hands:
                d = ((hd[0][0] - pts[wi][0]) ** 2 + ((hd[0][1] - pts[wi][1]) ** 2)) ** 0.5
                if d < bd:
                    best, bd = hd, d
            if best:
                pts[wi][0], pts[wi][1] = best[0][0], best[0][1]
                pts[wi][2] = max(pts[wi][2], 0.95)
                if len(best) >= 21:
                    pinky_i, index_i, thumb_i = fingers
                    if pinky_i < len(pts):
                        pts[pinky_i] = [best[20][0], best[20][1], max(pts[pinky_i][2], 0.9)]
                    if index_i < len(pts):
                        pts[index_i] = [best[8][0], best[8][1], max(pts[index_i][2], 0.9)]
                    if thumb_i < len(pts):
                        pts[thumb_i] = [best[4][0], best[4][1], max(pts[thumb_i][2], 0.9)]

        out.append(pts)

    # 3. Skelette mit Rumpf-Ankern abgleichen
    anchors = [a for a in anchors_from((d["label"], d["box"]) for d in items) if a[0] == "torso"]
    if anchors and len(out) > 1:
        sup = [pose_support(p, anchors) for p in out]
        if max(sup) > 0:
            out = [p for p, n in zip(out, sup) if n > 0]
    return out


def smooth_poses(tl, fps):
    """Zeitliche Glättung und Entprellung der Skelette für Videos:
    - Verfolgt Personen stabil über Frames hinweg (Track-ID).
    - Überbrückt kurze Erkennungslücken (1-4 Frames) durch lineare Interpolation.
    - Entfernt plötzliche 1-Frame-Ausreißer/Zucker (Outlier-Spikes).
    - Wendet gewichtete Gauß-Glättung auf Koordinaten und Sichtbarkeiten an,
      sodass die Pose nicht mehr flattert oder springt."""
    if len(tl) < 3:
        return
    gap = max(3, int(fps * 0.25))  # bis zu ~250ms Lücken überbrücken
    frame_map = {round(t * fps): ei for ei, (t, _) in enumerate(tl)}
    all_frames = sorted(frame_map.keys())
    if not all_frames:
        return

    def center(p):
        pts = [p[i] for i in (11, 12, 23, 24) if i < len(p) and p[i][2] >= 0.3] or [q for q in p if q[2] >= 0.3]
        if not pts:
            return None
        return sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts)

    # 1. Tracks aufbauen
    tracks = []  # {"last": (x, y), "end": Frame-Nr, "items": [(fn, person_data)]}
    for fn in all_frames:
        ei = frame_map[fn]
        t, people = tl[ei]
        used = set()
        for pi, p in enumerate(people):
            c = center(p)
            if c is None:
                continue
            best, bd = None, 0.20
            for tr in tracks:
                if id(tr) in used or fn - tr["end"] > gap:
                    continue
                d = ((tr["last"][0] - c[0]) ** 2 + (tr["last"][1] - c[1]) ** 2) ** 0.5
                if d < bd:
                    best, bd = tr, d
            if best is None:
                best = {"last": c, "end": fn, "items": []}
                tracks.append(best)
            used.add(id(best))
            best["last"], best["end"] = c, fn
            best["items"].append((fn, p))

    # 2. Pro Track: Lücken füllen, Ausreißer entfernen und Gauß-Filter anwenden
    sigma = max(1.5, 0.08 * fps)
    rad = int(sigma * 2.5)

    track_results = []
    for tr in tracks:
        its = tr["items"]
        if len(its) < 2:
            continue

        dense_items = []
        for idx in range(len(its)):
            fn, p = its[idx]
            if dense_items and 1 < fn - dense_items[-1][0] <= 5:
                prev_fn, prev_p = dense_items[-1]
                steps = fn - prev_fn
                arr_prev = np.array(prev_p, dtype=float)
                arr_curr = np.array(p, dtype=float)
                for step in range(1, steps):
                    frac = step / steps
                    interp = (1.0 - frac) * arr_prev + frac * arr_curr
                    dense_items.append((prev_fn + step, interp.tolist()))
            dense_items.append((fn, p))

        n = len(dense_items)
        if n < 3:
            continue

        fr = np.array([item[0] for item in dense_items], dtype=float)
        arr = np.array([item[1] for item in dense_items], dtype=float)  # (n, 33, 3)

        # Ausreißer-Spikes (1-Frame-Zucken) korrigieren
        for k in range(1, n - 1):
            if fr[k] - fr[k - 1] == 1.0 and fr[k + 1] - fr[k] == 1.0:
                d_prev = np.hypot(arr[k, :, 0] - arr[k - 1, :, 0], arr[k, :, 1] - arr[k - 1, :, 1])
                d_next = np.hypot(arr[k, :, 0] - arr[k + 1, :, 0], arr[k, :, 1] - arr[k + 1, :, 1])
                d_span = np.hypot(arr[k + 1, :, 0] - arr[k - 1, :, 0], arr[k + 1, :, 1] - arr[k - 1, :, 1])
                spike = (d_prev > 0.12) & (d_next > 0.12) & (d_span < 0.08)
                if np.any(spike):
                    arr[k, spike, 0] = (arr[k - 1, spike, 0] + arr[k + 1, spike, 0]) / 2.0
                    arr[k, spike, 1] = (arr[k - 1, spike, 1] + arr[k + 1, spike, 1]) / 2.0
                    arr[k, spike, 2] = (arr[k - 1, spike, 2] + arr[k + 1, spike, 2]) / 2.0

        # Gauß-Filter über Zeit
        res = arr.copy()
        for k in range(n):
            lo, hi = k, k
            while lo > 0 and fr[k] - fr[lo - 1] <= rad:
                lo -= 1
            while hi < n - 1 and fr[hi + 1] - fr[k] <= rad:
                hi += 1
            w = np.exp(-0.5 * ((fr[lo:hi + 1] - fr[k]) / sigma) ** 2)[:, None] * np.clip(arr[lo:hi + 1, :, 2], 0.05, 1.0)
            ws = w.sum(axis=0)
            ok = ws > 1e-6
            for c in (0, 1):
                v = (w * arr[lo:hi + 1, :, c]).sum(axis=0) / np.where(ok, ws, 1.0)
                res[k, :, c] = np.where(ok, v, arr[k, :, c])
            w_raw = np.exp(-0.5 * ((fr[lo:hi + 1] - fr[k]) / (sigma * 1.5)) ** 2)[:, None]
            res[k, :, 2] = np.clip((w_raw * arr[lo:hi + 1, :, 2]).sum(axis=0) / w_raw.sum(axis=0), 0.0, 1.0)

        for (fn_i, _), r in zip(dense_items, res):
            track_results.append((fn_i, [[round(float(a), 3), round(float(b), 3), round(float(c), 2)] for a, b, c in r]))

    frame_dict = {}
    for fn, p in track_results:
        frame_dict.setdefault(fn, []).append(p)

    tl.clear()
    for fn in sorted(frame_dict.keys()):
        tl.append([round(fn / fps, 3), frame_dict[fn]])


# ---------------------------------------------------------------- Haar- und Hautfarbe
# Ablauf: Gesichtsausschnitt aus den Pose-Gesichtspunkten (0-10) -> BiSeNet Face-Parsing (Haut/Haar-Maske)
# -> K-Means in Lab -> Haut: ITA-Winkel, Haar: Regeln über HSV.
_parse_sess = None
_parse_lock = threading.Lock()
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
SKIN_IDS, NOSE_ID, HAIR_ID = (1,), 10, 17


def _parser():
    global _parse_sess
    with _parse_lock:
        if _parse_sess is None:
            _parse_sess = onnxruntime.InferenceSession(PARSE_MODEL, providers=["CPUExecutionProvider"])
        return _parse_sess


def face_crop(bgr, pose):
    """Quadratischer Ausschnitt um Gesicht + Haar aus den Pose-Gesichtspunkten; None wenn zu klein/unsicher."""
    import cv2

    h, w = bgr.shape[:2]
    pts = [(p[0] * w, p[1] * h) for p in pose[:11] if p[2] >= 0.3]
    if len(pts) < 4:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    fw = max(max(xs) - min(xs), (max(ys) - min(ys)) * 0.9)
    if fw < 18:
        return None
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2 - 0.3 * fw
    half = int(round(1.2 * fw))
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    pad = max(0, -x0, -y0, x0 + 2 * half - w, y0 + 2 * half - h)
    img = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else bgr
    return img[y0 + pad:y0 + pad + 2 * half, x0 + pad:x0 + pad + 2 * half]


_age_sess = None


def estimate_age(crop):
    """Geschätztes Alter aus dem Gesichtsausschnitt (InsightFace genderage, 96x96); None ohne Modell.
    Grobe Schätzung (Fehler ca. +-5-8 Jahre), nur zum Aussortieren gedacht."""
    global _age_sess
    import cv2

    if not (AGE_ON and os.path.exists(AGE_MODEL)):
        return None
    with _parse_lock:
        if _age_sess is None:
            _age_sess = onnxruntime.InferenceSession(AGE_MODEL, providers=["CPUExecutionProvider"])
    # Der Ausschnitt ist ca. 2,4x so groß wie das Gesicht; das Modell erwartet das Gesicht mit etwas Rand (ca. 1,5x)
    h, w = crop.shape[:2]
    m = int(w * 0.18)
    face = crop[m + int(h * 0.04):h - m + int(h * 0.04), m:w - m]
    if face.size == 0:
        return None
    x = cv2.cvtColor(cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    out = _age_sess.run(None, {_age_sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
    return int(round(float(out[2]) * 100))


def parse_face(crop):
    import cv2

    rgb = cv2.cvtColor(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
    sess = _parser()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return out[0].argmax(0).astype(np.uint8), rgb


def _dominant_lab(lab_px, k=3):
    """K-Means in Lab; liefert (Lab-Mittel, Anteil) pro Cluster, größte zuerst."""
    import cv2

    if len(lab_px) > 3000:
        lab_px = lab_px[np.random.default_rng(0).choice(len(lab_px), 3000, replace=False)]
    k = max(1, min(k, len(lab_px) // 40))
    if k == 1:
        return [(lab_px.mean(0), 1.0)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lbl, cen = cv2.kmeans(lab_px.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lbl.ravel(), minlength=k)
    order = np.argsort(-cnt)
    return [(cen[i], cnt[i] / len(lab_px)) for i in order]


def _lab_to_rgb(lab):
    import cv2

    px = np.uint8([[np.clip(lab, 0, 255)]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return int(r), int(g), int(b)


def skin_class(L, b):
    """ITA = atan2(L-50, b): Schwellen nach Chardon et al. (Lab-L 0..100, b mit Vorzeichen)."""
    ita = np.degrees(np.arctan2(L - 50.0, max(b, 0.5)))
    for lim, name in ((55, "sehr hell"), (41, "hell"), (28, "mittel"), (10, "gebräunt"), (-30, "braun")):
        if ita > lim:
            return name
    return "dunkel"


def hair_class(rgb):
    import colorsys

    r, g, b = [c / 255.0 for c in rgb]
    hh, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = hh * 360
    if s < 0.18:
        return "weiß" if v > 0.78 else "grau" if v > 0.40 else "schwarz"
    if v < 0.22:
        return "schwarz"
    if s > 0.55 and v > 0.45 and not (8 <= hue <= 50):
        return "bunt"
    if (hue < 18 or hue > 340) and s > 0.5 and v > 0.35:
        return "rot"
    if v < 0.30:
        return "schwarz"
    if v < 0.45:
        return "dunkelbraun"
    if v < 0.62:
        return "braun"
    return "blond" if s < 0.75 else "rot"


def _skin_lab(px):
    """Typische Hautfarbe (Lab, 8 Bit) aus Hautpixeln: dominante Cluster, dunkle Schattencluster ignoriert."""
    cl = _dominant_lab(px)
    top = cl[0][0][0]
    keep = [(c, wgt) for c, wgt in cl if c[0] >= top - 255 * 0.15]
    tw = sum(wgt for _, wgt in keep)
    return sum(c * wgt for c, wgt in keep) / tw


# Gliedmaßen (Pose-Punkte) und Rumpf, aus denen Körperhaut gesammelt wird
LIMBS = ((11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28))
TORSO = (11, 12, 24, 23)


def body_mask(shape, pose, items):
    """Maske der Körperregionen (Rumpf, Arme, Beine, freiliegende NudeNet-Teile) im Frame."""
    import cv2

    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    pt = lambda i: (int(pose[i][0] * w), int(pose[i][1] * h))  # noqa: E731
    ok = lambda i: pose[i][2] >= 0.4  # noqa: E731
    sw = np.hypot(*(np.subtract(pt(11), pt(12)))) if ok(11) and ok(12) else 0
    th = max(3, int(sw * 0.2))
    for a, b in LIMBS:
        if ok(a) and ok(b):
            cv2.line(m, pt(a), pt(b), 1, th)
    if all(ok(i) for i in TORSO):
        quad = np.array([pt(i) for i in TORSO], np.float32)
        c = quad.mean(0)
        cv2.fillPoly(m, [((quad - c) * 0.8 + c).astype(np.int32)], 1)  # Rand (Arme, Kleidung) weglassen
    for d in items or []:
        lab = str(d["label"])
        if "EXPOSED" in lab and not lab.startswith("FACE"):
            x, y, bw, bh = d["box"]
            x0, y0, x1, y1 = int((x + bw * .1) * w), int((y + bh * .1) * h), int((x + bw * .9) * w), int((y + bh * .9) * h)
            m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
    return m


def body_skin(bgr, pose, items, ref):
    """Hautpixel vom ganzen Körper: nur Pixel, die hautfarben sind (YCrCb) und zur Gesichtsfarbe passen.
    Kleidung fliegt so raus. Liefert (Lab 8 Bit, Pixelzahl) oder None."""
    import cv2

    m = body_mask(bgr.shape, pose, items)
    if m.sum() < 150:
        return None
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    cr, cb = ycc[..., 1], ycc[..., 2]
    ok = (m > 0) & (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135) & (lab[..., 0] > 255 * 0.12) & (lab[..., 0] < 255 * 0.97)
    px = lab[ok].astype(np.float32)
    if ref is not None and len(px):
        d = np.sqrt(((px[:, 0] - ref[0]) * 100 / 255) ** 2 + (px[:, 1] - ref[1]) ** 2 + (px[:, 2] - ref[2]) ** 2)
        px = px[d < 30]
    if len(px) < 150:
        return None
    return _skin_lab(px), len(px)


def person_colors(bgr, pose, items=None):
    """Haar- und Hautfarbe einer Person (Gesicht + ganzer Körper); fehlende Teile bleiben None."""
    import cv2

    res = {"hair": None, "skin": None, "hair_rgb": None, "skin_rgb": None, "skin_face": None, "skin_body": None, "body_rgb": None, "age": None}
    face_lab, face_n = None, 0
    crop = face_crop(bgr, pose)
    if crop is not None:
        try:
            res["age"] = estimate_age(crop)
        except Exception:  # noqa: BLE001
            pass
        seg, rgb = parse_face(crop)
        lab_img = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
        ker = np.ones((5, 5), np.uint8)
        skin = cv2.erode(np.isin(seg, SKIN_IDS + (NOSE_ID,)).astype(np.uint8), ker)
        px = lab_img[skin > 0].astype(np.float32)
        px = px[(px[:, 0] > 255 * 0.12) & (px[:, 0] < 255 * 0.97)]  # tiefe Schatten und Glanzlichter raus
        if len(px) >= 80:
            face_lab, face_n = _skin_lab(px), len(px)
            res["skin_face"] = skin_class(face_lab[0] * 100 / 255, face_lab[2] - 128)
        hair = cv2.erode((seg == HAIR_ID).astype(np.uint8), ker)
        if hair.sum() >= 0.015 * seg.size:
            cl = _dominant_lab(lab_img[hair > 0].astype(np.float32))
            rgbc = _lab_to_rgb(cl[0][0])
            res["hair"] = hair_class(rgbc)
            res["hair_rgb"] = rgbc
    body = body_skin(bgr, pose, items, face_lab)
    if body:
        res["skin_body"] = skin_class(body[0][0] * 100 / 255, body[0][2] - 128)
        res["body_rgb"] = _lab_to_rgb(body[0])
    # Gesamtwert: Gesicht und Körper nach Pixelzahl gewichtet (Gesicht zählt höchstens so viel wie der Körper)
    if face_lab is not None or body:
        parts = []
        if face_lab is not None:
            parts.append((face_lab, min(face_n, body[1]) if body else face_n))
        if body:
            parts.append((body[0], body[1]))
        tw = sum(w for _, w in parts)
        c = sum(l * w for l, w in parts) / tw
        res["skin"] = skin_class(c[0] * 100 / 255, c[2] - 128)
        res["skin_rgb"] = _lab_to_rgb(c)
    return res if any(v for v in res.values()) else None


def colors_for(bgr, poses, items=None):
    if not (COLORS_ON and poses and os.path.exists(PARSE_MODEL)):
        return None
    out = []
    for p in poses:
        try:
            out.append(person_colors(bgr, p, items))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


class ScanCancelled(Exception):
    """Der Aufrufer hat die Verbindung geschlossen (Beitrag zu weit zurück): Scan abbrechen."""


class Req(BaseModel):
    path: str
    kind: str  # "image" | "video"
    parts: int = 1  # Video: in so viele Zeitabschnitte teilen und parallel scannen


@app.get("/healthz")
def healthz():
    return "ok"


NIP_LABELS = ("FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED")


def find_nipple(bgr, box):
    """Brustwarze/Areola in einer Brust-Box (x, y, w, h normiert) per Farbanalyse. Liefert (x, y, conf) normiert oder None."""
    import cv2

    H, W = bgr.shape[:2]
    x0, y0 = max(0, int(box[0] * W)), max(0, int(box[1] * H))
    x1, y1 = min(W, int((box[0] + box[2]) * W)), min(H, int((box[1] + box[3]) * H))
    bw, bh = x1 - x0, y1 - y0
    if bw < 24 or bh < 24:
        return None
    s = (256.0 if NIP_PREC >= 2 else 160.0) / max(bw, bh)  # auf feste Größe skalieren: schnell und größenunabhängig
    crop = cv2.resize(bgr[y0:y1, x0:x1], (max(8, int(bw * s)), max(8, int(bh * s))), interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    m = float(max(cw, ch))
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    skin = (ycc[..., 1] >= 130) & (ycc[..., 1] <= 185) & (ycc[..., 2] >= 70) & (ycc[..., 2] <= 140) & (hsv[..., 1] < 215)
    if skin.sum() < 0.2 * skin.size:
        return None
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    med = lambda c: float(np.median(c[skin]))
    # Areola/Warze: röter (a*) und dunkler als die Haut ringsum
    sc = (A - med(A)) * 1.0 + (med(L) - L) * 0.35 + (B - med(B)) * 0.2
    sc[~skin] = -50
    raw = sc.copy()
    sc = cv2.GaussianBlur(sc, (0, 0), max(2.0, 0.06 * m))
    # Scheiben-Suche (Differenz zweier Unschärfen): findet runde Flecken statt Kanten und Schatten.
    # Nicht-Haut zählt neutral (0), damit Ränder keine Scheiben vortäuschen.
    rel = np.where(skin, raw - float(np.median(raw[skin])), 0.0).astype(np.float32)
    fracs = (0.045, 0.07, 0.10) if NIP_PREC >= 2 else (0.07,)
    blob = None
    for f in fracs:
        sg = max(1.5, f * m)
        d = cv2.GaussianBlur(rel, (0, 0), sg) - cv2.GaussianBlur(rel, (0, 0), sg * 2.2)
        blob = d if blob is None else np.maximum(blob, d)
    skinf = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), max(2.0, 0.05 * m))
    yy, xx = np.mgrid[0:ch, 0:cw]
    # Die Brustwarze liegt meist ziemlich genau in der Mitte der Brust-Box
    prior = np.exp(-(((xx - cw / 2) / (0.30 * cw)) ** 2 + ((yy - ch * 0.5) / (0.30 * ch)) ** 2))
    edge = np.ones_like(sc)
    mx, my = int(cw * 0.10), int(ch * 0.10)
    edge[:my] = 0; edge[-my:] = 0; edge[:, :mx] = 0; edge[:, -mx:] = 0
    r = np.clip(blob, 0, None) * prior * edge * skinf
    if r.max() <= 0:
        r = (sc - sc[skin].mean()) * prior * edge
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    # Feinsuche: im Umkreis der Areola mit kleiner Unschärfe das gewichtete Zentrum der stärksten Pixel nehmen
    fine = cv2.GaussianBlur(raw, (0, 0), max(1.2, 0.02 * m))
    rad = int((0.09 if NIP_PREC >= 2 else 0.12) * m)
    ya, yb, xa, xb = max(0, iy - rad), min(ch, iy + rad + 1), max(0, ix - rad), min(cw, ix + rad + 1)
    win = fine[ya:yb, xa:xb].copy()
    win[~skin[ya:yb, xa:xb]] = -50
    wmax = win.max()
    wt = np.clip(win - (wmax - 0.35 * (wmax - np.median(win))), 0, None) ** 2
    if wt.sum() > 0:
        gy, gx = np.mgrid[ya:yb, xa:xb]
        iy, ix = int(round((wt * gy).sum() / wt.sum())), int(round((wt * gx).sum() / wt.sum()))
    conf = float(np.clip((sc[iy, ix] - sc[skin].mean()) / 12.0, 0.2, 0.95))
    return ((x0 + (ix + .5) / s) / W, (y0 + (iy + .5) / s) / H, conf)


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
    boxes = (yolo_persons([img], YOLO_IMGSZ) or [None])[0]
    items = detect_person_crops(det, img, boxes, items, w, h)
    for d in items:
        if d["label"] in NIP_LABELS:
            nn = find_nipple(img, d["box"])
            if nn:
                d["nip"] = [round(nn[0], 4), round(nn[1], 4), round(nn[2], 2)]
    out = {"kind": "image", "width": w, "height": h, "detections": items}
    poses, hs = [], []
    if HOLISTIC_ON:
        hol = Holistic(False)
        try:
            poses, hs = hol.run(img)
        finally:
            hol.close()
    else:
        if POSE_ON:
            pose = Pose(False)
            pose_crop = Pose(False, conf=POSE_CROP_CONF, num=1) if boxes and len(boxes) > 1 else None
            try:
                poses = pose.run(img)
                poses = crop_poses(pose_crop, img, boxes, limit_by_persons(boxes, poses, [])[0])
            finally:
                pose.close()
                if pose_crop:
                    pose_crop.close()
        if HANDS_ON:
            hands = Hands(False)
            try:
                hs = hands.run(img)
            finally:
                hands.close()
    poses, hs = limit_by_persons(boxes, poses, hs, anchors_from((d["label"], d["box"]) for d in items))
    poses = reconcile(poses, hs, items)
    if HOLISTIC_ON or POSE_ON:
        out["poses"] = poses
    if HOLISTIC_ON or HANDS_ON:
        out["hands"] = hs
    if boxes is not None:
        out["persons"] = len(boxes)
    cols = colors_for(img, poses, items)
    if cols is not None:
        out["appearance"] = cols
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


def _video_part(det, path, cancel, w, h, fps, start=0, count=None):
    """Ein Abschnitt des Videos (ab Frame `start`, `count` Frames; None = bis zum Ende). Liefert die Rohdaten."""
    size = w * h * 3
    cmd = ["ffmpeg", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start / fps:.4f}"]
    cmd += ["-i", path, "-an"]
    if count:
        cmd += ["-frames:v", str(count)]
    # Kein fps-Filter: wirklich jeder Frame wird geprüft.
    cmd += ["-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    labels = {}
    timeline = []  # pro Frame mit Treffern: [Zeit, [[Label-Nr, Score, x, y, b, h], ...]]
    names = []
    pose_tl = []  # pro Frame mit Personen: [Zeit, [Person(33 x [x, y, Sichtbarkeit]), ...]]
    hand_tl = []
    hol = Holistic(True) if HOLISTIC_ON else None
    pose = Pose(True) if (POSE_ON and not HOLISTIC_ON) else None
    hands = Hands(True) if (HANDS_ON and not HOLISTIC_ON) else None
    pose_ex = ThreadPoolExecutor(1) if (hol or pose or hands or YOLO_ON) else None
    color_tl = []  # [Zeit, [{hair, skin, ...} je Person]] nur für jeden n-ten Frame
    person_tl = []  # nur Änderungen der Personenzahl: [Zeit, Anzahl]
    person_max, person_frames = 0, 0  # läuft parallel zu NudeNet, bleibt in Frame-Reihenfolge
    total = 0
    batch, idx = [], []

    def pose_batch(frames, ids):
        boxes = yolo_persons(frames)
        out = []
        for k, (f, i) in enumerate(zip(frames, ids)):
            b = boxes[k] if boxes else None
            t_frame = i / fps
            if hol:
                p_out, h_out = hol.run(f, t_frame)
            else:
                p_out = pose.run(f, t_frame) if pose else []
                h_out = hands.run(f, t_frame) if hands else []
            out.append((t_frame, b, p_out, h_out))
        return out

    def flush():
        nonlocal batch, idx, person_max, person_frames
        if not batch:
            return
        fut = pose_ex.submit(pose_batch, batch, idx) if pose_ex else None
        frame_items = []
        for i, fr0, dets in zip(idx, batch, detect_many(det, batch)):
            t = i / fps
            seen = set()
            row = []
            for d in dets:
                if d["score"] < MIN_SCORE:
                    continue
                n = norm(d, w, h)
                if n["label"] not in names:
                    names.append(n["label"])
                r_ = [names.index(n["label"]), n["score"]] + [round(v, 3) for v in n["box"]]
                if n["label"] in NIP_LABELS:
                    nn = find_nipple(fr0, n["box"])
                    if nn:
                        r_ += [round(nn[0], 4), round(nn[1], 4), round(nn[2], 2)]
                row.append(r_)
                e = labels.setdefault(n["label"], {"label": n["label"], "max": 0.0, "frames": 0, "first": t, "last": t, "box": n["box"], "at": t})
                if n["score"] > e["max"]:
                    e["max"], e["box"], e["at"] = n["score"], n["box"], t
                if n["label"] not in seen:
                    e["frames"] += 1
                    seen.add(n["label"])
                e["last"] = t
            if row:
                timeline.append([round(t, 3), row])
            frame_items.append([{"label": names[r[0]], "box": r[2:6]} for r in row])
        if fut:
            for (t, b, people, hs), fi, fr, ix in zip(fut.result(), frame_items, batch, idx):
                fa = anchors_from((d["label"], d["box"]) for d in fi)
                people, hs = limit_by_persons(b, people, hs, fa)
                npers = len(b) if b is not None else None
                people = reconcile(people, hs, fi)
                if npers is not None and (not person_tl or person_tl[-1][1] != npers):
                    person_tl.append([round(t, 3), npers])
                if npers:
                    person_frames += 1
                    person_max = max(person_max, npers)
                if people:
                    pose_tl.append([round(t, 3), people])
                    if COLORS_ON and ix % COLORS_EVERY == 0:
                        cols = colors_for(fr, people, fi)
                        if cols and any(cols):
                            color_tl.append([round(t, 3), cols])
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
            idx.append(start + total)
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
        if hol:
            hol.close()
        if pose:
            pose.close()
        if hands:
            hands.close()
    return {"labels": labels, "timeline": timeline, "names": names, "pose_tl": pose_tl, "hand_tl": hand_tl,
            "color_tl": color_tl, "person_tl": person_tl, "person_max": person_max, "person_frames": person_frames,
            "total": total, "hol": bool(hol), "pose": bool(pose), "hands": bool(hands)}


class _AnyStop:
    """Abbruch, sobald der Auftrag abgebrochen wird oder ein anderer Abschnitt fehlschlägt."""

    def __init__(self, cancel, own):
        self.cancel, self.own = cancel, own

    def is_set(self):
        return self.own.is_set() or (self.cancel is not None and self.cancel.is_set())


def _merge_parts(states):
    names, labels = [], {}
    timeline, pose_tl, hand_tl, color_tl, person_tl = [], [], [], [], []
    person_max = person_frames = total = 0
    for st in states:
        remap = []
        for n in st["names"]:
            if n not in names:
                names.append(n)
            remap.append(names.index(n))
        for t, row in st["timeline"]:
            timeline.append([t, [[remap[r[0]]] + r[1:] for r in row]])
        for lab, e in st["labels"].items():
            m = labels.get(lab)
            if m is None:
                labels[lab] = dict(e)
                continue
            if e["max"] > m["max"]:
                m["max"], m["box"], m["at"] = e["max"], e["box"], e["at"]
            m["frames"] += e["frames"]
            m["first"], m["last"] = min(m["first"], e["first"]), max(m["last"], e["last"])
        pose_tl += st["pose_tl"]
        hand_tl += st["hand_tl"]
        color_tl += st["color_tl"]
        for t, n in st["person_tl"]:
            if not person_tl or person_tl[-1][1] != n:
                person_tl.append([t, n])
        person_max = max(person_max, st["person_max"])
        person_frames += st["person_frames"]
        total += st["total"]
    return {"labels": labels, "timeline": timeline, "names": names, "pose_tl": pose_tl, "hand_tl": hand_tl,
            "color_tl": color_tl, "person_tl": person_tl, "person_max": person_max, "person_frames": person_frames,
            "total": total, "hol": any(x["hol"] for x in states), "pose": any(x["pose"] for x in states),
            "hands": any(x["hands"] for x in states)}


def scan_video(det, path, cancel=None, parts=1):
    w, h, fps, duration = probe(path)
    if w > MAX_WIDTH:
        h = int(h * MAX_WIDTH / w) // 2 * 2
        w = MAX_WIDTH
    w, h = max(2, w // 2 * 2), max(2, h // 2 * 2)
    est = int(round(duration * fps)) if duration else 0
    parts = max(1, min(int(parts or 1), 16, est // (BATCH * 6) if est else 1))
    if parts <= 1:
        states = [_video_part(det, path, cancel, w, h, fps)]
    else:
        # Extra-Detektoren: erst aus dem Pool, sonst neu laden (werden danach verworfen).
        dets, fromp = [det], []
        for _ in range(parts - 1):
            try:
                d = pool.get_nowait()
                fromp.append(d)
            except queue.Empty:
                d = NudeDetector()
            dets.append(d)
        edges = [round(i * est / parts) for i in range(parts + 1)]
        own = threading.Event()
        errs = []
        stop = _AnyStop(cancel, own)

        def job(i):
            try:
                cnt = None if i == parts - 1 else edges[i + 1] - edges[i]
                return _video_part(dets[i], path, stop, w, h, fps, edges[i], cnt)
            except BaseException as e:
                if not isinstance(e, ScanCancelled):
                    errs.append(e)
                own.set()
                raise

        try:
            with ThreadPoolExecutor(parts) as ex:
                states = list(ex.map(job, range(parts)))
        except ScanCancelled:
            if errs:
                raise errs[0]
            raise
        finally:
            for d in fromp:
                pool.put(d)
    st = _merge_parts(states)
    labels, timeline, names, pose_tl, hand_tl, color_tl, person_tl = (
        st["labels"], st["timeline"], st["names"], st["pose_tl"], st["hand_tl"], st["color_tl"], st["person_tl"])
    total, person_max, person_frames = st["total"], st["person_max"], st["person_frames"]
    hol, pose, hands = st["hol"], st["pose"], st["hands"]
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
    if hol or pose:
        smooth_poses(pose_tl, fps)
        res["pose_timeline"] = pose_tl
        res["pose_frames"] = len(pose_tl)
    if YOLO_ON:
        res["persons_max"] = person_max
        res["person_frames"] = person_frames
        res["person_timeline"] = person_tl
    if color_tl:
        res["appearance_timeline"] = color_tl
    if hol or hands:
        res["hand_timeline"] = hand_tl
        res["hand_frames"] = len(hand_tl)
    return res


def work(req: Req, cancel=None):
    path = os.path.realpath(req.path)
    if not path.startswith(ROOT + "/"):
        raise HTTPException(400, "Pfad nicht erlaubt")
    det = pool.get()
    try:
        return scan_image(det, path) if req.kind == "image" else scan_video(det, path, cancel, req.parts)
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
