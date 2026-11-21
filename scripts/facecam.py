"""Find the streamer's face cam in a few frames of a clip, so layout.json starts filled in.

  python facecam.py <frame.jpg> [<frame.jpg> ...]     prints the suggestion as JSON

How:
  1. Find faces in every frame (OpenCV's YuNet detector, ~230 KB, downloaded once).
  2. The streamer is the face that stays in the same place in every frame (game characters move).
  3. A big face near the middle means a full-screen camera (podcast, IRL, Just Chatting): no face cam
     box, and focus_x goes on the face. Two people far apart: fit the whole picture instead.
  4. Faces in a corner or along the bottom sit in a face cam. Its borders are the straight edges around
     the face that stay put across the frames (game edges move), or the edge of the picture; cut-out
     streamers with no border (and face cams with two people) get a box around the people.
  5. layout_segments() looks at one frame a second and splits the clip where the layout changes.
Claude still looks at the frames: this is a starting point, with a confidence.
"""
import json
import sys
import urllib.request

import numpy as np

from common import HOME_DIR

MODEL = HOME_DIR / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FULL_CAMERA_AREA = 0.012   # a face covering more than this share of the picture = full-screen camera
                           # (in a face cam overlay the face is ~0.3%; a full camera ~2% and up)


def _detector(w, h):
    import cv2
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # quiet "targets not supported" warnings
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL)
    return cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), score_threshold=0.75)


def faces(img):
    """[(x, y, w, h, score)] in pixels."""
    h, w = img.shape[:2]
    det = _detector(w, h)
    _, found = det.detect(img)
    return [] if found is None else [tuple(map(float, f[:4])) + (float(f[14]),) for f in found]


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def same_face(a, b):
    """The same person in two frames: overlapping boxes, or centers closer than ~ a face width
    (small face cam faces move a few pixels, which wrecks the overlap of small boxes)."""
    if iou(a, b) > 0.3:
        return True
    dx = (a[0] + a[2] / 2) - (b[0] + b[2] / 2)
    dy = (a[1] + a[3] / 2) - (b[1] + b[3] / 2)
    size = max(a[2], b[2])
    return (dx * dx + dy * dy) ** 0.5 < 0.8 * size and 0.5 <= (a[2] * a[3]) / (b[2] * b[3] + 1e-9) <= 2.0


def stable_face(per_frame):
    """The face found in the same place in the most frames: (box, frames it was in)."""
    best = (None, 0)
    for i, fs in enumerate(per_frame):
        for f in fs:
            hits = [f] + [max(other, key=lambda g: iou(f, g)) for j, other in enumerate(per_frame)
                          if j != i and other and max(iou(f, g) for g in other) > 0.4]
            if len(hits) > best[1]:
                box = tuple(float(np.median([h[k] for h in hits])) for k in range(4))
                best = (box, len(hits))
    return best


def static_edges(grays):
    """Edges present in every frame: borders of overlays (the face cam box, chat), not the game."""
    import cv2
    mags = []
    for g in grays:
        gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        mags.append((gx, gy))
    return np.min([m[0] for m in mags], axis=0), np.min([m[1] for m in mags], axis=0)


def _best_line(profile, lo, hi, border, outward):
    """The face cam's border in [lo, hi) of a 1-D static-edge profile. Rooms behind streamers are
    static too, so furniture makes strong lines inside the box; the border is the strong line
    farthest out from the face. The picture's own edge counts as a border."""
    lo, hi = max(0, int(lo)), min(len(profile), int(hi))
    if hi - lo < 2:
        return border, 0.0
    seg = profile[lo:hi]
    med = np.median(profile) + 1e-6
    strong = np.where((seg >= 0.6 * seg.max()) & (seg >= 3.0 * med))[0]
    if not len(strong):  # no clear line: use the picture's edge if it's in range
        return (border, 1.0) if lo <= border <= hi else (None, 0.0)
    k = int(strong[-1] if outward > 0 else strong[0])
    return lo + k, float(seg[k] / med)


def box_around(face, gx, gy, W, H):
    """The face cam rectangle around a face, from static edges. Returns (x, y, w, h, confidence)."""
    fx, fy, fw, fh = face
    rows = slice(int(max(0, fy - fh)), int(min(H, fy + 2 * fh)))
    cols = slice(int(max(0, fx - fw)), int(min(W, fx + 2 * fw)))
    vprof = gx[rows].mean(axis=0)  # vertical lines: strength per column
    hprof = gy[:, cols].mean(axis=1)  # horizontal lines: strength per row
    L, sl = _best_line(vprof, fx - 4 * fw, fx - 0.2 * fw, 0, -1)
    R, sr = _best_line(vprof, fx + 1.2 * fw, fx + 5 * fw, W - 1, +1)
    T, st = _best_line(hprof, fy - 3 * fh, fy - 0.2 * fh, 0, -1)
    B, sb = _best_line(hprof, fy + 1.3 * fh, fy + 5 * fh, H - 1, +1)
    found = [x is not None for x in (L, R, T, B)]
    # fall back to a box about 2.6 face widths wide around the face for sides without a line
    L = L if L is not None else max(0, fx - 0.8 * fw)
    R = R if R is not None else min(W - 1, fx + 1.8 * fw)
    T = T if T is not None else max(0, fy - 0.7 * fh)
    B = B if B is not None else min(H - 1, fy + 1.9 * fh)
    conf = sum(found) / 4 * min(1.0, np.mean([s for s in (sl, sr, st, sb) if s]) / 4 if any(found) else 0)
    return L, T, R - L, B - T, round(float(conf), 2)


def stable_faces(per_frame, min_share=0.5):
    """Every face seen in the same place in at least half the frames: [(box, frames)]."""
    out = []
    need = max(1, int(np.ceil(min_share * len(per_frame))))
    for i, fs in enumerate(per_frame):
        for f in fs:
            hits = [f] + [next(g for g in other if same_face(f, g)) for j, other in enumerate(per_frame)
                          if j != i and any(same_face(f, g) for g in other)]
            if len(hits) < need:
                continue
            box = tuple(float(np.median([h[k] for h in hits])) for k in range(4))
            if all(not same_face(box, b) for b, _ in out):
                out.append((box, len(hits)))
    return out


def is_full_camera(faces_, W, H):
    """A full-screen camera (podcast, IRL, the streamer filling the picture) rather than a face cam
    overlay on gameplay: one big face, or a face that's big AND near the middle. Face cams sit in a
    corner or along the bottom, and can hold two people with fair-sized faces."""
    upper = 0
    for x, y, w, h in faces_:
        area, cx, cy = w * h / (W * H), (x + w / 2) / W, (y + h / 2) / H
        if area > 0.06 or (area > FULL_CAMERA_AREA and 0.25 <= cx <= 0.75 and cy <= 0.7):
            return True
        if area > 0.03 and cy < 0.6:  # a big face high up, even off to the side: face cam faces are never this big
            return True
        if area > 0.008 and cy < 0.55:
            upper += 1
    return upper >= 2  # two people in the upper half: a camera shot of both (podcast, green screen)


def union_box(faces_, W, H):
    """A face cam box around one or more faces with no border to find (cut-out streamers over the
    game): each face plus its shoulders, joined, stretched to the picture's edge when close to it."""
    x1 = max(0.0, min(x - 0.6 * w for x, y, w, h in faces_))
    y1 = max(0.0, min(y - 0.5 * h for x, y, w, h in faces_))
    x2 = min(float(W), max(x + 1.6 * w for x, y, w, h in faces_))
    y2 = min(float(H), max(y + 2.6 * h for x, y, w, h in faces_))
    if x1 < 0.05 * W:
        x1 = 0.0
    if x2 > 0.95 * W:
        x2 = float(W)
    if y2 > 0.9 * H:
        y2 = float(H)
    return x1, y1, x2 - x1, y2 - y1


def suggest(paths):
    """{"facecam": [x, y, w, h] fractions or None, "focus_x": 0..1, "mode": "stack"|"crop"|"fit",
    "confidence": 0..1, "how": str}"""
    import cv2
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return {"facecam": None, "focus_x": 0.5, "mode": "crop", "confidence": 0.0, "how": "no frames"}
    H, W = imgs[0].shape[:2]
    per = [faces(i) for i in imgs]
    found = stable_faces(per)
    if not found:
        return {"facecam": None, "focus_x": 0.5, "mode": "crop", "confidence": 0.3,
                "how": "no face found: gameplay only, or a face cam too small to see"}
    boxes = [b for b, _ in found]
    steady = max(n for _, n in found) / len(imgs)
    # co-hosts in the same face cam who only showed up in some frames (a zoom, someone leaning out)
    companions = []
    for fs in per:
        if not any(same_face(f[:4], b) for f in fs for b in boxes):
            continue  # a co-host shows up next to the main face, not instead of it (that's the same person moving)
        for f in fs:
            fb = f[:4]
            if any(same_face(fb, b) for b in boxes + companions):
                continue
            if any(abs((fb[0] + fb[2] / 2) - (b[0] + b[2] / 2)) < 4 * b[2] and abs((fb[1] + fb[3] / 2) - (b[1] + b[3] / 2)) < 1.5 * b[3]
                   and 0.4 <= fb[2] * fb[3] / (b[2] * b[3]) <= 2.5 for b in boxes):
                companions.append(fb)
    if is_full_camera(boxes, W, H):
        centers = [(x + w / 2) / W for x, y, w, h in boxes if w * h / (W * H) > FULL_CAMERA_AREA]
        if len(centers) >= 2 and max(centers) - min(centers) > 0.3:  # two people far apart: show them both
            return {"facecam": None, "focus_x": round(float(np.mean(centers)), 3), "mode": "fit",
                    "confidence": round(0.5 + 0.5 * steady, 2), "how": "full-screen camera with people far apart: fit it"}
        return {"facecam": None, "focus_x": round(float(np.mean(centers or [0.5])), 3), "mode": "crop",
                "confidence": round(0.5 + 0.5 * steady, 2), "how": "big face near the middle: full-screen camera"}
    people = boxes + companions
    if len(people) == 1:
        gx, gy = static_edges([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs])
        x, y, w, h, c = box_around(boxes[0], gx, gy, W, H)
        how = f"face cam box around a face seen in {found[0][1]} of {len(imgs)} frames"
        if c < 0.3:  # no clear border: a cut-out streamer
            x, y, w, h = union_box(people, W, H)
            c, how = 0.5, how + " (no border found, boxed around the person)"
    else:
        x, y, w, h = union_box(people, W, H)
        c, how = 0.6, f"face cam with {len(people)} people, boxed around them"
    return {"facecam": [round(x / W, 3), round(y / H, 3), round(w / W, 3), round(h / H, 3)],
            "focus_x": 0.5, "mode": "stack", "confidence": round(c * steady, 2), "how": how}


def frame_kind(img):
    """'camera', 'facecam' or 'none' for one frame."""
    H, W = img.shape[:2]
    fs = [f[:4] for f in faces(img)]
    if not fs:
        return "none"
    return "camera" if is_full_camera(fs, W, H) else "facecam"


def layout_segments(frame_paths, times, base, min_run=3.0, box_for=None):
    """Split a clip where its layout changes (gameplay with a face cam, then a full-screen camera...).
    frame_paths/times: one frame per second or so, times in seconds from the clip's start.
    base: suggest()'s answer for the clip. Returns None when the whole clip is one layout,
    else [{"start", "end", "facecam", "focus_x", "mode"}] for layout.json."""
    import cv2
    kinds = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        kinds.append(frame_kind(img) if img is not None else "none")
    if not kinds:
        return None
    kinds = [k if k != "none" else None for k in kinds]  # a second without a face keeps its neighbors' layout
    for i in range(len(kinds)):
        if kinds[i] is None:
            kinds[i] = next((k for k in kinds[i:] if k), None) or next((k for k in reversed(kinds[:i]) if k), "facecam")
    runs = []
    for t, k in zip(times, kinds):
        if runs and runs[-1][1] == k:
            runs[-1][2] = t
        else:
            runs.append([t, k, t])
    step = (times[1] - times[0]) if len(times) > 1 else 1.0
    merged = []  # runs shorter than min_run are editing flashes: fold them into the previous layout
    for r in runs:
        if merged and (r[2] - r[0] + step < min_run or merged[-1][1] == r[1]):
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    if len(merged) < 2:
        return None
    segs = []
    for i, (a, k, b) in enumerate(merged):
        end = merged[i + 1][0] if i + 1 < len(merged) else times[-1] + step
        inside = [p for p, t in zip(frame_paths, times) if a <= t <= b]
        sample = [inside[j] for j in np.linspace(0, len(inside) - 1, min(5, len(inside))).astype(int)] if inside else []
        g = (box_for or suggest)(sample) if sample else base
        if k == "facecam" and g.get("facecam"):
            lay = {"facecam": g["facecam"], "focus_x": base.get("focus_x", 0.5), "mode": "stack"}
        elif k == "camera":
            lay = {"facecam": None, "focus_x": g.get("focus_x", 0.5), "mode": g.get("mode") if g.get("mode") in ("crop", "fit") else "crop"}
        else:
            lay = {"facecam": None, "focus_x": 0.5, "mode": "fit"}
        segs.append({"start": round(float(a), 1), "end": round(float(end), 1), **lay})
    segs[0]["start"] = 0.0
    return segs


if __name__ == "__main__":
    print(json.dumps(suggest(sys.argv[1:]), indent=2))
