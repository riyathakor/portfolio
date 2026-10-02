#!/usr/bin/env python3
"""
Pre-extract the head-rotation trajectory from public/character.mp4 into WebP frames.

Step 1 - inspect (always run this first, it never writes frames):
    python scripts/extract_frames.py inspect

    Prints frame count / fps, the background colour, and a motion timeline,
    and writes scripts/_debug/contact_sheet.jpg with every Nth frame labelled by its
    frame number so you can read off the 8 compass poses with your own eyes.
    It also prints "pose plateaus" (spans where motion drops to ~0) as suggestions.

Step 2 - extract:
    python scripts/extract_frames.py extract \
        --anchors UP=12,UP-RIGHT=40,RIGHT=70,DOWN-RIGHT=100,DOWN=130,DOWN-LEFT=160,LEFT=190,UP-LEFT=220 \
        --center 260 \
        --wrap 250

    --anchors  frame numbers of the 8 poses, in the order the head travels (clockwise
               on screen). The script maps the 64 output frames onto the real video
               frames between those anchors, so the in-between angles are real footage.
    --center   frame number of the neutral, looking-at-camera pose (the last frame by default)
    --wrap-start  optional frame where the UP-LEFT -> UP turn begins when it happens at the
               start of the video (use with --wrap = the UP frame)
    --wrap     optional frame where the head is back at UP after UP-LEFT (closes the loop).
               If omitted, UP-LEFT -> UP is filled from UP-LEFT up to the end of the turn.

Output:
    public/frames/f00.webp ... f63.webp   (f00 = UP, index grows clockwise, ~5.625 deg apart)
    public/frames/center.webp
    public/frames/meta.json               (size, background hex, face centre guess)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

VIDEO = "public/character.mp4"
OUT_DIR = "public/frames"
DEBUG_DIR = "scripts/_debug"
N_FRAMES = 64
ORDER = ["UP", "UP-RIGHT", "RIGHT", "DOWN-RIGHT", "DOWN", "DOWN-LEFT", "LEFT", "UP-LEFT"]


def read_all(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Cannot open {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def detect_background(frames):
    """Median colour of the frame border pixels across a sample of frames."""
    samples = []
    step = max(1, len(frames) // 40)
    for f in frames[::step]:
        h, w = f.shape[:2]
        m = max(4, min(h, w) // 50)
        border = np.concatenate([
            f[:m].reshape(-1, 3), f[-m:].reshape(-1, 3),
            f[:, :m].reshape(-1, 3), f[:, -m:].reshape(-1, 3),
        ])
        samples.append(border)
    bgr = np.median(np.concatenate(samples), axis=0).astype(int)
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(h):
    h = h.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], dtype=np.float32)


import base64

# The video has a mouse-cursor overlay baked in (white arrow, dark outline, soft shadow).
# CURSOR_B64 is a grayscale crop of it, CURSOR_MASK_B64 is its silhouette (arrow + shadow).
CURSOR_B64 = "iVBORw0KGgoAAAANSUhEUgAAAB4AAAAtCAAAAACctYR9AAAD7UlEQVQ4EU3BSWyUdRzH4c/3/77vrGXGUqWspS0TFqEG1AMRJXUDTVy4VAIHTSQxngSGIDEBY2I0iCYcVCJwMBQUCrSJKKJGDCGg3IjBaNimbPNiBCtQSpeZeX9ObRCeR0WM/4i7CIwqFTGqTOIuAqNKRYwqk7iLwKhSCAdPzp8Wxc3E/wRGlUL6156Y83JDf51MiBECo0ohPct/zzUubvJczDcHCENgVCnk2opzbUfHL8nhvJgTCENgVCmkN3+xq7C26fVMOVbrORB3KORW/lzX2H2fPLhorJfwhcQdukxfvtA5Mdn12Zwncs6zQCBu02X68oXOiV7foa1TX5jsEcckblNIb767s8FB+7YZbY3UCIQhhimkN9/d2eCgd39H85K6VGBIhhimkJv57s4mzIYyXe1z59e6eGBOjFDIzXx3ZzORieDAx62P3ufHcWKEQm7lC51NgHnG5m9nPjkx5TsHGKCQ/nyhqxEwZ9zY2jXvkdyomCcwQCGDK8/ubUKYjMrfmw/Omj8rG5hcBChkaOWZvc0MMytXrrQfbV4wI+3kRYBCKstP7ckBZliF8sDGw7Ofmhr34gYoJHrj1J4cYIZVHNGFXYdaFjak4s6EikTLT+7NUWVQAnr62n+a1dqS8RAqEq062TEFmSErYRF+z46dLS/OyFR8FFJ568yORvPKeGbcGD2gMld3H5k+rzEdcwqprD27fZL5JQKLrvdeSESlfmefXmhrrU05hZTXFXaN8W1ooC8er/l5w6X6AXonJVIts+9N+SpSevvsl+MGa65vPD93afr06tPTokT2/gzJCaPTMYU2tK57e71/Zce7NQ9tzo7auHdeixdTkCRWlw4U2uD7pzc1//Pe/lRj4c22muOrc88lAueXiWWSgUKLNlzaYu98M71pwpFJ62v71px7dSZDyAWxmKfQru478tKxA1MeGz/mj0OrnlHHh4sXlOJxz/OdUGjBlo7sr60PT0753XueXVf6a835j/xkWr6QFNq1r75P1c8cm02ke3byQTa5bf2yuXW1EkgqEp0qKOuSo5JW+W334lei7pXR0geyMROgIqW+q4Pmp1OxW6kbm/o3jru6/YunFyfTAqSLFpUr0fUg4Xn4pcO7X3v+ZmF1blkyE8iELplVIjNJLgqi7q+9Fd6Pn89clK6NO0BFZBHIYah/6JfvxvqFMQvTdZmUH4GKOAMEEZR6T/xw3B6fXR9ksmknQ0WQIWQG0cD5Y6fGNNUnRyUyKU+GilSJYQalnj8H/YDEPS7lB4CKgBhmQDR4a2DQxQPfj3sOUBHECENE5chA5vkMUxExwgSYQQSSTICK4m4GhrNIzqj6F3xtq3hfJXe3AAAAAElFTkSuQmCC"
CURSOR_MASK_B64 = "iVBORw0KGgoAAAANSUhEUgAAAB4AAAAtCAAAAACctYR9AAAArklEQVQ4EX3BiUHDQBAAMU3/RS82ASd+LlKMrMTISoysxHjJXYyX3MX4l6sYh1zEeMtZjLecxfiUTzE+5VOMs7zFOMtbjIscYlzkEOMq/2Lc5E+Mu7zEuMtLjGchxrMQ41mIsZIYK4mxkhgribEWYy3GSmKsJMajbGI8yS7Gk+xiPMkuxpPsYjzKJsajbGKQX+MtmxjkZRyyyck4ZJOTccgmZ+NPdrkau/zKV/nqB3oCLSy+/UrEAAAAAElFTkSuQmCC"
SCALES = [round(x, 2) for x in np.arange(0.4, 1.46, 0.05)]  # the cursor changes size across the video
_TPL = None


def _templates():
    global _TPL
    if _TPL is None:
        dec = lambda b: cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_GRAYSCALE)
        g, m = dec(CURSOR_B64), dec(CURSOR_MASK_B64)
        rs = lambda im, k: cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        _TPL = [(rs(g, k), rs(m, k)) for k in SCALES]
    return _TPL


def find_cursors(frame, thresh=0.6):
    """[(x, y, w, h, score, mask)] for every cursor-shaped overlay in the frame."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = []
    for tp, mk in _templates():
        r = cv2.matchTemplate(g, tp, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(r > thresh)
        found += [(float(r[y, x]), int(x), int(y), tp, mk) for y, x in zip(ys, xs)]
    found.sort(key=lambda t: -t[0])
    keep = []
    for sc, x, y, tp, mk in found:
        if all(abs(x - k[0]) > 20 or abs(y - k[1]) > 20 for k in keep):
            keep.append((x, y, tp.shape[1], tp.shape[0], sc, mk))
    return keep


def remove_cursors(frame, hits):
    """Inpaint the cursors in `hits`. Over plain background the area around the cursor is
    wiped (it also removes the glow); over the character only its silhouette is erased."""
    if not hits:
        return frame
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for x, y, w, h, _, mk in hits:
        m = 12 + int(0.25 * max(w, h))
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
        region = frame[y0:y1, x0:x1].astype(np.float32)
        med = np.median(region.reshape(-1, 3), axis=0)
        plain = np.mean(np.linalg.norm(region - med, axis=2) < 26)
        if plain > 0.85:  # background only
            mask[y0:y1, x0:x1] = 255
        else:  # touching hair / skin / shirt: erase just the cursor silhouette
            sil = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            ys, xs = min(h, H - y), min(w, W - x)
            mask[y:y + ys, x:x + xs] |= sil[:ys, :xs]
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


def detect_cursors_all(frames, thresh=0.6):
    """Cursor hits for every frame, plus the 'dirty' frames: transition frames inside the span
    where the cursor is on screen but is too smeared to match (it becomes a grey smudge)."""
    hits = []
    for i, f in enumerate(frames):
        hits.append(find_cursors(f, thresh))
        if i % 40 == 0:
            print(f"  scanning for cursors... {i}/{len(frames)}")
    have = [i for i, h in enumerate(hits) if h]
    dirty = set()
    if have:
        dirty = {i for i in range(min(have), max(have) + 1) if not hits[i]}
    return hits, dirty


def motion_curve(frames):
    small = [cv2.cvtColor(cv2.resize(f, (160, int(160 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)
             for f in frames]
    d = [0.0] + [float(np.mean(cv2.absdiff(small[i], small[i - 1]))) for i in range(1, len(small))]
    return np.array(d)


def plateaus(motion, thresh_ratio=0.25, min_len=3):
    """Spans where motion is low, i.e. the head is holding a pose."""
    t = np.max(motion) * thresh_ratio
    spans, start = [], None
    for i, m in enumerate(motion):
        if m < t and start is None:
            start = i
        elif m >= t and start is not None:
            if i - start >= min_len:
                spans.append((start, i - 1))
            start = None
    if start is not None and len(motion) - start >= min_len:
        spans.append((start, len(motion) - 1))
    return spans


def contact_sheet(frames, every, path, cols=10, thumb_w=160):
    idx = list(range(0, len(frames), every))
    h, w = frames[0].shape[:2]
    th = int(thumb_w * h / w)
    rows = (len(idx) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * thumb_w, 3), np.uint8)
    for k, i in enumerate(idx):
        t = cv2.resize(frames[i], (thumb_w, th))
        cv2.putText(t, str(i), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(t, str(i), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * thumb_w:(c + 1) * thumb_w] = t
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def cmd_inspect(_):
    frames, fps = read_all(VIDEO)
    h, w = frames[0].shape[:2]
    print(f"{VIDEO}: {len(frames)} frames, {fps:.2f} fps, {w}x{h}, {len(frames) / fps:.2f}s")
    print("background:", detect_background(frames))
    m = motion_curve(frames)
    print("pose plateaus (start-end), candidates for the 8 poses + neutral:")
    for s, e in plateaus(m):
        print(f"  {s}-{e}  (middle: {(s + e) // 2})")
    every = max(1, len(frames) // 60)
    path = os.path.join(DEBUG_DIR, "contact_sheet.jpg")
    contact_sheet(frames, every, path)
    print(f"contact sheet every {every} frames -> {path}")


def parse_anchors(s):
    pairs = dict(p.split("=") for p in s.split(","))
    missing = [k for k in ORDER if k not in pairs]
    if missing:
        sys.exit(f"Missing anchors: {missing}")
    return [int(pairs[k]) for k in ORDER]


def cmd_extract(a):
    frames, _ = read_all(VIDEO)
    n = len(frames)
    anchors = parse_anchors(a.anchors)
    center = a.center if a.center is not None else n - 1
    end_of_turn = a.wrap if a.wrap is not None else max(anchors[-1], min(center, n) - 1)
    knots = anchors + [end_of_turn]  # 9 knots: UP ... UP-LEFT, back to UP
    seg = N_FRAMES / 8  # 8 output frames per compass segment

    os.makedirs(OUT_DIR, exist_ok=True)
    bg_hex = detect_background(frames)

    hits, dirty = [[] for _ in frames], set()
    if not a.no_clean:
        hits, dirty = detect_cursors_all(frames, a.cursor_thresh)
        print(f"cursors found in {sum(1 for h in hits if h)} frames; "
              f"{len(dirty)} smeared transition frames will be skipped: {sorted(dirty)}")

    def snap(i):
        """Nearest source frame that is not a smeared transition frame."""
        if i not in dirty:
            return i
        for d in range(1, 8):
            for j in (i - d, i + d):
                if 0 <= j < n and j not in dirty:
                    return j
        return i

    cleaned = {}

    def get(i):
        if i not in cleaned:
            cleaned[i] = frames[i] if a.no_clean else remove_cursors(frames[i], hits[i])
        return cleaned[i]

    picked = []
    for i in range(N_FRAMES):
        s = min(int(i // seg), 7)
        t = (i - s * seg) / seg
        start = knots[s]
        if s == 7 and a.wrap_start is not None:
            start = a.wrap_start  # UP-LEFT -> UP can come from the start of the video
        src = int(round(start + t * (knots[s + 1] - start)))
        src = snap(max(0, min(n - 1, src)))
        picked.append(src)
        cv2.imwrite(os.path.join(OUT_DIR, f"f{i:02d}.webp"), get(src), [cv2.IMWRITE_WEBP_QUALITY, 92])
    cv2.imwrite(os.path.join(OUT_DIR, "center.webp"), get(snap(center)), [cv2.IMWRITE_WEBP_QUALITY, 95])

    h, w = frames[0].shape[:2]
    meta = {
        "width": w, "height": h, "count": N_FRAMES,
        "background": bg_hex,
        "faceCenter": {"x": a.face_x, "y": a.face_y},  # fractions of the frame; tweak by eye
        "sourceFrames": picked,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {N_FRAMES} frames + center.webp to {OUT_DIR}; background {bg_hex}")

    contact_sheet([get(p) for p in picked], 1, os.path.join(DEBUG_DIR, "result_sheet.jpg"), cols=16, thumb_w=110)
    print(f"check the result: {os.path.join(DEBUG_DIR, 'result_sheet.jpg')} (should sweep a smooth circle, no cursors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    e = sub.add_parser("extract")
    e.add_argument("--anchors", required=True)
    e.add_argument("--center", type=int)
    e.add_argument("--wrap", type=int)
    e.add_argument("--wrap-start", type=int)
    e.add_argument("--no-clean", action="store_true", help="skip cursor removal")
    e.add_argument("--cursor-thresh", type=float, default=0.6, help="match score for a cursor (lower = more aggressive)")
    e.add_argument("--face-x", type=float, default=0.5)
    e.add_argument("--face-y", type=float, default=0.38)
    e.set_defaults(fn=cmd_extract)
    args = ap.parse_args()
    args.fn(args)