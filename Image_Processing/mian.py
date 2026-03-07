# ──────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────
CAMERA_SOURCE        = 0
USE_DEPTH_ESTIMATION = True
DEPTH_MODEL_TYPE     = "MiDaS_small"
YOLO_MODEL           = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
OUTPUT_JSON_FILE     = "Image_Processing/anchor_output.json"
OUTPUT_LOG_FILE      = "Image_Processing/anchor_log.jsonl"

CAMERA_FX = 800.0
CAMERA_FY = 800.0
CAMERA_CX = 640.0
CAMERA_CY = 360.0

# How many pixels outside the bounding box an anchor is allowed
# to drift before it gets reset to the fresh detection point.
ANCHOR_DRIFT_TOLERANCE = 20   # pixels

# Minimum pixel distance between two feature anchor points
FEATURE_MIN_DIST = 5  # pixels

# ──────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────
import cv2
import json
import time
import numpy as np
import torch
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────────────────────

def load_yolo(model_path):
    print(f"[INIT] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print("[INIT] YOLO ready.")
    return model


def load_midas(model_type):
    if not USE_DEPTH_ESTIMATION:
        return None, None
    print(f"[INIT] Loading MiDaS depth model: {model_type}")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    if torch.cuda.is_available():
        midas = midas.cuda()
        print("[INIT] MiDaS running on GPU.")
    else:
        print("[INIT] MiDaS running on CPU.")
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = (transforms.small_transform if "small" in model_type
                 else transforms.dpt_transform)
    print("[INIT] MiDaS ready.")
    return midas, transform


# ──────────────────────────────────────────────────────────────
#  DEPTH ESTIMATION
# ──────────────────────────────────────────────────────────────

def estimate_depth(frame, midas, transform):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    with torch.no_grad():
        pred = midas(input_tensor)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=frame.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
    depth = pred.cpu().numpy()
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 0:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


# ──────────────────────────────────────────────────────────────
#  ANCHOR EXTRACTION
# ──────────────────────────────────────────────────────────────

BOX_ANCHOR_NAMES = [
    "center", "top_left", "top_right",
    "bottom_left", "bottom_right",
    "top_center", "bottom_center",
]

def extract_box_anchors(x1, y1, x2, y2):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return {
        "center":        (cx, cy),
        "top_left":      (x1, y1),
        "top_right":     (x2, y1),
        "bottom_left":   (x1, y2),
        "bottom_right":  (x2, y2),
        "top_center":    (cx, y1),
        "bottom_center": (cx, y2),
    }


def deduplicate_points(points, min_dist=FEATURE_MIN_DIST):
    """
    Remove points that are within min_dist pixels of an already-
    accepted point.  Fixes duplicate feature anchors (Bug 5).
    """
    kept = []
    for p in points:
        too_close = False
        for k in kept:
            if abs(p[0] - k[0]) < min_dist and abs(p[1] - k[1]) < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append(p)
    return kept


def extract_feature_anchors(frame, x1, y1, x2, y2, max_points=8):
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    orb  = cv2.ORB_create(nfeatures=max_points * 2)  # over-detect then filter
    kps, _ = orb.detectAndCompute(gray, None)
    pts = [(int(kp.pt[0]) + x1c, int(kp.pt[1]) + y1c) for kp in kps]
    pts = deduplicate_points(pts)   # remove duplicates  ← Bug 5 fix
    return pts[:max_points]


def lift_to_3d(u, v, depth_map, fx, fy, cx, cy):
    h, w = depth_map.shape
    uc = int(np.clip(u, 0, w - 1))
    vc = int(np.clip(v, 0, h - 1))
    z  = float(depth_map[vc, uc])
    return (round((u - cx) * z / fx, 6),
            round((v - cy) * z / fy, 6),
            round(z, 6))


# ──────────────────────────────────────────────────────────────
#  ANCHOR DRIFT CORRECTION  (fixes Bug 4)
# ──────────────────────────────────────────────────────────────

def clamp_box_anchors(tracked_box, fresh_box, x1, y1, x2, y2,
                      tol=ANCHOR_DRIFT_TOLERANCE):
    """
    For each named anchor, if the tracked pixel position has drifted
    more than `tol` pixels outside the current bounding box, reset
    it to the freshly computed position from the YOLO detection.

    This prevents anchors from flying off the object when the camera
    or object moves quickly and optical flow loses the point.
    """
    corrected = {}
    for name in BOX_ANCHOR_NAMES:
        tu, tv = tracked_box.get(name, fresh_box[name])
        # Check if drifted outside bounding box + tolerance
        if (tu < x1 - tol or tu > x2 + tol or
                tv < y1 - tol or tv > y2 + tol):
            corrected[name] = fresh_box[name]   # reset to fresh
        else:
            corrected[name] = (tu, tv)
    return corrected


# ──────────────────────────────────────────────────────────────
#  IoU-BASED STABLE ID ASSIGNMENT
# ──────────────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter)


class StableIDAssigner:
    def __init__(self, iou_thresh=0.35):
        self.iou_thresh  = iou_thresh
        self.next_id     = 0
        self.prev_boxes  = {}
        self.prev_labels = {}

    def assign(self, detections):
        used_prev = set()
        for det in detections:
            best_iou = self.iou_thresh
            best_sid = None
            for sid, pbox in self.prev_boxes.items():
                if sid in used_prev:
                    continue
                if self.prev_labels.get(sid) != det["label"]:
                    continue
                score = compute_iou(det["box"], pbox)
                if score > best_iou:
                    best_iou = score
                    best_sid = sid
            if best_sid is not None:
                det["stable_id"] = best_sid
                used_prev.add(best_sid)
            else:
                det["stable_id"] = f"{det['label']}_{self.next_id}"
                self.next_id += 1

        self.prev_boxes  = {d["stable_id"]: d["box"]   for d in detections}
        self.prev_labels = {d["stable_id"]: d["label"] for d in detections}
        return detections


# ──────────────────────────────────────────────────────────────
#  OPTICAL FLOW TRACKER
# ──────────────────────────────────────────────────────────────

class AnchorTracker:
    def __init__(self):
        self.prev_gray = None
        self.history   = {}   # {sid: {"box_pts": [...], "feat_pts": [...]}}
        self.lk_params = dict(
            winSize  = (21, 21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30, 0.01),
        )

    def update(self, frame, detections):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked = {}

        for det in detections:
            sid      = det["stable_id"]
            box_a    = det["box_anchors"]
            feat_a   = det["feature_anchors"]
            box_pts  = list(box_a.values())   # 7 ordered points
            feat_pts = list(feat_a)
            x1, y1, x2, y2 = det["box"]

            if self.prev_gray is not None and sid in self.history:
                prev_box_pts  = self.history[sid]["box_pts"]
                prev_feat_pts = self.history[sid]["feat_pts"]
                all_prev      = prev_box_pts + prev_feat_pts
                n_box         = len(prev_box_pts)
                n_feat        = len(prev_feat_pts)

                if n_box > 0:
                    prev_arr = np.array(all_prev,
                                        dtype=np.float32).reshape(-1, 1, 2)
                    curr_arr, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, prev_arr, None, **self.lk_params
                    )
                    status = status.flatten()

                    # Box anchors
                    tracked_box_raw = {}
                    new_box_pts     = []
                    for idx, name in enumerate(BOX_ANCHOR_NAMES):
                        if idx < n_box and status[idx]:
                            pt = (int(curr_arr[idx][0][0]),
                                  int(curr_arr[idx][0][1]))
                        else:
                            pt = box_pts[idx] if idx < len(box_pts) \
                                 else box_a[name]
                        tracked_box_raw[name] = pt
                        new_box_pts.append(pt)

                    # ── Drift correction (Bug 4 fix) ──────────────
                    tracked_box = clamp_box_anchors(
                        tracked_box_raw, box_a, x1, y1, x2, y2
                    )
                    # Sync new_box_pts with clamped values
                    new_box_pts = [tracked_box[n] for n in BOX_ANCHOR_NAMES]

                    # Feature anchors
                    new_feat_pts = []
                    for idx in range(n_feat):
                        j = n_box + idx
                        if j < len(status) and status[j]:
                            fp = (int(curr_arr[j][0][0]),
                                  int(curr_arr[j][0][1]))
                            # Only keep feature if still inside bbox
                            if (x1 - ANCHOR_DRIFT_TOLERANCE <= fp[0]
                                    <= x2 + ANCHOR_DRIFT_TOLERANCE and
                                    y1 - ANCHOR_DRIFT_TOLERANCE <= fp[1]
                                    <= y2 + ANCHOR_DRIFT_TOLERANCE):
                                new_feat_pts.append(fp)

                    # Deduplicate tracked feature points too
                    new_feat_pts = deduplicate_points(new_feat_pts)

                    if sum(status[:n_box]) >= 4:
                        tracked[sid]      = {"box":      tracked_box,
                                             "features": new_feat_pts}
                        self.history[sid] = {"box_pts":  new_box_pts,
                                             "feat_pts": new_feat_pts}
                        continue

            # Fresh detection fallback
            tracked[sid]      = {"box": box_a, "features": feat_a}
            self.history[sid] = {"box_pts":  box_pts, "feat_pts": feat_pts}

        # Prune stale IDs
        active = {d["stable_id"] for d in detections}
        for sid in list(self.history):
            if sid not in active:
                del self.history[sid]

        self.prev_gray = gray
        return tracked


# ──────────────────────────────────────────────────────────────
#  3D CUBOID OVERLAY
# ──────────────────────────────────────────────────────────────

def project_3d_box(frame, box_anchors_3d, color):
    def pt(name):
        a = box_anchors_3d.get(name)
        return (int(a["px"]), int(a["py"])) if a else None

    tl = pt("top_left");    tr = pt("top_right")
    bl = pt("bottom_left"); br = pt("bottom_right")
    tc = pt("top_center");  bc = pt("bottom_center")

    if None in (tl, tr, bl, br):
        return frame

    # Front face
    cv2.line(frame, tl, tr, color, 2)
    cv2.line(frame, tr, br, color, 2)
    cv2.line(frame, br, bl, color, 2)
    cv2.line(frame, bl, tl, color, 2)

    # Rear face
    cx_  = (tl[0] + br[0]) // 2
    cy_  = (tl[1] + br[1]) // 2
    s    = 0.20
    dim  = tuple(max(0, c - 80) for c in color)

    def rear(p):
        return (int(p[0] + (cx_ - p[0]) * s),
                int(p[1] + (cy_ - p[1]) * s))

    rtl, rtr = rear(tl), rear(tr)
    rbl, rbr = rear(bl), rear(br)

    cv2.line(frame, rtl, rtr, dim, 1)
    cv2.line(frame, rtr, rbr, dim, 1)
    cv2.line(frame, rbr, rbl, dim, 1)
    cv2.line(frame, rbl, rtl, dim, 1)
    for f, r in [(tl, rtl), (tr, rtr), (bl, rbl), (br, rbr)]:
        cv2.line(frame, f, r, dim, 1)

    if tc and bc:
        cv2.line(frame, tc, bc, color, 1)

    return frame


# ──────────────────────────────────────────────────────────────
#  OUTPUT FORMATTING
# ──────────────────────────────────────────────────────────────

def build_payload(frame_id, timestamp, label, obj_id, confidence,
                  tracked_anchors, depth_map, fx, fy, cx, cy):
    box_anchors_3d = {}
    for name, (u, v) in tracked_anchors["box"].items():
        if depth_map is not None:
            x3, y3, z3 = lift_to_3d(u, v, depth_map, fx, fy, cx, cy)
        else:
            x3, y3, z3 = float(u), float(v), 0.0
        box_anchors_3d[name] = {"px": u, "py": v,
                                 "x": x3, "y": y3, "z": z3}

    feature_anchors_3d = []
    for idx, (u, v) in enumerate(tracked_anchors["features"]):
        if depth_map is not None:
            x3, y3, z3 = lift_to_3d(u, v, depth_map, fx, fy, cx, cy)
        else:
            x3, y3, z3 = float(u), float(v), 0.0
        feature_anchors_3d.append({
            "id": f"feat_{idx:02d}",
            "px": u, "py": v, "x": x3, "y": y3, "z": z3,
        })

    return {
        "frame_id":        frame_id,
        "timestamp":       round(timestamp, 4),
        "object_id":       obj_id,
        "label":           label,
        "confidence":      round(confidence, 4),
        "box_anchors":     box_anchors_3d,
        "feature_anchors": feature_anchors_3d,
    }


# ──────────────────────────────────────────────────────────────
#  VISUALIZATION
# ──────────────────────────────────────────────────────────────

COLORS = [
    (0, 200, 255), (255, 100, 0), (0, 255, 100),
    (200, 0, 255), (255, 220, 0), (0, 150, 255),
]

def draw_detection(frame, label, confidence, box,
                   tracked_anchors, box_anchors_3d, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for name, (u, v) in tracked_anchors["box"].items():
        cv2.circle(frame, (u, v), 5, color, -1)
        cv2.circle(frame, (u, v), 5, (255, 255, 255), 1)
        cv2.putText(frame, name[:3], (u + 6, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    for (u, v) in tracked_anchors["features"]:
        cv2.drawMarker(frame, (u, v), (255, 255, 255),
                       cv2.MARKER_CROSS, 8, 1)

    # frame = project_3d_box(frame, box_anchors_3d, color)
    return frame


def draw_hud(frame, frame_id, fps, num_objects):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, f"Frame:   {frame_id:05d}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    cv2.putText(frame, f"FPS:     {fps:.1f}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    cv2.putText(frame, f"Objects: {num_objects}",
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    cv2.putText(frame, "Q=quit  S=snapshot  D=toggle depth",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1)
    return frame


# ──────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def main():
    yolo_model       = load_yolo(YOLO_MODEL)
    midas, transform = load_midas(DEPTH_MODEL_TYPE)
    tracker          = AnchorTracker()
    id_assigner      = StableIDAssigner()

    print(f"[INIT] Opening camera: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera: {CAMERA_SOURCE}\n"
            "  PC webcam → CAMERA_SOURCE = 0\n"
            "  Phone     → CAMERA_SOURCE = 'http://IP:PORT/video'"
        )
    print("[INIT] Ready.  Q=quit  S=snapshot  D=toggle depth\n")

    frame_id   = 0
    fps        = 0.0
    t_prev     = time.time()
    depth_map  = None
    depth_skip = 0
    show_depth = True
    log_file   = open(OUTPUT_LOG_FILE, "a")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed, retrying...")
                time.sleep(0.05)
                continue

            frame_id += 1

            # Depth every 3 frames
            if USE_DEPTH_ESTIMATION and midas is not None:
                depth_skip += 1
                if depth_skip >= 3:
                    depth_map  = estimate_depth(frame, midas, transform)
                    depth_skip = 0

            # Detection
            results    = yolo_model(frame, conf=CONFIDENCE_THRESHOLD,
                                    verbose=False)
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = yolo_model.names[int(box.cls)]
                conf  = float(box.conf)
                detections.append({
                    "label":           label,
                    "confidence":      conf,
                    "box":             (x1, y1, x2, y2),
                    "box_anchors":     extract_box_anchors(x1, y1, x2, y2),
                    "feature_anchors": extract_feature_anchors(
                                           frame, x1, y1, x2, y2),
                })

            # Stable IDs
            detections = id_assigner.assign(detections)

            # Tracking
            all_tracked = tracker.update(frame, detections)

            # ── Collect ALL payloads first, THEN write JSON ────
            # (Bug 3 fix: do NOT write inside this loop)
            all_payloads = []
            for i, det in enumerate(detections):
                sid      = det["stable_id"]
                tracked_a = all_tracked.get(sid, {
                    "box":      det["box_anchors"],
                    "features": det["feature_anchors"],
                })
                color   = COLORS[i % len(COLORS)]
                payload = build_payload(
                    frame_id, time.time(),
                    det["label"], sid, det["confidence"],
                    tracked_a, depth_map,
                    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
                )
                all_payloads.append(payload)

                frame = draw_detection(
                    frame, det["label"], det["confidence"],
                    det["box"], tracked_a, payload["box_anchors"], color,
                )

            # ── Single write after all objects are processed ───
            frame_output = {
                "frame_id":  frame_id,
                "timestamp": round(time.time(), 4),
                "objects":   all_payloads,          # ALL objects here
            }
            with open(OUTPUT_JSON_FILE, "w") as f:
                json.dump(frame_output, f, indent=2)
            log_file.write(json.dumps(frame_output) + "\n")
            log_file.flush()

            # FPS
            t_now  = time.time()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # HUD + depth minimap
            frame = draw_hud(frame, frame_id, fps, len(detections))
            if show_depth and depth_map is not None:
                dvis   = (depth_map * 255).astype(np.uint8)
                dvis   = cv2.applyColorMap(dvis, cv2.COLORMAP_MAGMA)
                h, w   = frame.shape[:2]
                th, tw = h // 5, w // 5
                dvis   = cv2.resize(dvis, (tw, th))
                frame[h - th:h, w - tw:w] = dvis
                cv2.putText(frame, "Depth",
                            (w - tw + 5, h - th + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)

            cv2.imshow("Anchor Detection Pipeline v3", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                name = f"snapshot_{frame_id:05d}.png"
                cv2.imwrite(name, frame)
                print(f"[INFO] Snapshot: {name}")
            elif key == ord('d'):
                show_depth = not show_depth

            if frame_id % 30 == 0:
                ids = [d["stable_id"] for d in detections]
                print(f"[{frame_id:05d}] FPS={fps:.1f}  "
                      f"Objects={len(detections)}  IDs={ids}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"\n[DONE] frame={frame_id}  "
              f"log={OUTPUT_LOG_FILE}  json={OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    main()
