"""
Real-Time Hand Tracking — MediaPipe Tasks API (0.10+), no mp.solutions
=======================================================================
Requirements:
    pip install mediapipe opencv-python

First run auto-downloads hand_landmarker.task (~9 MB).

Controls:
    Q / ESC  - Quit
    S        - Save screenshot
    F        - Toggle FPS
    L        - Toggle landmark IDs
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import os
import threading
import urllib.request

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

# ── Hand connection pairs (21 landmarks) ─────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

FINGER_TIPS = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_hand(frame, hand_lm, label, show_ids):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (180, 180, 180), 2, cv2.LINE_AA)

    for i, (cx, cy) in enumerate(pts):
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), cv2.FILLED)
        if show_ids:
            cv2.putText(frame, str(i), (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.6,
                        (255, 255, 0), 1, cv2.LINE_AA)

    for tip_id in FINGER_TIPS:
        cx, cy = pts[tip_id]
        cv2.circle(frame, (cx, cy), 10, (0, 255, 180), cv2.FILLED)
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 1)

    fingers = count_fingers(hand_lm, label)
    wx, wy = pts[0]
    cv2.putText(frame, f"{label} ({fingers}f)",
                (wx - 30, wy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 200, 0), 2, cv2.LINE_AA)


def count_fingers(hand_lm, label: str) -> int:
    up = 0
    if label == "Right":
        if hand_lm[4].x < hand_lm[3].x: up += 1
    else:
        if hand_lm[4].x > hand_lm[3].x: up += 1
    for tip in [8, 12, 16, 20]:
        if hand_lm[tip].y < hand_lm[tip - 2].y: up += 1
    return up


def draw_banner(frame, text, y_start=0, bg=(30, 30, 30), alpha=0.55):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_start + 36), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (10, y_start + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (220, 220, 220), 1, cv2.LINE_AA)

# ── Shared async result ───────────────────────────────────────────────────────
latest_result = None
result_lock   = threading.Lock()

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
        result_callback=result_callback,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_fps  = True
    show_ids  = False
    prev_time = time.time()
    ss_dir    = "screenshots"

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            timestamp = int(time.time() * 1000)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            landmarker.detect_async(mp_image, timestamp)

            with result_lock:
                result = latest_result

            h, w      = frame.shape[:2]
            num_hands = 0

            if result and result.hand_landmarks:
                num_hands = len(result.hand_landmarks)
                for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].display_name
                    draw_hand(frame, hand_lm, label, show_ids)

            curr_time = time.time()
            fps       = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time

            status = f"Hands: {num_hands}"
            if show_fps:
                status += f"   |   FPS: {fps:.1f}"
            draw_banner(frame, status)

            cv2.putText(frame, "Q/ESC: Quit   S: Screenshot   F: FPS   L: IDs",
                        (10, h - 8), cv2.FONT_HERSHEY_PLAIN, 0.95,
                        (120, 120, 120), 1, cv2.LINE_AA)

            cv2.imshow("Hand Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("f"):
                show_fps = not show_fps
            elif key == ord("l"):
                show_ids = not show_ids
            elif key == ord("s"):
                os.makedirs(ss_dir, exist_ok=True)
                fname = os.path.join(ss_dir, f"hand_{int(time.time())}.png")
                cv2.imwrite(fname, frame)
                print(f"Screenshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()