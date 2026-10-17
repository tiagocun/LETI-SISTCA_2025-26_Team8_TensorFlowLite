"""
Final Challenge — Sign Language Recognition
===========================================
Pipeline: MediaPipe HandLandmarker → feature extraction → RandomForest → transcription.

This challenge brings together concepts from all previous exercises:
    • Ex1 — video capture and image processing with OpenCV
    • Ex2 — inference with a pre-trained model and label/confidence extraction
    • Ex3 — real-time detection in a camera loop
    • Ex4 — hand landmark detection with MediaPipe

Useful reference: solutions/solução_ex4_mediapipe.py shows how to use the
MediaPipe Hand Landmarker API (the same one used here, already configured).

Your goal is to implement the four sections marked with TODO:

    TODO 1 — extract_features()   : normalise the 21 landmarks into a 64-dim vector
    TODO 2 — SignStore.train()    : train a RandomForestClassifier with scikit-learn
    TODO 3 — SignStore.predict()  : run inference and return (label, confidence)
    TODO 4 — LIVE block           : vote buffer + transcription logic

Everything else (UI, data collection, controls, threading) is already implemented.

Requirements:
    pip install mediapipe opencv-python scikit-learn numpy joblib

Workflow to test:
    1. Run the script.
    2. Press [TAB] to enter COLLECT mode.
    3. Press a letter (e.g. A) and [SPACE] to collect 100 samples.
    4. Repeat for at least one more letter.
    5. Press [1] to train, then [TAB] to switch to LIVE and test transcription.

Controls:
    Q / ESC      - Quit
    TAB          - Toggle COLLECT / LIVE
    A-Z          - Select label (COLLECT mode only)
    SPACE        - Start/cancel auto-collect (100 samples, in COLLECT)
    BACKSPACE    - Delete last transcription character
    P            - Clear transcription
    1            - Train model
    2            - Save dataset + model
    3            - Load dataset + model
    4            - Clear samples for current label
    F            - Toggle FPS
    L            - Toggle landmark IDs
    S            - Screenshot
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import os
import threading
import urllib.request
import json
from collections import deque, Counter
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATH    = "hand_landmarker.task"
MODEL_URL     = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
DATASET_DIR   = "dataset"
DATASET_FILE  = os.path.join(DATASET_DIR, "sign_dataset.json")
CLF_FILE      = os.path.join(DATASET_DIR, "sign_clf.joblib")

# ── Tuning ────────────────────────────────────────────────────────────────────
AUTO_COLLECT_TARGET   = 100      # number of samples per SPACE in COLLECT mode
CAPTURE_INTERVAL      = 0.04     # ~25 Hz → 100 samples in ~4 s
CONF_THRESH           = 0.75     # minimum confidence to consider a prediction
VOTE_MIN              = 5        # minimum matching votes in pred_buffer to commit
RELEASE_FRAMES        = 4        # frames without a hand to consider "released"
MIN_INTER_COMMIT_S    = 0.25     # minimum cooldown between commits (anti-jitter)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ── Hand topology ─────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
FINGER_TIPS = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}

# ── Feature extraction ───────────────────────────────────────────────────────
def extract_features(hand_lm, handedness_label: str) -> np.ndarray:
    """Converts 21 hand landmarks (as seen in ex4) into a 64-dimensional vector,
    invariant to translation and scale.
    Hint: normalise relative to the wrist (landmark 0) and divide by the largest
    Euclidean norm. Append handedness as an extra feature.
    (Same normalisation principle as ex2's pixel /255.0, applied to coordinates.)
    """
    # Convert landmarks to a numpy array (shape: 21x3)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)

    # TODO 1a: Subtract the wrist position (pts[0]) from all points
    #          to make the vector translation-invariant.

    # TODO 1b: If handedness_label == "Left", flip the X axis (pts[:, 0] *= -1)
    #          so that left-hand gestures are normalised the same as right-hand ones.

    # TODO 1c: Compute the Euclidean norm of each point and divide all points by
    #          the maximum value (scale invariance) — skip if the maximum is < 1e-6.
    #          Hint: np.linalg.norm(pts, axis=1)

    # TODO 1d: Flatten the array (pts.flatten()), append 0.0 for "Right" or 1.0
    #          for "Left", and return everything as np.float32.

    pass  # remove this pass and implement the TODOs above


# ── Dataset / model state ────────────────────────────────────────────────────
class SignStore:
    def __init__(self):
        self.samples: dict[str, list[list[float]]] = {}
        self.clf: RandomForestClassifier | None = None
        self.encoder: LabelEncoder | None = None

    def add(self, label: str, feats: np.ndarray):
        self.samples.setdefault(label, []).append(feats.tolist())

    def clear(self, label: str):
        self.samples.pop(label, None)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.samples.items()}

    def train(self) -> str:
        if len(self.samples) < 2:
            return "Need at least 2 classes."
        # Build X (features) and y (labels) from self.samples
        X, y = [], []
        for label, items in self.samples.items():
            X.extend(items)
            y.extend([label] * len(items))
        X = np.asarray(X, dtype=np.float32)

        # TODO 2a: Create a LabelEncoder, fit it on y (list of strings) and store
        #          it in self.encoder. Transform y into encoded labels (integers)
        #          to use during training.

        # TODO 2b: Create a RandomForestClassifier with 300 estimators and store it
        #          in self.clf. Train it with X and the encoded labels.

        # TODO 2c: Compute training accuracy (self.clf.score) and return an
        #          informative string with the number of samples, classes and accuracy.
        #          E.g.: "Trained on 200 samples / 2 classes. Train acc=0.995"

        pass  # remove this pass and implement the TODOs above

    def predict(self, feats: np.ndarray) -> tuple[str, float] | None:
        """Returns (label, confidence) or None if no model has been trained.
        (Similar to decode_predictions from ex2, but with scikit-learn.)
        """
        if self.clf is None or self.encoder is None:
            return None

        # TODO: Use self.clf.predict_proba([feats])[0] to get a probability array.
        #       Find the index with the highest value (np.argmax), convert it to a
        #       label name with self.encoder.inverse_transform, and return
        #       (label_str, confidence_float).

        pass  # remove this pass and implement the TODO above

    def save(self):
        Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
        Path(DATASET_FILE).write_text(json.dumps(self.samples))
        if self.clf is not None and self.encoder is not None:
            joblib.dump({"clf": self.clf, "encoder": self.encoder}, CLF_FILE)

    def load(self) -> str:
        msg = []
        if Path(DATASET_FILE).exists():
            self.samples = json.loads(Path(DATASET_FILE).read_text())
            msg.append(f"dataset ({sum(len(v) for v in self.samples.values())} samples)")
        if Path(CLF_FILE).exists():
            data = joblib.load(CLF_FILE)
            self.clf, self.encoder = data["clf"], data["encoder"]
            msg.append("model")
        return "Loaded: " + ", ".join(msg) if msg else "Nothing to load."

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
                        cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    for tip_id in FINGER_TIPS:
        cx, cy = pts[tip_id]
        cv2.circle(frame, (cx, cy), 8, (0, 255, 180), cv2.FILLED)
    wx, wy = pts[0]
    cv2.putText(frame, label, (wx - 30, wy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)

def draw_banner(frame, text, y_start=0, bg=(30, 30, 30), alpha=0.6):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_start + 36), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (10, y_start + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

def draw_transcription(frame, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text[-60:], (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 2, cv2.LINE_AA)

def draw_progress_bar(frame, pct, label):
    """Barra de progresso da auto-collect."""
    h, w = frame.shape[:2]
    bar_y = 44
    bar_h = 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, bar_y + bar_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (0, bar_y), (int(w * pct), bar_y + bar_h),
                  (0, 180, 255), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, label, (10, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def draw_ready_indicator(frame, ready: bool):
    """Chip no canto superior direito: READY (verde) ou HOLD (laranja)."""
    h, w = frame.shape[:2]
    text  = "READY" if ready else "HOLD"
    color = (0, 220, 0) if ready else (0, 140, 255)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x2, y1 = w - 12, 80
    x1, y2 = x2 - tw - 16, y1 + th + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, text, (x1 + 8, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# ── Async result ──────────────────────────────────────────────────────────────
latest_result = None
result_lock   = threading.Lock()

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_model()
    store = SignStore()

    # Autoload: if a dataset/model already exists on disk, load it.
    if Path(DATASET_DIR).is_dir() and Path(DATASET_FILE).exists():
        print("[AUTOLOAD]", store.load())

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── App state ────────────────────────────────────────────────────────────
    mode             = "LIVE"          # LIVE | COLLECT
    current_label    = "A"
    show_fps         = True
    show_ids         = False
    transcription    = ""
    pred_buffer      = deque(maxlen=8)
    prev_time        = time.time()

    # Auto-collect
    auto_collecting  = False
    auto_count       = 0
    auto_label       = ""
    last_capture_ts  = 0.0

    # Release detection / repeated letters
    last_committed   = None
    last_commit_ts   = 0.0
    no_hand_frames   = 0
    released         = True

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            ts = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarker.detect_async(mp_image, ts)

            with result_lock:
                result = latest_result

            num_hands = 0
            primary_feats = None

            if result and result.hand_landmarks:
                num_hands = len(result.hand_landmarks)
                hand_lm   = result.hand_landmarks[0]
                hd_label  = result.handedness[0][0].display_name
                primary_feats = extract_features(hand_lm, hd_label)
                for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
                    draw_hand(frame, hand_lm, handedness[0].display_name, show_ids)

            # ── Release detection ────────────────────────────────────────────
            # (tracks when the hand leaves the frame — needed for repeated letters)
            if num_hands == 0:
                no_hand_frames += 1
                if no_hand_frames >= RELEASE_FRAMES:
                    if not released:
                        released = True
                        pred_buffer.clear()
            else:
                no_hand_frames = 0

            # ── COLLECT mode: auto-collect ──────────────────────────────────
            if mode == "COLLECT" and auto_collecting and primary_feats is not None:
                now = time.time()
                if now - last_capture_ts >= CAPTURE_INTERVAL:
                    store.add(auto_label, primary_feats)
                    auto_count += 1
                    last_capture_ts = now
                    if auto_count >= AUTO_COLLECT_TARGET:
                        auto_collecting = False
                        print(f"[AUTO] Done: {AUTO_COLLECT_TARGET} samples for "
                              f"'{auto_label}' (total={len(store.samples[auto_label])})")

            # ── LIVE mode: prediction + automatic transcription ─────────────
            if mode == "LIVE" and store.clf is not None and primary_feats is not None:
                # TODO 4: Implement the prediction and transcription logic.
                #
                # Why a vote buffer?
                # A single frame can produce a wrong prediction due to noise or
                # motion. Accumulating several frames and picking the most frequent
                # letter (majority vote) makes the system much more stable.
                #
                # Why release detection?
                # If the user holds their hand still showing "A", we don't want to
                # transcribe "AAAA…". We only commit a repeated letter once the hand
                # has left the frame and come back (released == True).
                #
                # Steps to implement:
                #   1. Call store.predict(primary_feats). If it returns None, do nothing.
                #   2. If confidence >= CONF_THRESH, append the label to pred_buffer.
                #   3. Get the most common element in pred_buffer with Counter().most_common(1).
                #   4. If that element appears >= VOTE_MIN times, consider committing:
                #        • If the letter is DIFFERENT from last_committed and the
                #          MIN_INTER_COMMIT_S cooldown has passed → commit.
                #        • If the letter is the SAME as last_committed, only commit if
                #          released == True AND the cooldown has passed.
                #   5. On commit:
                #        • Append the letter to transcription (or " " for "_SPACE_",
                #          or delete the last char for "_DEL_").
                #        • Update last_committed, last_commit_ts, set released = False.
                #        • Clear pred_buffer.
                #   6. Display the label and confidence on the frame with cv2.putText
                #      (see the commented example below).
                #
                pass  # remove this pass and implement the TODO above

                # Exemplo de como mostrar label + confiança no frame:
                # cv2.putText(frame, f"{label}  {conf*100:.0f}%",
                #             (15, 110), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.9, (0, 255, 255), 2, cv2.LINE_AA)

            # ── FPS ─────────────────────────────────────────────────────────
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            # ── Status banner ───────────────────────────────────────────────
            counts = store.counts()
            n_total = sum(counts.values())
            status = (f"MODE:{mode}  LABEL:'{current_label}' "
                      f"({counts.get(current_label,0)})  "
                      f"classes:{len(counts)} samples:{n_total}  hands:{num_hands}")
            if show_fps:
                status += f"  fps:{fps:.0f}"
            draw_banner(frame, status)

            if auto_collecting:
                pct = auto_count / AUTO_COLLECT_TARGET
                draw_progress_bar(frame, pct,
                                  f"AUTO-COLLECT '{auto_label}'  "
                                  f"{auto_count}/{AUTO_COLLECT_TARGET}  "
                                  f"({pct*100:.0f}%)  [SPACE to cancel]")

            if mode == "LIVE" and store.clf is not None:
                draw_transcription(frame, "▶ " + transcription)
                draw_ready_indicator(frame, released or last_committed is None)

            h, w = frame.shape[:2]
            cv2.putText(frame,
                        "Q:Quit  TAB:LIVE/COLLECT  A-Z:Label  SPACE:Auto-collect  "
                        "BKSP:Del  P:Clear  1:Train  2:Save  3:Load  4:Clear-label",
                        (10, h - 8), cv2.FONT_HERSHEY_PLAIN, 0.85,
                        (130, 130, 130), 1, cv2.LINE_AA)

            cv2.imshow("Sign Language", frame)

            # ── Input ───────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            # BACKSPACE → delete last transcription character
            if key == 8:
                transcription = transcription[:-1]
                continue

            if key in (ord("q"), 27):
                break
            elif key == 9:  # TAB → toggle COLLECT / LIVE
                mode = "COLLECT" if mode != "COLLECT" else "LIVE"
                if mode == "COLLECT":
                    pred_buffer.clear()
                else:
                    auto_collecting = False
                    last_committed = None
                    released       = True
            elif key == ord("p"):
                transcription = ""
                last_committed = None
                released       = True
                pred_buffer.clear()
                print("[CLEAR-TEXT] transcription cleared")
            elif key == ord("1"):
                msg = store.train()
                print("[TRAIN]", msg)
            elif key == ord("2"):
                store.save()
                print("[SAVE] dataset + model")
            elif key == ord("3"):
                print("[LOAD]", store.load())
            elif key == ord("4"):
                store.clear(current_label)
                print(f"[CLEAR] '{current_label}'")
            elif key == ord("f"):
                show_fps = not show_fps
            elif key == ord("l"):
                show_ids = not show_ids
            elif key == ord(" "):
                if mode != "COLLECT":
                    print("[INFO] SPACE only works in COLLECT mode (press TAB).")
                elif auto_collecting:
                    auto_collecting = False
                    print(f"[AUTO] Cancelled at {auto_count}/{AUTO_COLLECT_TARGET} "
                          f"for '{auto_label}'.")
                else:
                    auto_collecting = True
                    auto_label      = current_label
                    auto_count      = 0
                    last_capture_ts = 0.0
                    print(f"[AUTO] Started: target={AUTO_COLLECT_TARGET} "
                          f"for '{auto_label}'.")
            elif key == ord("s"):
                os.makedirs("screenshots", exist_ok=True)
                fname = f"screenshots/sl_{int(time.time())}.png"
                cv2.imwrite(fname, frame)
                print(f"[SHOT] {fname}")
            elif mode == "COLLECT" and not auto_collecting:
                # A-Z / a-z → instantly select label
                if ord('A') <= key <= ord('Z'):
                    current_label = chr(key)
                    print(f"[LABEL] '{current_label}'")
                elif ord('a') <= key <= ord('z'):
                    current_label = chr(key - 32)
                    print(f"[LABEL] '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
