"""
Final Challenge — Sign Language Recognition
===========================================
Pipeline: MediaPipe HandLandmarker → feature extraction → RandomForest → transcription.
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
AUTO_COLLECT_TARGET   = 100
CAPTURE_INTERVAL      = 0.04
CONF_THRESH           = 0.75
VOTE_MIN              = 5
RELEASE_FRAMES        = 4
MIN_INTER_COMMIT_S    = 0.25

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

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(hand_lm, handedness_label: str) -> np.ndarray:
    """Converts 21 hand landmarks into a 64-dimensional vector,
    invariant to translation and scale."""

    # Convert landmarks to numpy array (shape: 21x3)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)

    # TODO 1a: Subtract wrist (landmark 0) → translation invariance
    pts -= pts[0]

    # TODO 1b: Flip X axis for left hand → mirror to match right-hand representation
    if handedness_label == "Left":
        pts[:, 0] *= -1

    # TODO 1c: Divide by max Euclidean norm → scale invariance
    norms = np.linalg.norm(pts, axis=1)
    max_norm = norms.max()
    if max_norm >= 1e-6:
        pts /= max_norm

    # TODO 1d: Flatten + append handedness flag (0.0 = Right, 1.0 = Left)
    handedness_flag = 1.0 if handedness_label == "Left" else 0.0
    features = np.append(pts.flatten(), handedness_flag).astype(np.float32)

    return features  # shape: (64,)  — 21*3 + 1


# ── Dataset / model state ─────────────────────────────────────────────────────
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

        X, y = [], []
        for label, items in self.samples.items():
            X.extend(items)
            y.extend([label] * len(items))
        X = np.asarray(X, dtype=np.float32)

        # TODO 2a: Encode string labels → integers
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)

        # TODO 2b: Train Random Forest
        self.clf = RandomForestClassifier(n_estimators=300, random_state=42)
        self.clf.fit(X, y_encoded)

        # TODO 2c: Return diagnostic string
        accuracy = self.clf.score(X, y_encoded)
        n_classes = len(self.encoder.classes_)
        return (f"Trained on {len(X)} samples / {n_classes} classes. "
                f"Train acc={accuracy:.3f}")

    def predict(self, feats: np.ndarray) -> tuple[str, float] | None:
        """Returns (label, confidence) or None if no model has been trained."""
        if self.clf is None or self.encoder is None:
            return None

        # TODO 3: predict_proba → argmax → inverse_transform
        proba = self.clf.predict_proba([feats])[0]
        idx = np.argmax(proba)
        confidence = float(proba[idx])
        label = self.encoder.inverse_transform([idx])[0]

        return label, confidence

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
    h, w = frame.shape[:2]
    bar_y = 44
    bar_h = 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, bar_y + bar_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (0, bar_y), (int(w * pct), bar_y + bar_h), (0, 180, 255), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, label, (10, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def draw_ready_indicator(frame, ready: bool):
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

    print("Warming up camera...")
    for _ in range(30):
        cap.read()
        time.sleep(0.05)
    print("Camera ready.")

    mode             = "LIVE"
    current_label    = "A"
    show_fps         = True
    show_ids         = False
    transcription    = ""
    pred_buffer      = deque(maxlen=8)
    prev_time        = time.time()

    auto_collecting  = False
    auto_count       = 0
    auto_label       = ""
    last_capture_ts  = 0.0

    last_committed   = None
    last_commit_ts   = 0.0
    no_hand_frames   = 0
    released         = True

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue # skip bad frames instead of exiting
            frame = cv2.flip(frame, 1)
            ts = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarker.detect_async(mp_image, ts)

            with result_lock:
                result = latest_result

            num_hands     = 0
            primary_feats = None

            if result and result.hand_landmarks:
                num_hands     = len(result.hand_landmarks)
                hand_lm       = result.hand_landmarks[0]
                hd_label      = result.handedness[0][0].display_name
                primary_feats = extract_features(hand_lm, hd_label)
                for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
                    draw_hand(frame, hand_lm, handedness[0].display_name, show_ids)

            # ── Release detection ─────────────────────────────────────────────
            if num_hands == 0:
                no_hand_frames += 1
                if no_hand_frames >= RELEASE_FRAMES:
                    if not released:
                        released = True
                        pred_buffer.clear()
            else:
                no_hand_frames = 0

            # ── COLLECT mode ──────────────────────────────────────────────────
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

            # ── LIVE mode: prediction + transcription ─────────────────────────
            if mode == "LIVE" and store.clf is not None and primary_feats is not None:

                # TODO 4 — implemented below

                # Step 1: get prediction
                prediction = store.predict(primary_feats)
                if prediction is not None:
                    label, conf = prediction

                    # Step 2: add to vote buffer if confident enough
                    if conf >= CONF_THRESH:
                        pred_buffer.append(label)

                    # Step 3: majority vote
                    if pred_buffer:
                        most_common = Counter(pred_buffer).most_common(1)[0]
                        top_label, top_count = most_common

                        # Step 4: check if we should commit
                        if top_count >= VOTE_MIN:
                            now = time.time()
                            cooldown_ok = (now - last_commit_ts) >= MIN_INTER_COMMIT_S

                            different_letter = (top_label != last_committed)
                            same_letter_ok   = (top_label == last_committed and
                                                released and cooldown_ok)

                            if cooldown_ok and (different_letter or same_letter_ok):
                                # Step 5: commit
                                if top_label == "_SPACE_":
                                    transcription += " "
                                elif top_label == "_DEL_":
                                    transcription = transcription[:-1]
                                else:
                                    transcription += top_label

                                last_committed = top_label
                                last_commit_ts = now
                                released       = False
                                pred_buffer.clear()
                                print(f"[COMMIT] '{top_label}' ({conf:.0%})")

                    # Step 6: display current prediction on frame
                    cv2.putText(frame, f"{label}  {conf*100:.0f}%",
                                (15, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 255), 2, cv2.LINE_AA)

            # ── FPS ───────────────────────────────────────────────────────────
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            # ── Status banner ─────────────────────────────────────────────────
            counts  = store.counts()
            n_total = sum(counts.values())
            status  = (f"MODE:{mode}  LABEL:'{current_label}' "
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

            # ── Input ─────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key == 8:  # BACKSPACE
                transcription = transcription[:-1]
                continue

            if key in (ord("q"), 27):
                break
            elif key == 9:  # TAB
                mode = "COLLECT" if mode != "COLLECT" else "LIVE"
                if mode == "COLLECT":
                    pred_buffer.clear()
                else:
                    auto_collecting = False
                    last_committed  = None
                    released        = True
            elif key == ord("p"):
                transcription  = ""
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
                    auto_collecting  = True
                    auto_label       = current_label
                    auto_count       = 0
                    last_capture_ts  = 0.0
                    print(f"[AUTO] Started: target={AUTO_COLLECT_TARGET} "
                          f"for '{auto_label}'.")
            elif key == ord("s"):
                os.makedirs("screenshots", exist_ok=True)
                fname = f"screenshots/sl_{int(time.time())}.png"
                cv2.imwrite(fname, frame)
                print(f"[SHOT] {fname}")
            elif mode == "COLLECT" and not auto_collecting:
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