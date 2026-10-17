import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

import os
import urllib.request
import threading
import time

mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')
yolo_model = YOLO("yolov8n.pt")

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

latest_result = None
result_lock = threading.Lock()

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

def open_hand_detected(hand_landmarks, handedness):
    landmarks = hand_landmarks
    opened_fingers = 0
    if handedness == "Right":
        if landmarks[4].x > landmarks[3].x:
            opened_fingers += 1
    else:
        if landmarks[4].x < landmarks[3].x:
            opened_fingers += 1
    
    finger_tips = [8, 12, 16, 20]

    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            opened_fingers += 1

    return opened_fingers >= 5

def main():
    print("START")
    ensure_model()
    print("MODEL OK")

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=result_callback
    )

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("CREATING LANDMARKER")

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        
        print("LANDMARKER OK")

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            frame = cv2.flip(frame, 1)

            timestamp = int(time.time() * 1000)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # 1. Send frame to MediaPipe asynchronously
            landmarker.detect_async( # TODO )

            with result_lock:
                result = latest_result
            
            system_active = False

            if result and result.hand_landmarks:

                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    
                    handedness = result.handedness[i][0].category_name
                    
                    # 2. Check if hand is open and activate system
                    if # TODO:

                    height, width = frame.shape[:2]

                    # 3. Draw landmarks on frame
                    for landmark in hand_landmarks:

                        x = int(# TODO)
                        y = int(# TODO)

                        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

            if system_active:
                results_yolo = yolo_model(frame)
                
                for result_yolo in results_yolo:
                    boxes = result_yolo.boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            img_resized = cv2.resize(roi, (224, 224))
                            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
                            img_array = np.expand_dims(img_array, axis=0)

                            predictions = mobilenet_model.predict(img_array, verbose=0)
                            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

                            label = decoded[0][1]
                            confidence = decoded[0][2]

                            result_text = f"{label} ({confidence*100:.1f}%)"

                            cv2.putText(frame, result_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                cv2.putText(frame, "SYSTEM ACTIVE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
            else:
                cv2.putText(frame, "Show open hand to activate", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
            cv2.imshow("Integrated AI System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

# 4. Start the program
if __name__ == # TODO: