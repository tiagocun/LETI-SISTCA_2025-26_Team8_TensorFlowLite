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

# ===== [ MODELS ] ===============================================================================================================
# MovileNetV2 model pre-treined with the ImageNet dataset to calssify the objects detected by YOLO
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# YOLOv8 nano model for real-time object detection
yolo_model = YOLO("yolov8n.pt")

# ===== [ MEDIAPIPE MODEL ] ======================================================================================================
# Local path where the MediaPipe model will be sotred
MODEL_PATH = "hand_landmarker.task"

# MediaPipe model download URL
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Ensures the model is downloaded
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

# ===== [ ASYNC RESULT ] =========================================================================================================
# MediaPipe Tasks API works asynchronously so the latest detection result must be stored globally
latest_result = None

# Thread lock used to safely access the shared result
result_lock = threading.Lock()

# Callback function called whenever MediaPipe finishes a detection
def result_callback(result, output_image, timestamp_ms):
    global latest_result

    # Update of the latest result
    with result_lock:
        latest_result = result

# ===== [ OPEN HAND DETECTION ] ==================================================================================================
# FUnction that identifies if user's hand is open
def open_hand_detected(hand_landmarks, handedness):
    
    landmarks = hand_landmarks

    # Counter of opened fingers
    opened_fingers = 0

    # The thumb opens horizontally, so its X coordinate is analyzed instead of Y
    # 4: thumb
    # Different logic for each end due to the direction in which the thumbs open
    # Right hand
    if handedness == "Right":
        if landmarks[4].x > landmarks[3].x:
            opened_fingers += 1
    
    # Left hand
    else:
        if landmarks[4].x < landmarks[3].x:
            opened_fingers += 1
    
    # Landmark IDs of the rest of the fingertips
    # 8: index finger, 12: middle finger, 16: ring finger, 20: pinky
    finger_tips = [8, 12, 16, 20]

    
    # For each finger: the finger is considered open if the fingertip is above the middle joint
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            opened_fingers += 1

    # Hand is considered open if at least 5 fingers are detected as open
    return opened_fingers >= 5

# ===== [ MAIN ] =================================================================================================================
def main():

    # Ensures the MediaPipe model exists before starting
    print("START")
    ensure_model()
    print("MODEL OK")
    
    # Configuration of the MediaPipe hand tracker
    options = mp_vision.HandLandmarkerOptions(

        # Definition of the model file
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        
        # Live webcam stream mode
        running_mode=mp_vision.RunningMode.LIVE_STREAM,

        # Maximum number of hands to track
        num_hands=2,

        # Minimum confidence required to detect a hand
        min_hand_detection_confidence=0.7,

        # Minimum confidence required to continue tracking the hand
        min_tracking_confidence=0.7,

        # Callback fucntion executed asynchronously
        result_callback=result_callback
    )

    targeted_classes = [id for id, name in yolo_model.names.items() if name != 'person']

    # Connection to the camera's hardware
    cap = cv2.VideoCapture(0)

    # Window dimension
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("CREATING LANDMARKER")

    # Creates the MediaPipe hand tracking system
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        
        print("LANDMARKER OK")

        while True:

            # cam.read() return two values:
            # 'net': boolean that indicates if the frame was successfully read
            # 'frame': image matrix
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Horizontal flip for a more natural mirror effect
            # If the hand tracking system of the thumbs is inverted (finger has to be closed for the system to read the hand)
            # comment the following line
            frame = cv2.flip(frame, 1)

            # ===== [ MEDIAPIPE ] ================================================================================================
            # Timestamp required by MediaPipe Tasks API
            timestamp = int(time.time() * 1000)

            # Conversion from OpenCV BGR format to MediaPipe RGB format
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # Sends the image to MediaPipe asynchronously
            landmarker.detect_async(mp_image, timestamp)

            # Retrieval of the latest detection result
            with result_lock:
                result = latest_result
            
            # System disabled by default
            system_active = False

            # ===== [ HAND ANALYSIS ] ============================================================================================
            # Checks if a hand was detected
            if result and result.hand_landmarks:

                # Iterates through all detected hands
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    
                    handedness = result.handedness[i][0].category_name
                    
                    # Activates the AI system if an open hand is detected
                    if open_hand_detected(hand_landmarks, handedness):
                        system_active = True

                    # Acquisition of the frame dimensions
                    height, width = frame.shape[:2]

                    # Draw of all hand landmarks
                    for landmark in hand_landmarks:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

            # ===== [ AI SYSTEM ] ================================================================================================            
            # YOLO and TensorFlow only execute if the hand gesture activates the system
            if system_active:
                
                # YOLO object detection
                results_yolo = yolo_model(frame, classes=targeted_classes)
                
                # Iteration through YOLO results
                for result_yolo in results_yolo:

                    boxes = result_yolo.boxes
                    
                    # Iteration through ll detected objects
                    for box in boxes:

                        # Extraction of the bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw of the detection rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # ===== [ CLASSIFICATION ] ===============================================================================
                        # Extraction of only the detected object region
                        roi = frame[y1:y2, x1:x2]
                        
                        # Ensures the ROI (Region Of Interest) is valid before classification
                        if roi.size > 0:

                            # Resize to MobileNetV2 input size
                            img_resized = cv2.resize(roi, (224, 224))

                            # Pixel values (0-255) scaled to the range the model uses (-1 to 1)
                            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
                            
                            # TensorFlow expects a list of images
                            # Dimension added to turn one image (224, 224, 3) into a list of one image (1, 224, 224, 3)
                            img_array = np.expand_dims(img_array, axis=0)

                            # The model analyzes the pixels and returns a list of 1000 mathematical probabilities
                            predictions = mobilenet_model.predict(img_array, verbose=0)
                            
                            # Index of the highest probability converted into human-readable labels
                            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

                            # Result storage
                            label = decoded[0][1]       # Object name (String)
                            confidence = decoded[0][2]  # Accuracy score (0.0 to 1.0)

                            # Text displayed
                            result_text = f"{label} ({confidence*100:.1f}%)"

                            cv2.putText(frame, result_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # System status message
                cv2.putText(frame, "SYSTEM ACTIVE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
            else:

                # Instruction shown while waiting for the activation gesture
                cv2.putText(frame, "Show open hand to activate", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
            # ===== [ EXHIBITION ] ===============================================================================================
            # Final window shown
            cv2.imshow("Integrated AI System", frame)

            # ===== [ EXIT CONDITION ] ===========================================================================================
            # Waits a millisecond for a key
            # If 'q' is pressed, the program closes
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ===== [ CLEANUP ] =========================================================================================================
    # Releases the camera and closes all open windows
    cap.release()
    cv2.destroyAllWindows()

# ===== [ START ] ================================================================================================================
# Entry point of the program
if __name__ == "__main__":
    main()