import cv2
from ultralytics import YOLO

# ===== [ YOLO MODEL ]
# YOLOv8 nano model for real-time object detection
yolo_model = YOLO('yolov8n.pt')

# Removal of the "person" class to focus detection on other objects
targeted_classes = [id for id, name in yolo_model.names.items() if name != 'person']

# ===== [ VIDEO CAPTURE ] =================================================================================================
# Connection to the camera's hardware
cap = cv2.VideoCapture(0)

# Window dimension
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Creation of a resizable window with maintained aspect ratio
win_name = "YOLOv8 Real-Time Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

while True:

    # cam.read() returns two values:
    # 'net': boolean that indicates if the frame was successfully read
    # 'frame': image matrix
    ret, frame = cap.read()

    if not ret:
        break
    
    # Horizontal flip for a more natural mirror effect
    frame = cv2.flip(frame, 1)

    # ===== [ YOLO INFERENCE ]
    # YOLO object detection
    results_yolo = yolo_model(frame, stream=True, classes=targeted_classes)
    
    # Iteration through YOLO results
    for result_yolo in results_yolo:

        # Frame with bounding boxes and labels already drawn by YOLO
        annotated_frame = result_yolo.plot()
    
    # ===== [ EXHIBITION ] ===============================================================================================
    # Final window shown
    cv2.imshow(win_name, annotated_frame)

    # ===== [ EXIT CONDITION ] ===========================================================================================
    # Waits a millisecond for a key
    # If 'q' is pressed, the program closes
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== [ CLEANUP ] =========================================================================================================
# Releases the camera and closes all open windows
cap.release()
cv2.destroyAllWindows()