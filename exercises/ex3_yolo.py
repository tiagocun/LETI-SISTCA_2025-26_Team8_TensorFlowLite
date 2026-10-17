import cv2
from ultralytics import YOLO

# 1. Load YOLOv8 model
yolo_model = # TODO

targeted_classes = [id for id, name in yolo_model.names.items() if name != 'person']

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

win_name = "YOLOv8 Real-Time Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 2. Execute the detection on the current frame
    # HINT: Use "classes=" as a parameter as well
    results_yolo = # TODO

    # 3. Iterate through the results and generate the annotated frame
    # HINT: Use the ".plot()" method
    for result_yolo in results_yolo:
        # TODO
    
    cv2.imshow(win_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows