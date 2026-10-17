import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.applications.MobileNetV2(weights='imagenet')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    x1, y1 = width // 4, height // 4
    x2, y2 = 3 * width // 4, 3 * height // 4

    # 1. Extract the Region of Interest (ROI) based on the calculated points
    # HINT: ROI is calculated using [x2:y2, x1:y1]
    roi = # TODO

    # 2. Prepare the image
    # HINT: use cv2.resize() to resize the image to 224x224
    img_resized = # TODO
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

    # 3. Extract the label and confidence of the top of the list
    label = # TODO
    confidence = # TODO

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    result_text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, result_text, (x1, y1 -10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.imshow("Integrated System (OpenCV + TensorFlow)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()