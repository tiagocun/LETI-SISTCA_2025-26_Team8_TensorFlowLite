import cv2
import numpy as np
import tensorflow as tf

# ===== [ MOBILENETV2 MODEL ] =========================================================================================================
# MobileNetV2 model pre-trained with ImageNet dataset
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# ===== [ VIDEO CAPTURE ] =================================================================================================
# Connection to the camera's hardware
cap = cv2.VideoCapture(0)

# Window dimension
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

    # cam.read() returns two values:
    # 'ret': boolean that indicates if the frame was successfully read
    # 'frame': image matrix
    ret, frame = cap.read()

    if not ret:
        break
    
    # Horizontal flip for a more natural mirror effect
    frame = cv2.flip(frame, 1)

    # ===== [ DEFINITION OF THE AREA OF INTEREST (ROI) ] ==================================================================
    # Acquisiton of the frame dimensions
    height, width = frame.shape[:2]
    
    # Definition of the rectangle's points in order for it to occupy the central part of the feed
    x1, y1 = width // 4, height // 4
    x2, y2 = 3 * width // 4, 3 * height // 4

    # ===== [ AI WORK ] ===================================================================================================
    # Image slicing to send only the rectangle to the AI
    roi = frame[y1:y2, x1:x2]

    # Resize to MobileNetV2 input size
    img_resized = cv2.resize(roi, (224, 224))
    
    # Pixel values (0-255) scaled to the range the model uses (-1 to 1)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    
    # TensorFlow expects a list of images
    # Dimension added to turn one image (224, 224, 3) into a list of one image (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # The model analyzes the pixels and returns a list of 1000 mathematical probabilities
    predictions = model.predict(img_array, verbose=0)
    
    # Index of the highest probability converted into a human-readable name
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    
    # Result storage
    label = decoded[0][1]       # Object name (String)
    confidence = decoded[0][2]  # Accuracy score (0.0 to 1.0)

    # ===== [ VISUAL ELEMENTS ] ===========================================================================================
    # Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Text displayed
    result_text = f"{label} ({confidence*100:.1f}%)"

    cv2.putText(frame, result_text, (x1, y1 -10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ===== [ EXHIBITION ] ================================================================================================
    cv2.imshow("Integrated System (OpenCV + TensorFlow)", frame)

    # ===== [ EXIT CONDITION ] ============================================================================================
    # Waits a millisecond for a key
    # If 'q' is pressed, the program closes
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== [ CLEANUP ] ======================================================================================================
# Releases the camera and closes all open windows
cap.release()
cv2.destroyAllWindows()