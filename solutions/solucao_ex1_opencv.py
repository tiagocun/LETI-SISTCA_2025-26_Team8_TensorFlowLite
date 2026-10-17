import cv2

# ===== [ VIDEO CAPTURE ] =================================================================================================
# Connection to the camera's hardware
cap = cv2.VideoCapture(0)

while True:

    # cap.read() returns two values:
    # 'ret': boolean that indicates if the frame was successfully read
    # 'frame': image matrix
    ret, frame = cap.read()

    if not ret:
        break

    # Horizontal flip for a more natural mirror effect
    frame = cv2.flip(frame, 1)

    # ===== [ PRE-PROCESSING ] ============================================================================================
    # Conversion of the image from BGR (color) to grayscale
    # Used to reduce computacional cost in processing tasks (3 color channels -> 1 color channel)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===== [ ROI DEFINITION ] ==================================================================
    # Acquisiton of the frame dimensions
    height, width = frame.shape[:2]

    # Definition of central region of interest (ROI)
    x1, y1 = width // 4, height // 4            # Upper left corner
    x2, y2 = 3 * width // 4, 3 * height // 4    # Bottom right corner

    # ===== [ VISUAL ELEMENTS ] ===========================================================================================
    # Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Instruction for the user (above the rectangle)
    cv2.putText(frame, "Put your hand/object here", (x2, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ===== [ EXHIBITION ] ================================================================================================
    # Window with guide
    cv2.imshow("Feed", frame)
    
    # Secondary window of what the system sees internally
    cv2.imshow("Processing", gray_frame)

    # ===== [ EXIT CONDITION ] ===========================================================================================
    # Waits a millisecond for a key
    # If 'q' is pressed, the program closes
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== [ CLEANUP ] ======================================================================================================
# Releases the camera and closes all open windows
cap.release()
cv2.destroyAllWindows()