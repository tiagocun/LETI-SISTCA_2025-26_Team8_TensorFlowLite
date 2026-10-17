import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Convert the 'frame' into a grayscale
    # HINT: Use the 'cv2.cvtColor' function
    gray_frame = # TODO

    # 2. Define the rectangle on the center of the screen
    # HINT: Use 'cv2.rectangle(frame, point1, point2, color, thickness)
    height, width = frame.shape[:2]
    x1, y1 = # TODO
    x2, y2 = # TODO

    cv2.rectangle( ... )

    cv2.putText(frame, "Put your hand/object here", (x2, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Feed", frame)
    cv2.imshow("Processing", gray_frame)

    # 3. Complete the exit condition
    # TODO: If 'q' is pressed: break
    if cv2.waitKey(1) & # TODO:
        break

cap.release()
cv2.destroyAllWindows()