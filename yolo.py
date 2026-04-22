import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível aceder à webcam.")
    exit()


while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erro: Falha na captura de imagem da webcam.")
        break

    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()

    cv2.imshow('Detecao YOLO em Tempo Real', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()