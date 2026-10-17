# SISTCA Lab — Computer Vision with TensorFlow

Exercises and final challenge for the SISTCA lab classes.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

| Path | Description |
|------|-------------|
| `exercises/` | Exercise files with TODOs for students to complete |
| `solutions/` | Complete reference solutions |
| `challenge/` | Final challenge — sign language recognition |

## Exercises

- **Ex1** — OpenCV: video capture, grayscale conversion, ROI
- **Ex2** — TensorFlow: real-time classification with MobileNetV2
- **Ex3** — YOLOv8: real-time object detection
- **Ex4** — MediaPipe: hand landmark detection
- **Final challenge** — build a sign language letter recogniser from scratch

## Running

```
# Individual exercises
python exercises/ex1_opencv.py

# Final challenge
python challenge/final_challenge.py
```

The MediaPipe and YOLO models are downloaded automatically on first run.
