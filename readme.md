# Real-Time Face Recognition (OpenCV + CNN + Flask)

## Overview
A simple real-time face detection and recognition system using:
- OpenCV for video capture and face detection (Haar cascade)
- TensorFlow/Keras CNN for face classification
- Flask to serve a live annotated video feed in the browser

## Files
- `app.py` — Flask app and video stream
- `face_recognition.py` — FaceRecognizer class that loads model & annotates frames
- `model_training.py` — Training script to build a CNN model from `dataset/`
- `requirements.txt` — Python dependencies
- `templates/index.html` — Web UI
- `static/style.css` — Basic CSS
- `models/face_cnn_model.h5` — Trained model (not included)
- `models/label_map.json` — Label map saved during training

## Quick Start

1. **Clone / Create project folder**
   ```bash
   git clone <your-repo>    # or copy files into folder
   cd real-time-face-recognition
"# real-time-face-recognition" 
