# Release v1.0.0 — Real-Time Face Recognition

**Release date:** 28-Nov-2025

## Summary
Initial public release of the Real-Time Face Recognition project.  
Features:
- Real-time face detection (OpenCV Haar cascade)
- CNN-based face classification using TensorFlow/Keras
- Flask-based web UI to stream annotated video
- Training script to build your own model from `dataset/`
- Sample small model generation script (for testing)

## Files included
- `app.py` — Flask server + webcam streaming
- `face_recognition.py` — FaceRecognizer class (loads model & annotates frames)
- `model_training.py` — Training script
- `requirements.txt` — Required Python packages
- `templates/index.html`, `static/style.css` — UI
- (Optional) `models/face_cnn_model.h5` — Sample/trained model (attached as release asset)
- `models/label_map.json` — label map for model

## How to run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
