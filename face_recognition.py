"""
face_recognition.py
Contains FaceRecognizer class:
- Loads CNN model
- Detects faces using Haar cascade
- Runs prediction and returns annotated frames
"""
import json
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class FaceRecognizer:
    def __init__(self, model_path="models/face_cnn_model.h5", label_map_path=None,
                 cascade_path=None, input_size=(100, 100), threshold=0.5):
        """
        Args:
            model_path: Path to saved Keras .h5 model
            label_map_path: Optional JSON mapping of class indices to names
            cascade_path: Optional path to OpenCV Haar cascade XML (else use default)
            input_size: model input size (width, height)
            threshold: probability threshold for deciding label
        """
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.input_size = input_size
        self.threshold = threshold

        # Load face detector (Haar cascade)
        if cascade_path:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = load_model(self.model_path)

        # Load or build label map
        self.label_map = self._load_label_map(label_map_path)

    def _load_label_map(self, path):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # default: binary classifier -> 0: Unknown, 1: Known
        return {"0": "Unknown", "1": "Known"}

    def preprocess_face(self, face_img):
        """
        Resize, normalize and reshape face image for model prediction.
        Expects color or grayscale image (BGR).
        """
        # convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.input_size)
        arr = resized.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)           # (H, W, 1)
        arr = np.expand_dims(arr, axis=0)            # (1, H, W, 1)
        return arr

    def predict(self, face_img):
        """Return (label_str, probability)"""
        x = self.preprocess_face(face_img)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        label_idx = "1" if prob >= self.threshold else "0"
        label = self.label_map.get(label_idx, label_idx)
        return label, prob

    def annotate_frame(self, frame):
        """
        Detect faces in frame, run prediction on each, and draw boxes/labels.
        Returns the annotated BGR frame.
        """
        # Work on a copy to avoid modifying original
        out = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                   minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                label, prob = self.predict(face_roi)
                text = f"{label} ({prob:.2f})"
            except Exception:
                label, prob = "Error", 0.0
                text = "Error"

            # Draw rectangle and put text
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(out, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        return out
