"""
app.py
Flask web server that serves a live webcam stream and runs face recognition on each frame.
Run: python app.py
"""
import logging
import cv2
from flask import Flask, render_template, Response
from face_recognition import FaceRecognizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
recognizer = FaceRecognizer(model_path="models/face_cnn_model.h5", label_map_path="models/label_map.json")
camera = None

def get_camera(index=0):
    """Create and return a cv2.VideoCapture object. Reuse global if already opened."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(index)
        # set resolution (optional)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def generate_frames():
    cap = get_camera()
    if not cap.isOpened():
        logging.error("Webcam could not be opened.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            logging.warning("Failed to read frame from webcam.")
            break

        # Pass frame to recognizer which returns an annotated BGR frame
        try:
            annotated = recognizer.annotate_frame(frame)
        except Exception as e:
            logging.exception("Error during recognition: %s", e)
            annotated = frame

        # Encode as JPEG
        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    try:
        logging.info("Starting Flask server...")
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        if camera is not None:
            camera.release()
            logging.info("Released webcam.")
