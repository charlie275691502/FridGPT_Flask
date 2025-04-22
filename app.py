from flask import Flask, request, jsonify
from PIL import Image
import torch
from yolov5 import YOLOv5  # if you're using yolov5 module (not ultralytics)

app = Flask(__name__)

# Load model (once)
model = YOLOv5("best.pt", device="cpu")  # or 'cuda'

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = Image.open(file.stream)

    results = model.predict(image)
    json_output = results[0].json()

    filtered = [{"name": item["name"], "confidence": item["confidence"]} for item in json_output]
    return jsonify(filtered)

@app.route("/", methods=["GET"])
def home():
    return "YOLO model is up and running!"
