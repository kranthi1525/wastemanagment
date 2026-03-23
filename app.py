from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model("model.h5")

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Class labels
class_names = [
    "paper waste",
    "glass waste",
    "organic waste",
    "plastic waste",
    "automobile wastes",
    "metal waste",
    "battery waste",
    "E-waste",
    "light bulbs"
]

@app.route("/")
def home():
    with open("index.html", "r") as f:
        return f.read()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        files = request.files.getlist("files")

        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        images = []
        filenames = []

        for file in files:
            img = Image.open(file).convert("RGB")
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img = np.array(img, dtype=np.float32) / 255.0

            images.append(img)
            filenames.append(file.filename)

        images = np.array(images)

        # Prediction (NO training, safe)
        predictions = model.predict(images, verbose=0)

        results = []

        for i, pred in enumerate(predictions):
            class_index = int(np.argmax(pred))
            confidence = float(np.max(pred))

            results.append({
                "file": filenames[i],
                "class": class_names[class_index],
                "confidence": round(confidence * 100, 2)
            })

        return jsonify({"results": results})

    except Exception as e:
        print("ERROR:", str(e))   # Debug print
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)