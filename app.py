from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# load model
model = load_model("final_model.h5")

img_size = (128, 128)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]

        # folder upload
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)

        # kasih nama unik biar ga ketimpa
        filename = str(int(time.time())) + "_" + secure_filename(file.filename)

        filepath = os.path.join(upload_folder, filename)

        # simpan file
        file.save(filepath)

        # preprocessing
        img = load_img(filepath, target_size=img_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        # prediksi
        pred = model.predict(img)[0][0]

        if pred > 0.5:
            label = "Dog"
            confidence = float(pred)
        else:
            label = "Cat"
            confidence = float(1 - pred)

        return jsonify({
            "status": 200,
            "success": True,
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "image_url": "/" + filepath
        })

    except Exception as e:
        return jsonify({
            "status": 500,
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)