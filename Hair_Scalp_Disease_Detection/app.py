from flask import Flask, render_template, request, url_for, redirect
import os
import uuid
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array

# ----------------------------
# Flask app configuration
# ----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "hair_disease_best_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

# ----------------------------
# Class names and recommendations
# ----------------------------
CLASS_NAMES = [
    "Alopecia Areata",
    "Contact Dermatitis",
    "Folliculitis",
    "Head Lice",
    "Lichen Planus",
    "Male Pattern Baldness",
    "Psoriasis",
    "Seborrhea Dermatitis",
    "Telogen Effluvium",
    "Tinea Capitis",
    "Unknown Images"
]

CONFIDENCE_THRESHOLD = 85.0

DISEASE_RECOMMENDATIONS = {
    "Alopecia Areata": "Consult a dermatologist. Avoid stress and maintain a healthy diet.",
    "Contact Dermatitis": "Identify and avoid allergens. Use mild shampoos. See a dermatologist if irritation persists.",
    "Folliculitis": "Keep scalp clean. Avoid tight hairstyles. Topical antibiotics may help.",
    "Head Lice": "Use medicated lice shampoo and comb thoroughly. Wash bedding and hats.",
    "Lichen Planus": "Consult a dermatologist for topical treatment. Avoid scratching.",
    "Male Pattern Baldness": "Consider minoxidil or finasteride after consulting a doctor. Maintain scalp care.",
    "Psoriasis": "Use medicated shampoos. Moisturize scalp. Consult a dermatologist.",
    "Seborrhea Dermatitis": "Use anti-dandruff shampoo. Maintain scalp hygiene.",
    "Telogen Effluvium": "Reduce stress, maintain proper nutrition. Hair usually regrows naturally.",
    "Tinea Capitis": "Consult a doctor for antifungal treatment. Keep scalp clean.",
    "Unknown Images": "Image unclear. Try uploading a clearer photo."
}

# ----------------------------
# Helper functions
# ----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in {"png", "jpg", "jpeg"}

def preprocess_image(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(path):
    img_array = preprocess_image(path)
    preds = model.predict(img_array, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    conf_pct = float(preds[best_idx]) * 100.0

    if conf_pct < CONFIDENCE_THRESHOLD:
        predicted_label = "Invalid"
        recommendation = "Model not confident. Try uploading a clearer image."
    else:
        predicted_label = CLASS_NAMES[best_idx]
        recommendation = DISEASE_RECOMMENDATIONS.get(predicted_label, "No recommendation available.")

    return predicted_label, round(conf_pct, 2), recommendation

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename.strip() == "":
            return render_template("index.html", message="❌ No file selected.")
        if not allowed_file(file.filename):
            return render_template("index.html", message="❌ Invalid file type. Upload JPG, JPEG, or PNG.")

        ext = file.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(filepath)
            predicted_label, confidence, recommendation = predict_image(filepath)
            image_url = url_for("static", filename=f"uploads/{filename}")

            return render_template(
                "index.html",
                filename=filename,
                image_path=image_url,
                predicted_label=predicted_label,
                confidence=f"{confidence:.2f}%",
                recommendation=recommendation,
            )
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template("index.html", message=f"⚠️ Error: {e}")

    return render_template("index.html", message="📤 Upload a hair/scalp image")


@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
