from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "personality_model.pkl")
model = joblib.load(MODEL_PATH)

REQUIRED_FIELDS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {
            "Time_spent_Alone": float(request.form["Time_spent_Alone"]),
            "Stage_fear": request.form["Stage_fear"],
            "Social_event_attendance": float(request.form["Social_event_attendance"]),
            "Going_outside": float(request.form["Going_outside"]),
            "Drained_after_socializing": request.form["Drained_after_socializing"],
            "Friends_circle_size": float(request.form["Friends_circle_size"]),
            "Post_frequency": float(request.form["Post_frequency"]),
        }

        input_df = pd.DataFrame([data])
        result = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label = "Introvert" if result == 1 else "Extrovert"
        confidence = round(float(proba[result]) * 100, 2)

        prediction = f"{label} ({confidence}% confidence)"

    return render_template("index.html", prediction=prediction)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    input_df = pd.DataFrame([{f: data[f] for f in REQUIRED_FIELDS}])
    result = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    label = "Introvert" if result == 1 else "Extrovert"

    return jsonify({
        "personality": label,
        "confidence": round(float(proba[result]), 4),
        "probabilities": {
            "Extrovert": round(float(proba[0]), 4),
            "Introvert": round(float(proba[1]), 4),
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)