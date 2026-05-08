"""
Personality Predictor API
Predicts Introvert / Extrovert from 7 behavioural features.
"""
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model once at startup
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


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Personality Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict introvert or extrovert",
            "GET /health": "Health check",
        },
        "input_schema": {
            "Time_spent_Alone": "float (hours per day, 0-11)",
            "Stage_fear": "string ('Yes' | 'No')",
            "Social_event_attendance": "float (events per week, 0-10)",
            "Going_outside": "float (times per week, 0-7)",
            "Drained_after_socializing": "string ('Yes' | 'No')",
            "Friends_circle_size": "float (number of close friends, 0-20)",
            "Post_frequency": "float (social media posts per week, 0-10)",
        },
        "example_request": {
            "Time_spent_Alone": 8,
            "Stage_fear": "Yes",
            "Social_event_attendance": 2,
            "Going_outside": 1,
            "Drained_after_socializing": "Yes",
            "Friends_circle_size": 3,
            "Post_frequency": 1,
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    # Validate required fields
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{f: data[f] for f in REQUIRED_FIELDS}])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label = "Introvert" if prediction == 1 else "Extrovert"
        confidence = float(proba[prediction])

        return jsonify({
            "personality": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Extrovert": round(float(proba[0]), 4),
                "Introvert": round(float(proba[1]), 4),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
