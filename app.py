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


def format_prediction(result, proba):
    """
    Handles both model output formats:
    - numeric labels: 0 / 1
    - text labels: Introvert / Extrovert
    """

    if str(result).lower() == "introvert" or result == 1:
        label = "Introvert"
    else:
        label = "Extrovert"

    classes = list(model.classes_)

    probabilities = {}
    for i, class_name in enumerate(classes):
        if str(class_name).lower() == "introvert" or class_name == 1:
            probabilities["Introvert"] = round(float(proba[i]), 4)
        else:
            probabilities["Extrovert"] = round(float(proba[i]), 4)

    confidence = probabilities.get(label, max(probabilities.values()))

    return label, confidence, probabilities


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            data = {
                "Time_spent_Alone": float(request.form["Time_spent_Alone"]),
                "Stage_fear": request.form["Stage_fear"],
                "Social_event_attendance": float(request.form["Social_event_attendance"]),
                "Going_outside": float(request.form["Going_outside"]),
                "Drained_after_socializing": request.form["Drained_after_socializing"],
                "Friends_circle_size": float(request.form["Friends_circle_size"]),
                "Post_frequency": float(request.form["Post_frequency"]),
            }

            input_df = pd.DataFrame([data], columns=REQUIRED_FIELDS)

            result = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            label, confidence, probabilities = format_prediction(result, proba)

            prediction = f"{label} ({round(confidence * 100, 2)}% confidence)"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)

        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        input_df = pd.DataFrame([{field: data[field] for field in REQUIRED_FIELDS}], columns=REQUIRED_FIELDS)

        result = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label, confidence, probabilities = format_prediction(result, proba)

        return jsonify({
            "personality": label,
            "confidence": round(float(confidence), 4),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["GET"])
def predict_get():
    return jsonify({
        "message": "This endpoint accepts POST requests only.",
        "example_url": "/predict",
        "example_method": "POST",
        "example_body": {
            "Time_spent_Alone": 8,
            "Stage_fear": "Yes",
            "Social_event_attendance": 2,
            "Going_outside": 1,
            "Drained_after_socializing": "Yes",
            "Friends_circle_size": 3,
            "Post_frequency": 1
        }
    }), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)