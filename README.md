# Personality Predictor API

Binary classifier (Introvert vs Extrovert) served as a REST API.

## Model
- Algorithm: **Gradient Boosting** (sklearn, 200 trees)
- CV Accuracy (5-fold): **93.8 %**
- Test Accuracy: **91.7 %** | ROC-AUC: **95.6 %**

## Files
| File | Description |
|------|-------------|
| `personality_prediction.ipynb` | Full EDA + training notebook |
| `app.py` | Flask API |
| `personality_model.pkl` | Serialised sklearn Pipeline |
| `requirements.txt` | Python dependencies |
| `Procfile` | Gunicorn startup command (Render/Railway) |

---

## Deploy to Render (free tier)

1. Push this folder to a **public GitHub repo**.
2. Go to [render.com](https://render.com) → New → **Web Service**.
3. Connect your GitHub repo.
4. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Click **Deploy**. Render will give you a URL like  
   `https://personality-predictor-xxxx.onrender.com`

---

## API Reference

### `GET /`
Returns API description and input schema.

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /predict`

**Request**
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time_spent_Alone": 8,
    "Stage_fear": "Yes",
    "Social_event_attendance": 2,
    "Going_outside": 1,
    "Drained_after_socializing": "Yes",
    "Friends_circle_size": 3,
    "Post_frequency": 1
  }'
```

**Response**
```json
{
  "personality": "Introvert",
  "confidence": 0.9327,
  "probabilities": {
    "Extrovert": 0.0673,
    "Introvert": 0.9327
  }
}
```

**Input field reference**

| Field | Type | Range |
|-------|------|-------|
| `Time_spent_Alone` | float | 0–11 (hours/day) |
| `Stage_fear` | string | `"Yes"` or `"No"` |
| `Social_event_attendance` | float | 0–10 (events/week) |
| `Going_outside` | float | 0–7 (times/week) |
| `Drained_after_socializing` | string | `"Yes"` or `"No"` |
| `Friends_circle_size` | float | 0–20 |
| `Post_frequency` | float | 0–10 (posts/week) |

All fields are optional — missing values are imputed using training-set statistics.
