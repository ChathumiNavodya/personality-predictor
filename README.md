# Personality Predictor API

A machine learning web application that predicts whether a person is an **Introvert** or an **Extrovert** based on social behavior and lifestyle characteristics.

---

# Live Demo

### Web Application
https://web-production-346d6.up.railway.app

### API Endpoint
https://web-production-346d6.up.railway.app/predict

---

# Project Overview

This project was developed as part of the **AI/ML Intern Round 02 Technical Assignment**.

The application:
- Trains a machine learning classification model
- Predicts personality type
- Provides a REST API
- Includes a responsive web interface
- Is deployed publicly using Railway

---

# Features

- Machine Learning personality prediction
- Introvert vs Extrovert classification
- Confidence score prediction
- REST API with JSON support
- Responsive mobile-friendly UI
- Public cloud deployment
- Flask backend
- API testing using Postman

---

# Machine Learning Model

| Item | Details |
|------|---------|
| Algorithm | Gradient Boosting Classifier |
| Library | Scikit-learn |
| Accuracy | 91%+ |
| Target Classes | Introvert / Extrovert |

---

# Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML/CSS
- Railway
- Gunicorn

---

# Project Structure

```text
personality-predictor/
│
├── app.py
├── personality_model.pkl
├── personality_prediction.ipynb
├── requirements.txt
├── Procfile
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
```

---

# Files Description

| File | Description |
|------|-------------|
| `app.py` | Main Flask application |
| `personality_model.pkl` | Trained ML model |
| `personality_prediction.ipynb` | Data preprocessing + model training notebook |
| `requirements.txt` | Required Python packages |
| `Procfile` | Deployment startup command |
| `templates/index.html` | Frontend UI |

---

# Running Locally

## 1. Clone Repository

```bash
git clone https://github.com/ChathumiNavodya/personality-predictor.git
```

---

## 2. Navigate to Project Folder

```bash
cd personality-predictor
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Application

```bash
python app.py
```

---

## 5. Open Browser

```text
http://127.0.0.1:5000
```

---

# API Documentation

## POST `/predict`

Predicts whether a person is Introvert or Extrovert.

---

# Request Format

```json
{
  "Time_spent_Alone": 8,
  "Stage_fear": "Yes",
  "Social_event_attendance": 2,
  "Going_outside": 1,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 3,
  "Post_frequency": 1
}
```

---

# Response Format

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

---

# Input Field Details

| Field | Type | Description |
|------|------|-------------|
| Time_spent_Alone | Number | Hours spent alone per day |
| Stage_fear | String | Yes or No |
| Social_event_attendance | Number | Social events attended weekly |
| Going_outside | Number | Times going outside weekly |
| Drained_after_socializing | String | Yes or No |
| Friends_circle_size | Number | Number of close friends |
| Post_frequency | Number | Social media posts per week |

---

# API Testing with Postman

## URL

```text
https://web-production-346d6.up.railway.app/predict
```

## Method

```text
POST
```

## Body Type

```text
raw → JSON
```

---

# Deployment

This project is deployed publicly using:

- Railway (Cloud Platform)
- Gunicorn (WSGI Server)

---

# Model Training Notebook

The Jupyter notebook used for:
- data preprocessing
- feature engineering
- model training
- evaluation

is included in:

```text
personality_prediction.ipynb
```

---

# Author

### Chathumi Navodya Sathsarani

AI/ML Intern Candidate

GitHub:
https://github.com/ChathumiNavodya

---

# Integrity Declaration

This project was developed independently as part of the AI/ML Intern technical assignment submission.
