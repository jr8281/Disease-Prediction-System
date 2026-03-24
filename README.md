# 🩺Med Predict - AI Disease Prediction System

A Streamlit web app that predicts diseases based on symptoms using a **Random Forest** classifier.


## 🤖 How It Works

1. **Feature Engineering** — Each symptom is encoded using its severity weight (from `Symptom-severity.csv`) instead of plain binary encoding. This gives the model richer signal.

2. **Model** — `RandomForestClassifier` with 200 trees trained on 4,920 samples across 41 diseases.

3. **Inference** — Selected symptoms → severity-weighted feature vector → model predicts top-3 diseases with confidence %.

4. **Post-prediction** — The app displays:
   - Disease description
   - 4 recommended precautions
   - Severity score (Mild / Moderate / Severe)
   - Symptom breakdown chart

---

##  Model Performance

| Metric        | Value     |
|---------------|-----------|
| Algorithm     | Random Forest (200 trees) |
| Training size | ~3,936 samples |
| Test size     | ~984 samples |
| Diseases      | 41 |
| Symptoms      | 131 |
| Typical accuracy | ~98%+ |

