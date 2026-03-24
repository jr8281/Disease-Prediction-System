# 🩺 Disease Prediction System

A Streamlit web app that predicts diseases based on symptoms using a **Random Forest** classifier.

---

## 📁 Project Structure

```
disease_predictor/
├── app.py                        # Main Streamlit UI
├── utils.py                      # ML model, data loading, helpers
├── requirements.txt              # Python dependencies
├── model.pkl                     # Auto-generated on first run
└── data/
    ├── dataset.csv               # Training data (symptoms → disease)
    ├── Symptom-severity.csv      # Symptom severity weights
    ├── symptom_Description.csv   # Disease descriptions
    └── symptom_precaution.csv    # Disease precautions
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

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

## 📊 Model Performance

| Metric        | Value     |
|---------------|-----------|
| Algorithm     | Random Forest (200 trees) |
| Training size | ~3,936 samples |
| Test size     | ~984 samples |
| Diseases      | 41 |
| Symptoms      | 131 |
| Typical accuracy | ~98%+ |

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. Do not use it as a substitute for professional medical advice, diagnosis, or treatment.
