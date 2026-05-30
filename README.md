# 🩺 MedPredict — AI Disease Prediction System

> **Disclaimer:** This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

## 🔗 Live Demo
👉 **[Try it on Streamlit Cloud](https://your-app-link.streamlit.app)** ← replace with your link after deploying

---

## What it does

MedPredict takes a set of symptoms as input and predicts the most likely disease using a trained Random Forest classifier. It returns the top 3 possible conditions with confidence scores, a severity assessment, disease description, and recommended precautions.

---

## How it works

1. **Feature engineering** — each symptom is encoded using its severity weight from `Symptom-severity.csv` rather than plain binary encoding, giving the model richer signal
2. **Model** — `RandomForestClassifier` with 200 trees, trained on 4,920 samples across 41 diseases
3. **Inference** — selected symptoms → severity-weighted feature vector → top-3 predictions with confidence percentages
4. **Post-prediction** — disease description, precautions, severity score, and per-symptom breakdown

---

## Model performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest (200 trees) |
| Training samples | ~3,936 |
| Test samples | ~984 |
| Diseases covered | 41 |
| Symptoms | 131 |
| Test accuracy | ~98% |

> Note: high accuracy reflects the structured nature of the dataset. Real-world performance on noisy or overlapping symptoms will vary.

---

## Project structure

```
Disease-Prediction-System/
├── app.py                    # Streamlit UI
├── utils.py                  # Model training, prediction, data loading
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md
└── data/
    ├── dataset.csv               # Training data
    ├── symptom_Description.csv   # Disease descriptions
    ├── symptom_precaution.csv    # Recommended precautions
    └── Symptom-severity.csv      # Symptom severity weights
```

---

## Run locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Disease-Prediction-System.git
cd Disease-Prediction-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The model trains automatically on first run and is cached as `model.pkl` for subsequent runs.

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `app.py`
4. Click **Deploy**

---

## Known limitations

- Trained on a structured symptom dataset — not representative of real clinical variability
- Does not account for patient history, age, or comorbidities
- Overlapping symptoms between diseases can reduce prediction confidence
- No cross-validation reported — accuracy figure is single train/test split

---

## Developed by

- Jaswanth Babu Reddi
