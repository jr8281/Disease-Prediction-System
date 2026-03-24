import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_symptoms():
    """Load all valid symptoms with their severity weights."""
    sev = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
    sev = sev[sev["Symptom"] != "prognosis"].drop_duplicates("Symptom")
    sev["Symptom"] = sev["Symptom"].str.strip()
    return sev


def load_descriptions():
    """Return dict: disease -> description."""
    df = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
    return dict(zip(df["Disease"].str.strip(), df["Description"]))


def load_precautions():
    """Return dict: disease -> list of precautions."""
    df = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
    result = {}
    for _, row in df.iterrows():
        disease = str(row["Disease"]).strip()
        prec = [
            str(row[c]).strip()
            for c in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
            if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")
        ]
        result[disease] = prec
    return result


def build_feature_vector(selected_symptoms: list, all_symptoms: list, sev_dict: dict):
    """
    Convert a list of selected symptom names into a feature vector.
    Each feature = severity weight of the symptom (0 if not selected).
    """
    vec = np.zeros(len(all_symptoms))
    for i, sym in enumerate(all_symptoms):
        if sym in selected_symptoms:
            vec[i] = sev_dict.get(sym, 1)
    return vec


def load_and_preprocess():
    """
    Load the training dataset and convert symptom columns into a
    severity-weighted feature matrix.
    Returns: X (ndarray), y (array of disease names), all_symptoms (list), sev_dict (dict)
    """
    sev_df = load_symptoms()
    all_symptoms = sev_df["Symptom"].tolist()
    sev_dict = dict(zip(sev_df["Symptom"], sev_df["weight"]))

    df = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    df = df.where(pd.notna(df), "")

    X = []
    for _, row in df.iterrows():
        symptoms_in_row = [
            str(row[f"Symptom_{i}"]).strip()
            for i in range(1, 18)
            if str(row[f"Symptom_{i}"]).strip() != ""
        ]
        vec = build_feature_vector(symptoms_in_row, all_symptoms, sev_dict)
        X.append(vec)

    X = np.array(X)
    y = df["Disease"].str.strip().values
    return X, y, all_symptoms, sev_dict


def train_model():
    """Train Random Forest and return model + metadata."""
    X, y, all_symptoms, sev_dict = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, all_symptoms, sev_dict, acc


def get_or_train_model(model_path="model.pkl"):
    """Load cached model or retrain if not found."""
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["all_symptoms"], data["sev_dict"], data["accuracy"]

    model, all_symptoms, sev_dict, acc = train_model()
    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": model, "all_symptoms": all_symptoms, "sev_dict": sev_dict, "accuracy": acc},
            f,
        )
    return model, all_symptoms, sev_dict, acc


def predict_disease(selected_symptoms, model, all_symptoms, sev_dict, top_n=3):
    """
    Predict disease from selected symptoms.
    Returns list of (disease, probability) sorted by probability desc.
    """
    vec = build_feature_vector(selected_symptoms, all_symptoms, sev_dict)
    vec = vec.reshape(1, -1)
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    top_indices = np.argsort(proba)[::-1][:top_n]
    return [(classes[i], round(float(proba[i]) * 100, 2)) for i in top_indices]


def severity_score(selected_symptoms, sev_dict):
    """Return total severity score for selected symptoms."""
    return sum(sev_dict.get(s, 1) for s in selected_symptoms)


def severity_label(score):
    """Convert numeric severity score to a human-readable label."""
    if score <= 10:
        return "Mild", "#2ecc71"
    elif score <= 20:
        return "Moderate", "#f39c12"
    else:
        return "Severe", "#e74c3c"
