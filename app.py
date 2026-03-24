import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    get_or_train_model,
    predict_disease,
    load_descriptions,
    load_precautions,
    load_symptoms,
    severity_score,
    severity_label,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        color: white;
        margin-bottom: 1rem;
    }
    .result-disease {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .result-confidence {
        font-size: 1rem;
        opacity: 0.85;
    }
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .precaution-item {
        background: #fff3cd;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-left: 4px solid #ffc107;
    }
    .severity-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Load model & data (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model... this only happens once ⚙️")
def load_resources():
    model, all_symptoms, sev_dict, acc = get_or_train_model()
    descriptions = load_descriptions()
    precautions = load_precautions()
    sev_df = load_symptoms()
    return model, all_symptoms, sev_dict, acc, descriptions, precautions, sev_df


model, all_symptoms, sev_dict, acc, descriptions, precautions, sev_df = load_resources()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/stethoscope.png", width=80)
    st.markdown("##  Disease Predictor")
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "Select your symptoms from the panel on the right. "
        "The AI model will predict the most likely disease and provide "
        "a description and precautions."
    )
    st.markdown("### Model Info")
    st.success(f" Random Forest Algorithm\n\n Test Accuracy: **{acc*100:.1f}%**\n\n Diseases: **41**\n\n Symptoms: **{len(all_symptoms)}**")
    st.markdown("---")
   

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title"> Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Select your symptoms below to get an AI-powered disease prediction</p>', unsafe_allow_html=True)

st.markdown("---")

# Symptom selection
st.markdown("### 🔍 Select Your Symptoms")
st.markdown("Type to search and select all symptoms you are experiencing:")

# Format symptoms for display (replace underscores, title case)
symptom_display = {s: s.replace("_", " ").title() for s in all_symptoms}
display_to_raw = {v: k for k, v in symptom_display.items()}
display_options = sorted(symptom_display.values())

selected_display = st.multiselect(
    label="Symptoms",
    options=display_options,
    placeholder="Search and select symptoms...",
    label_visibility="collapsed",
)

selected_symptoms = [display_to_raw[d] for d in selected_display]

# ─── Predict button ────────────────────────────────────────────────────────────
col_btn, col_clear = st.columns([2, 8])
with col_btn:
    predict_btn = st.button("🔮 Predict Disease", type="primary", use_container_width=True)

st.markdown("---")

# ─── Results ──────────────────────────────────────────────────────────────────
if predict_btn:
    if len(selected_symptoms) < 2:
        st.warning("⚠️ Please select at least **2 symptoms** for a meaningful prediction.")
    else:
        predictions = predict_disease(selected_symptoms, model, all_symptoms, sev_dict, top_n=3)
        score = severity_score(selected_symptoms, sev_dict)
        sev_text, sev_color = severity_label(score)
        top_disease, top_conf = predictions[0]

        # ── Top metrics row ───────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("🦠 Predicted Disease", top_disease)
        with m2:
            st.metric("📊 Confidence", f"{top_conf}%")
        with m3:
            st.metric("⚡ Symptom Severity Score", f"{score}")

        st.markdown("---")

        # ── Main result + severity ─────────────────────────────────────────────
        res_col, sev_col = st.columns([3, 1])
        with res_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-disease">🦠 {top_disease}</div>
                    <div class="result-confidence">Confidence: {top_conf}% &nbsp;|&nbsp; Based on {len(selected_symptoms)} symptom(s)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sev_col:
            st.markdown(
                f"""
                <div style="background:{sev_color};border-radius:12px;padding:1rem 1.2rem;color:white;text-align:center;height:100%">
                    <div style="font-size:2rem">{'🟢' if sev_text=='Mild' else '🟡' if sev_text=='Moderate' else '🔴'}</div>
                    <div style="font-size:1.3rem;font-weight:700">{sev_text}</div>
                    <div style="font-size:0.85rem;opacity:0.9">Severity Level</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Description + Precautions ──────────────────────────────────────────
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("#### 📋 Disease Description")
            desc = descriptions.get(top_disease, "Description not available.")
            st.markdown(
                f'<div class="info-card">{desc}</div>',
                unsafe_allow_html=True,
            )

        with right_col:
            st.markdown("#### 🛡️ Recommended Precautions")
            prec_list = precautions.get(top_disease, [])
            if prec_list:
                for i, p in enumerate(prec_list, 1):
                    st.markdown(
                        f'<div class="precaution-item">✅ <b>Step {i}:</b> {p.capitalize()}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No precautions found for this disease.")

        st.markdown("---")

        # ── Other possible diseases ────────────────────────────────────────────
        if len(predictions) > 1:
            st.markdown("#### 🔎 Other Possible Conditions")
            alt_cols = st.columns(len(predictions) - 1)
            for idx, (disease, conf) in enumerate(predictions[1:]):
                with alt_cols[idx]:
                    d_desc = descriptions.get(disease, "")
                    short_desc = d_desc[:120] + "..." if len(d_desc) > 120 else d_desc
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size:1.1rem;font-weight:600;color:#333">{disease}</div>
                            <div style="font-size:1.5rem;font-weight:700;color:#1a73e8">{conf}%</div>
                            <div style="font-size:0.8rem;color:#777;margin-top:0.4rem">{short_desc}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Symptom severity breakdown ─────────────────────────────────────────
        with st.expander("📊 Symptom Severity Breakdown"):
            sev_data = [(s.replace("_", " ").title(), sev_dict.get(s, 1)) for s in selected_symptoms]
            sev_data_sorted = sorted(sev_data, key=lambda x: x[1], reverse=True)
            sev_col1, sev_col2 = st.columns(2)
            for i, (sym, weight) in enumerate(sev_data_sorted):
                col = sev_col1 if i % 2 == 0 else sev_col2
                bar = "█" * weight + "░" * (7 - weight)
                col.markdown(f"**{sym}** &nbsp; `{bar}` &nbsp; weight: **{weight}**")

elif not predict_btn:
    # ── Placeholder / instruction state ────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:3rem;color:#aaa;">
            <div style="font-size:4rem">🔬</div>
            <div style="font-size:1.2rem;margin-top:1rem">Select your symptoms above and click <b>Predict Disease</b></div>
            <div style="font-size:0.9rem;margin-top:0.5rem">The model will predict the most likely disease with confidence scores</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

