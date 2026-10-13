import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="IDRAK — Banking Intent Classifier",
    page_icon="🏦",
    layout="wide"
)


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("banking_pipeline.joblib")
    categories = joblib.load("categories.joblib")
    return model, categories


try:
    model, categories = load_model_artifacts()
    model_ready = True
except Exception:
    model_ready = False

# Sidebar
with st.sidebar:
    st.title("🏦 IDRAK | إدراك")
    st.caption("NLP-Based Banking Query Router")
    st.divider()
    st.markdown("**Dataset:** Banking77 (PolyAI)")
    st.markdown("**Top-1 Accuracy:** `89.42%`")
    st.markdown("**Top-3 Accuracy:** `>96%`")
    st.markdown("**Latency:** `< 50 ms`")
    st.divider()
    st.info("💡 **Demo Tip:** Use the quick preset buttons to test realistic customer scenarios instantly.")

# Main Header
st.title("⚡ Smart Banking Query Router")
st.markdown("Analyze customer banking messages in real time to predict intent and determine automated routing actions.")

if not model_ready:
    st.error(
        "⚠️ Model artifacts not found. Please ensure 'banking_pipeline.joblib' and 'categories.joblib' are in the project folder.")
    st.stop()

# Quick Presets for Live Presentation
st.markdown("##### 📌 Quick Demo Presets:")
col1, col2, col3, col4 = st.columns(4)

selected_query = None
if col1.button("💳 Stolen Card"):
    selected_query = "My credit card was stolen while traveling, please freeze it immediately!"
if col2.button("⏳ Pending Transfer"):
    selected_query = "Why is my international wire transfer still pending after three days?"
if col3.button("🏧 ATM Cash Error"):
    selected_query = "The ATM did not dispense my cash but the amount was deducted from my account."
if col4.button("🔒 PIN Change"):
    selected_query = "How do I change the security PIN code for my new card?"

# Text Input Field
user_input = st.text_area(
    "Customer Message:",
    value=selected_query if selected_query else "",
    placeholder="Type any banking inquiry here...",
    height=90
)

# Inference Action
if st.button("🚀 Classify & Route Query", type="primary"):
    if not user_input.strip():
        st.warning("Please enter or select a customer query first.")
    else:
        # Predict Probabilities
        probs = model.predict_proba([user_input])[0]
        top_indices = np.argsort(probs)[::-1][:3]

        top1_intent = categories[top_indices[0]]
        top1_score = probs[top_indices[0]]

        st.divider()
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.subheader("🎯 Primary Prediction")
            formatted_intent = top1_intent.replace("_", " ").title()
            st.metric(label="Predicted Intent", value=formatted_intent)

            # Confidence Progress Bar
            st.progress(float(top1_score))
            st.caption(f"Confidence Level: **{top1_score * 100:.1f}%**")

            # Smart Routing Recommendation
            if top1_score >= 0.75:
                st.success("🤖 **Automated Action:** High confidence. Automated self-service workflow triggered.")
            elif top1_score >= 0.40:
                st.warning("⚠️ **Specialist Routing:** Moderate confidence. Routed to domain specialist queue.")
            else:
                st.info("👤 **Human Escalation:** Low confidence. Routed to human support agent for review.")

        with res_col2:
            st.subheader("📊 Top-3 Predictions")
            for idx in top_indices:
                intent_name = categories[idx].replace("_", " ").title()
                score = probs[idx]
                st.markdown(f"**{intent_name}** — `{score * 100:.1f}%`")
                st.progress(float(score))