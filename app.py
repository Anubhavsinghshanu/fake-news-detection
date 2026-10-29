"""Streamlit interface for the Fake News Detection project."""
from __future__ import annotations
import streamlit as st
from src.predict import predict_text, SUPPORTED_MODELS, ModelNotFoundError
from config.settings import ARTIFACT_DIR

st.set_page_config(page_title="Fake News Detection | NLP", page_icon="📰", layout="wide")
st.title("Fake News Detection")
st.caption("NLP-based text classification using TF-IDF and classical machine learning.")
st.warning("A model prediction is not a fact-check. Verify claims against trustworthy primary sources.")

with st.sidebar:
    st.header("Classifier")
    model_name = st.selectbox("Choose trained model", SUPPORTED_MODELS, index=2,
                              format_func=lambda name: {"logistic_regression": "Logistic Regression",
                                                        "multinomial_nb": "Multinomial Naive Bayes",
                                                        "linear_svc": "Linear SVM"}[name])
    st.markdown("**Label mapping**\n- Fake = 1\n- Real = 0")
    st.caption("Train models first from the project root. See README.md.")

article = st.text_area("Paste a news article", height=260,
                       placeholder="Paste the article headline and body here (at least 20 characters).")
if st.button("Analyze article", type="primary", use_container_width=False):
    if not article.strip():
        st.error("Enter article text first.")
    else:
        try:
            result = predict_text(article, model_name)
            if result["label"] == 1:
                st.error("Prediction: FAKE")
            else:
                st.success("Prediction: REAL")
            col1, col2, col3 = st.columns(3)
            col1.metric("Selected model", result["model"].replace("_", " ").title())
            if "scores" in result:
                col2.metric("Real score", f"{result['scores']['Real']:.1%}")
                col3.metric("Fake score", f"{result['scores']['Fake']:.1%}")
            else:
                col2.metric("Decision score", result.get("decision_score", "N/A"))
                col3.metric("Score type", "SVM margin")
            st.info(result["note"])
        except (ModelNotFoundError, ValueError, OSError) as exc:
            st.error(str(exc))

metrics_path = ARTIFACT_DIR / "model_metrics.csv"
chart_path = ARTIFACT_DIR / "model_comparison.png"
if metrics_path.is_file():
    st.subheader("Model evaluation")
    import pandas as pd
    st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
    if chart_path.is_file():
        st.image(str(chart_path), caption="Evaluation scores on the held-out test split", use_container_width=True)
with st.expander("How to interpret this tool"):
    st.write("The model learns statistical patterns in the training dataset. Dataset bias, outdated examples, satire, short text, and distribution changes can cause errors. Do not treat the output as a definitive truth verdict.")
