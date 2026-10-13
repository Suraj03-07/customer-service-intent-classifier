from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (
    APP_DIR / "intent_pipeline.joblib",  
    APP_DIR / "intnet_pipeline.joblib",   
)

st.set_page_config(
    page_title="Customer Service Intent Classifier",
    page_icon="💬",
    layout="centered",
)


# -----------------------------------------------------------------------------
# Model loading: lazy-load only after the user requests a prediction, then cache
# the result so the artifact is not loaded again on every Streamlit rerun.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    model_path = next((path for path in MODEL_CANDIDATES if path.is_file()), None)

    if model_path is None:
        expected = ", ".join(path.name for path in MODEL_CANDIDATES)
        existing = sorted(path.name for path in APP_DIR.iterdir() if path.is_file())
        raise FileNotFoundError(
            f"Could not find the saved model artifact next to app.py.\n"
            f"Expected one of: {expected}\n"
            f"Folder checked: {APP_DIR}\n"
            f"Files currently in that folder: {', '.join(existing) or '(none)'}\n\n"
            "Copy the .joblib file created by the notebook into the same folder as app.py."
        )

    bundle = joblib.load(model_path)
    required_keys = {"model", "vectorizer", "label_encoder", "intent_to_category"}
    if not isinstance(bundle, dict):
        raise ValueError(
            "The saved .joblib file is not in the expected dictionary format. "
            "Re-run the notebook's model-saving cell (the cell with joblib.dump)."
        )

    missing_keys = required_keys.difference(bundle.keys())
    if missing_keys:
        raise ValueError(
            "The saved model is missing required items: " + ", ".join(sorted(missing_keys))
            + ". Re-run the notebook's model-saving cell and try again."
        )

    return bundle, model_path.name


# -----------------------------------------------------------------------------
# Preprocessing: exactly matches the notebook's clean_text function.
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def analyze_message(message: str, bundle: dict) -> dict:
    cleaned_message = clean_text(message)
    if not cleaned_message:
        raise ValueError("Enter a customer message before clicking Predict intent.")

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    label_encoder = bundle["label_encoder"]
    intent_to_category = bundle["intent_to_category"]

    features = vectorizer.transform([cleaned_message])
    predicted_encoded = np.asarray(model.predict(features), dtype=int)
    predicted_intent = str(label_encoder.inverse_transform(predicted_encoded)[0])
    predicted_category = str(intent_to_category.get(predicted_intent, "Unknown"))

    ranked_results = []
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_ids = np.asarray(model.classes_, dtype=int)
        top_positions = np.argsort(probabilities)[::-1][:3]
        top_class_ids = class_ids[top_positions]
        top_intents = label_encoder.inverse_transform(top_class_ids)

        ranked_results = [
            {
                "Intent": str(intent),
                "Category": str(intent_to_category.get(str(intent), "Unknown")),
                "Model score": float(probabilities[position]),
            }
            for intent, position in zip(top_intents, top_positions)
        ]
        confidence = ranked_results[0]["Model score"] if ranked_results else None

    return {
        "intent": predicted_intent,
        "category": predicted_category,
        "confidence": confidence,
        "top_results": ranked_results,
        "cleaned_message": cleaned_message,
    }


# -----------------------------------------------------------------------------
# User interface. Keep the initial page responsive; load the model only on submit.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { max-width: 850px; padding-top: 2rem; padding-bottom: 3rem; }
      .hero { padding: 1.4rem 1.5rem; border: 1px solid #e2e8f0; border-radius: 16px;
              margin-bottom: 1.2rem; }
      .hero p { margin-bottom: 0; color: #64748b; }
      div.stButton > button[kind="primary"] { min-height: 2.7rem; font-weight: 600; }
    </style>
    <div class="hero">
      <h1>Customer Service Intent Classifier</h1>
      <p>Enter a customer message to predict its intent, category, and the model's top suggestions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("intent_prediction_form"):
    message = st.text_area(
        "Customer message",
        placeholder="Example: I need to cancel my order",
        height=130,
        help="Typos and extra spaces are okay; preprocessing follows the training notebook.",
    )
    submitted = st.form_submit_button(
        "Predict intent",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not clean_text(message):
        st.warning("Please enter a customer message first.")
    else:
        try:
            with st.spinner("Loading classifier and analyzing your message..."):
                bundle, artifact_name = load_pipeline()
                result = analyze_message(message, bundle)

            st.success("Prediction complete")
            intent_col, category_col, score_col = st.columns(3)
            intent_col.metric("Predicted intent", result["intent"].replace("_", " ").title())
            category_col.metric("Category", result["category"].replace("_", " ").title())
            if result["confidence"] is not None:
                score_col.metric("Top model score", f"{result['confidence']:.1%}")
            else:
                score_col.metric("Top model score", "Unavailable")

            if result["intent"] == "out_of_scope":
                st.info("This message may be outside the customer-service intents supported by this classifier.")

            if result["top_results"]:
                st.markdown("### Top 3 intent suggestions")
                scores = pd.DataFrame(result["top_results"])
                scores["Model score"] = scores["Model score"].map(lambda value: f"{value:.1%}")
                st.dataframe(scores, use_container_width=True, hide_index=True)

            with st.expander("Show text sent to the model"):
                st.write(result["cleaned_message"])

            st.caption(
                f"Loaded `{artifact_name}`. Model scores are estimates from the classifier, "
                "not guarantees that a prediction is correct."
            )

        except FileNotFoundError as exc:
            st.error("Saved model file not found.")
            st.code(str(exc))
        except Exception as exc:
            st.error("The message could not be classified. Check the details below.")
            st.exception(exc)
else:
    st.caption("The saved model loads only after you click **Predict intent**, so the app page can appear first.")
