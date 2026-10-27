"""Local Streamlit interface for lung histopathology image predictions."""

import os
import hashlib

import streamlit as st

from src.inference import (
    MODEL_CONFIGS,
    load_model_bundle,
    model_is_available,
    predict_image_bytes,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def get_model_bundle(model_name):
    """Load each selected model once per running Streamlit process."""
    return load_model_bundle(PROJECT_ROOT, model_name)


st.set_page_config(
    page_title="LungScan | Image Classification",
    page_icon="🫁",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f8fc; }
    .block-container { max-width: 1120px; padding-top: 2.2rem; }
    .hero {
        padding: 2rem 2.2rem; border-radius: 22px;
        background: linear-gradient(120deg, #102b46, #176b72);
        color: white; margin-bottom: 1.4rem;
    }
    .hero h1 { color: white; font-size: 2.45rem; margin: 0; }
    .hero p { color: #d4e9e9; font-size: 1rem; margin: .65rem 0 0; }
    .result-card {
        background: white; border: 1px solid #e1eaf0; border-radius: 18px;
        padding: 1.3rem 1.5rem; box-shadow: 0 8px 24px #14314a0c;
    }
    .muted { color: #607080; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🫁 LungScan</h1>
      <p>Histopathology image classification · Upload an image to view the model output</p>
    </div>
    """,
    unsafe_allow_html=True,
)

available_models = [
    name for name in MODEL_CONFIGS if model_is_available(PROJECT_ROOT, name)
]
if not available_models:
    st.error(
        "No trained model was found. Train a model first, then restart this app."
    )
    st.stop()

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.subheader("Analyze an image")
    model_name = st.selectbox("Model", available_models)
    uploaded_file = st.file_uploader(
        "Choose a histopathology image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG and PNG.",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        analyze = st.button("Analyze image", type="primary", use_container_width=True)
        image_bytes = uploaded_file.getvalue()
        result_key = (model_name, hashlib.sha256(image_bytes).hexdigest())
    else:
        st.info("Upload a JPG or PNG image to begin.")
        analyze = False
        image_bytes = None
        result_key = None

with right:
    st.subheader("Prediction")
    if uploaded_file is not None and analyze:
        try:
            with st.spinner(f"Analyzing with the {model_name} model..."):
                model, metadata, preprocessing = get_model_bundle(model_name)
                result = predict_image_bytes(
                    image_bytes, model, metadata, preprocessing
                )

            st.session_state["prediction_result"] = {
                "key": result_key,
                "result": result,
                "metadata": metadata,
            }
        except (OSError, ValueError, KeyError, ImportError) as error:
            st.error(f"Could not complete prediction: {error}")
        except Exception as error:
            st.exception(error)

    saved_prediction = st.session_state.get("prediction_result")
    if saved_prediction and saved_prediction["key"] == result_key:
        result = saved_prediction["result"]
        metadata = saved_prediction["metadata"]
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.metric("Predicted class", result["label"])
        st.metric("Model confidence", f'{result["confidence"]:.1%}')
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Class probabilities")
        for class_name, probability in result["probabilities"].items():
            st.write(f"**{class_name}** — {probability:.1%}")
            st.progress(probability)

        st.caption(
            f"Model: {model_name} · Best method: "
            f"{metadata.get('best_method', 'saved model')} · "
            f"Validation accuracy: {metadata['accuracy']:.1%}"
        )
    else:
        st.markdown(
            '<div class="result-card muted">Upload an image and click '
            "Analyze image to see the prediction. The result stays visible "
            "while this image and model are selected.</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "Research/demo tool only. This model is not a medical device and does "
    "not provide a diagnosis. Results must be reviewed by a qualified clinician."
)
