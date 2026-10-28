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
    st.subheader("Analyze images")
    model_name = st.selectbox("Model", available_models)
    uploaded_files = st.file_uploader(
        "Choose histopathology images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supported formats: JPG and PNG.",
    )

    if uploaded_files:
        image_bytes_list = [uploaded_file.getvalue() for uploaded_file in uploaded_files]
        result_key = (
            model_name,
            tuple(
                (uploaded_file.name, hashlib.sha256(image_bytes).hexdigest())
                for uploaded_file, image_bytes in zip(uploaded_files, image_bytes_list)
            ),
        )
        with st.expander(f"Preview {len(uploaded_files)} selected image(s)"):
            for uploaded_file in uploaded_files:
                st.image(
                    uploaded_file,
                    caption=uploaded_file.name,
                    use_container_width=True,
                )
        analyze = st.button(
            f"Analyze {len(uploaded_files)} image(s)",
            type="primary",
            use_container_width=True,
        )
    else:
        st.info("Upload one or more JPG or PNG images to begin.")
        analyze = False
        image_bytes_list = []
        result_key = None

with right:
    st.subheader("Prediction")
    saved_prediction = st.session_state.get("prediction_batch")
    if saved_prediction and (not uploaded_files or saved_prediction.get("key") != result_key):
        st.session_state.pop("prediction_batch", None)
        saved_prediction = None

    if uploaded_files and analyze:
        batch_results = []
        try:
            with st.spinner(
                f"Analyzing {len(uploaded_files)} image(s) with the {model_name} model..."
            ):
                model, metadata, preprocessing = get_model_bundle(model_name)
                for uploaded_file, image_bytes in zip(uploaded_files, image_bytes_list):
                    try:
                        result = predict_image_bytes(
                            image_bytes, model, metadata, preprocessing
                        )
                        batch_results.append(
                            {
                                "filename": uploaded_file.name,
                                "result": result,
                                "error": None,
                            }
                        )
                    except (OSError, ValueError, KeyError, ImportError) as error:
                        batch_results.append(
                            {
                                "filename": uploaded_file.name,
                                "result": None,
                                "error": str(error),
                            }
                        )
                    except Exception as error:
                        batch_results.append(
                            {
                                "filename": uploaded_file.name,
                                "result": None,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
        except (OSError, ValueError, KeyError, ImportError) as error:
            st.error(f"Could not load the {model_name} model: {error}")
        except Exception as error:
            st.exception(error)
        else:
            st.session_state["prediction_batch"] = {
                "key": result_key,
                "results": batch_results,
                "metadata": metadata,
            }
            saved_prediction = st.session_state["prediction_batch"]

    if saved_prediction and saved_prediction["key"] == result_key:
        metadata = saved_prediction["metadata"]
        rows = []
        for item in saved_prediction["results"]:
            result = item["result"]
            rows.append(
                {
                    "Image": item["filename"],
                    "Predicted class": result["label"] if result else "Failed",
                    "Confidence": f'{result["confidence"]:.1%}' if result else "—",
                    "Status": "Success" if result else item["error"],
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)

        for item in saved_prediction["results"]:
            result = item["result"]
            with st.expander(
                f'{item["filename"]} — {result["label"] if result else "Prediction failed"}'
            ):
                if result:
                    st.metric("Predicted class", result["label"])
                    st.metric("Model confidence", f'{result["confidence"]:.1%}')
                    st.markdown("#### Class probabilities")
                    for class_name, probability in result["probabilities"].items():
                        st.write(f"**{class_name}** — {probability:.1%}")
                        st.progress(probability)
                else:
                    st.error(f'Could not predict this image: {item["error"]}')

        st.caption(
            f"Model: {model_name} · Best method: "
            f"{metadata.get('best_method', 'saved model')} · "
            f"Validation accuracy: {metadata['accuracy']:.1%}"
        )
    else:
        st.markdown(
            '<div class="result-card muted">Upload an image and click '
            "Analyze images to see the predictions. Results stay visible "
            "while these images and model are selected.</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "Research/demo tool only. This model is not a medical device and does "
    "not provide a diagnosis. Results must be reviewed by a qualified clinician."
)
