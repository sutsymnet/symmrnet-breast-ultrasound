"""
streamlit_app.py - SymMRNet-Sym2 breast ultrasound classification prototype
==========================================================================
Chapter VI deployment prototype for the thesis.

WHAT CHANGED FROM THE PREVIOUS VERSION (and why):
  1. Loads ONLY symmrnet_symlet2_3blocks.h5. The ultrasound_model_vmc_net.h5
     fallback and the AdvancedLearnableEntropyPooling2D layer are gone. A
     fallback chain meant the app could silently serve a different model than
     the thesis claims - that is the traceability defect the examiner flagged.
  2. Preprocessing is now the real SymPreprocess (dwt2/sym2 -> sub-band
     rescale -> idwt2 -> clip), not "grayscale + /255". The old function was
     written for VMC-Net and would have produced meaningless predictions.
  3. Decode/resize now uses tf.image.resize bilinear to match
     image_dataset_from_directory exactly.
  4. Fails loudly. No "Strategy 3 partial load" that reports success while
     loading a broken graph.
  5. Shows the model fingerprint in the UI so a screenshot is self-documenting
     evidence for Table 7.2.

Requires in the same folder: symmrnet_core.py, symmrnet_symlet2_3blocks.h5
requirements.txt: streamlit, tensorflow, PyWavelets, numpy, pillow
"""

import numpy as np
import streamlit as st
from PIL import Image

from symmrnet_core import (
    CLASS_NAMES,
    EXPECTED_PARAM_COUNT,
    MODEL_FILENAME,
    apply_symlet2_preprocessing_numpy,
    decode_and_resize,
    load_symmrnet,
)
from input_gate import check_image_bytes

MODEL_LABEL = "SymMRNet-Symlet2-3Blocks (Wavelet-Sym2, preprocessed)"
TRAINING_SOURCE = "Kaggle: Ultrasound Breast Images for Breast Cancer (DS03.3)"
REPORTED_TEST_ACC = "93.90% (902 held-out images)"

st.set_page_config(page_title="SymMRNet Breast Ultrasound", page_icon="🩺",
                   layout="wide")


@st.cache_resource
def get_model():
    return load_symmrnet(MODEL_FILENAME, verify=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This prototype runs **SymMRNet**, a CNN in which every downsampling "
        "stage is a fixed Symlet-2-derived wavelet pooling operation, to "
        "classify breast ultrasound images as **Benign** or **Malignant**."
    )

    st.header("📋 Instructions")
    st.markdown(
        "1. Upload a breast ultrasound image\n"
        "2. Click **Analyze Image**\n"
        "3. Review the classification result"
    )

    st.header("🔒 Model provenance")
    st.code(
        f"weights : {MODEL_FILENAME}\n"
        f"model   : {MODEL_LABEL}\n"
        f"input   : 64 x 64 x 1 (grayscale)\n"
        f"params  : {EXPECTED_PARAM_COUNT:,}\n"
        f"decision: argmax (threshold 0.50)\n"
        f"trained : {TRAINING_SOURCE}\n"
        f"test acc: {REPORTED_TEST_ACC}",
        language="text",
    )

    st.header("⚠️ Disclaimer")
    st.warning(
        "Research prototype only. Not a medical device and not validated for "
        "clinical use. Always consult a qualified radiologist."
    )


# --------------------------------------------------------------- main panel
st.title("🩺 Breast Ultrasound Classification — SymMRNet")

try:
    model = get_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"❌ Model failed to load: {exc}")
    st.info(
        f"`{MODEL_FILENAME}` must sit next to `streamlit_app.py` in the repo "
        "root. There is deliberately no fallback model — serving a different "
        "network than the one reported in the thesis would invalidate the "
        "results."
    )
    st.stop()

st.success(
    f"✅ Loaded `{MODEL_FILENAME}` — verified {EXPECTED_PARAM_COUNT:,} "
    f"parameters, input 64×64×1"
)

uploaded = st.file_uploader(
    "Upload a breast ultrasound image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded is None:
    st.info("⬆️ Upload an image to begin.")
    st.stop()

image_bytes = uploaded.getvalue()

needs_confirm, gate_msg, gate_info = check_image_bytes(image_bytes)
if needs_confirm:
    st.warning(gate_msg)
    with st.expander("Technical detail"):
        st.json(gate_info)
    if not st.checkbox(
        "ยืนยันว่าเป็นภาพ B-mode และต้องการดำเนินการต่อ",
        key=f"confirm_{uploaded.name}",
    ):
        st.stop()
    st.caption("⚠️ ผู้ใช้ยืนยันดำเนินการต่อ — ผลทำนายอาจคลาดเคลื่อน")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input")
    st.image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)

if not st.button("🔍 Analyze Image", type="primary", use_container_width=True):
    st.stop()

with st.spinner("Running SymMRNet…"):
    try:
        raw = decode_and_resize(image_bytes)                  # (64,64,1) in [0,255]
        batch = apply_symlet2_preprocessing_numpy(raw[None])  # (1,64,64,1) in [0,1]
        probs = model.predict(batch, verbose=0)[0]            # softmax, 2 classes
    except Exception as exc:  # noqa: BLE001
        st.error(f"Inference failed: {exc}")
        st.stop()

pred_idx = int(np.argmax(probs))
pred_label = CLASS_NAMES[pred_idx].capitalize()
confidence = float(probs[pred_idx])

with col_right:
    st.subheader("Result")
    if pred_idx == 1:
        st.error(f"🔴 **{pred_label}** — {confidence:.1%} confidence")
    else:
        st.success(f"🟢 **{pred_label}** — {confidence:.1%} confidence")

    st.write("**Class probabilities**")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(float(p), text=f"{name.capitalize()}: {p:.4f}")

    st.caption(
        "Decision rule: argmax over the softmax output (equivalent to a 0.50 "
        "threshold), identical to the protocol used for the reported "
        "93.90% test accuracy. No per-image threshold tuning is applied."
    )

    with st.expander("Preprocessing applied to this image"):
        st.markdown(
            "1. Decode as single-channel grayscale\n"
            "2. Bilinear resize to 64×64\n"
            "3. Scale to [0, 1]\n"
            "4. Single-level `pywt.dwt2(..., 'sym2', mode='symmetric')`\n"
            "5. Rescale sub-bands: cA×1.05, cH×1.02, cV×1.02, cD×1.01\n"
            "6. `pywt.idwt2` reconstruction, crop/pad to 64×64, clip to [0, 1]\n\n"
            "No CLAHE, no histogram equalisation, no augmentation — matching "
            "the training pipeline exactly."
        )

st.divider()
st.caption(
    "SymMRNet = Symlet Multi-Resolution Network. Wavelet pooling here is a "
    "fixed Symlet-2-derived low-pass depthwise filter followed by ReLU and "
    "2×2 average pooling — an approximation of the LL sub-band rather than a "
    "full separable DWT."
)
