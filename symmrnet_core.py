"""
symmrnet_core.py
================
Single source of truth for the SymMRNet-Sym2 deployed model.

Both `streamlit_app.py` and `evaluate_ec_batch.py` import from this file so the
inference path CANNOT drift between the web prototype (Ch.6) and the clinical
evaluation (Ch.7).

Provenance
----------
Layer class and preprocessing are copied VERBATIM from the training notebook
    147.4_3Block_TrainValTestDS03.3_SymPreprocess+Sym2Pooling_Epoch12_BS=8.ipynb
(cells 2, 4, 5). Do not "fix" or "improve" anything in this file: the trained
weights in symmrnet_symlet2_3blocks.h5 are bound to these exact operations.
"""

import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.layers import Layer

# ---------------------------------------------------------------- constants
# Verbatim from notebook cell 2
NUM_CLASSES = 2
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 1

# Verbatim from notebook cell 19/20: class_names = test_ds.class_names
# image_dataset_from_directory sorts folder names alphabetically.
CLASS_NAMES = ["benign", "malignant"]
BENIGN_IDX, MALIGNANT_IDX = 0, 1

MODEL_FILENAME = "symmrnet_symlet2_3blocks.h5"

# Expected fingerprint - asserted at load time so a wrong file fails loudly.
EXPECTED_INPUT_SHAPE = (None, 64, 64, 1)
EXPECTED_PARAM_COUNT = 565_658


# ------------------------------------------------------- Symlet2 pooling
class Symlet2PoolingLayer(Layer):
    """
    Pure TensorFlow implementation of Symlet2 wavelet pooling.
    NOTE: VALID padding, matching the notebook's pooling behaviour

    NOTE FOR THE THESIS (do not change the code, change the text):
    This is a fixed Symlet-2-derived low-pass depthwise filter followed by ReLU
    and 2x2 average pooling. It is an APPROXIMATION of the LL sub-band, not a
    separable two-dimensional DWT. The high-pass kernel `g_kernel` is
    constructed but intentionally unused - detail sub-bands are discarded.
    """

    def __init__(self, pool_size=(2, 2), **kwargs):
        super(Symlet2PoolingLayer, self).__init__(**kwargs)
        self.pool_size = pool_size

        # Actual Symlet2 filter coefficients (h0, h1, h2, h3)
        symlet2_h = [
            -0.12940952255092145,
            0.22414386804185735,
            0.836516303737469,
            0.48296291314469025,
        ]
        symlet2_g = [
            -0.48296291314469025,
            0.836516303737469,
            -0.22414386804185735,
            -0.12940952255092145,
        ]

        self.h_kernel = tf.constant(
            [[symlet2_h[0], symlet2_h[1]], [symlet2_h[2], symlet2_h[3]]],
            dtype=tf.float32,
        )
        self.g_kernel = tf.constant(
            [[symlet2_g[0], symlet2_g[1]], [symlet2_g[2], symlet2_g[3]]],
            dtype=tf.float32,
        )

        self.h_kernel = self.h_kernel / tf.reduce_sum(tf.abs(self.h_kernel))
        self.g_kernel = self.g_kernel / tf.reduce_sum(tf.abs(self.g_kernel))

    def build(self, input_shape):
        self.channels = input_shape[-1]
        super(Symlet2PoolingLayer, self).build(input_shape)

    def call(self, inputs):
        h_kernel_expanded = tf.reshape(self.h_kernel, [2, 2, 1, 1])
        h_kernel_tiled = tf.tile(h_kernel_expanded, [1, 1, self.channels, 1])

        low_pass = tf.nn.depthwise_conv2d(
            inputs, h_kernel_tiled, strides=[1, 1, 1, 1], padding="SAME"
        )
        activated = tf.nn.relu(low_pass)
        pooled = tf.nn.avg_pool2d(
            activated,
            ksize=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.pool_size[0], self.pool_size[1], 1],
            padding="VALID",
        )
        return pooled

    def compute_output_shape(self, input_shape):
        def calc(n, p, s):
            return None if n is None else (n - p) // s + 1

        return (
            input_shape[0],
            calc(input_shape[1], self.pool_size[0], self.pool_size[0]),
            calc(input_shape[2], self.pool_size[1], self.pool_size[1]),
            input_shape[3],
        )

    def get_config(self):
        config = super().get_config()
        config.update({"pool_size": self.pool_size})
        return config


CUSTOM_OBJECTS = {"Symlet2PoolingLayer": Symlet2PoolingLayer}


# ----------------------------------------------------- SymPreprocess stage
def apply_symlet2_preprocessing_numpy(image_batch):
    """
    Numpy-based Symlet2 preprocessing function. Verbatim from notebook cell 5.

    Input : float32 array (B, 64, 64, 1) with values in [0, 255]
    Output: float32 array (B, 64, 64, 1) with values in [0, 1]

    IMPORTANT: there is NO CLAHE here. CLAHE was never part of the training
    pipeline. Adding it creates a train/deploy mismatch.
    """
    if not isinstance(image_batch, np.ndarray):
        image_batch = np.array(image_batch)

    if image_batch.dtype != np.float32:
        image_batch = image_batch.astype(np.float32)

    if np.max(image_batch) > 1.0:
        image_batch = image_batch / 255.0

    batch_size = image_batch.shape[0]
    processed_batch = np.zeros_like(image_batch, dtype=np.float32)

    for b in range(batch_size):
        for c in range(IMAGE_CHANNELS):
            channel_data = image_batch[b, :, :, c]
            try:
                coeffs = pywt.dwt2(channel_data, "sym2", mode="symmetric")
                cA, (cH, cV, cD) = coeffs

                enhanced_cA = cA * 1.05
                enhanced_cH = cH * 1.02
                enhanced_cV = cV * 1.02
                enhanced_cD = cD * 1.01

                reconstructed = pywt.idwt2(
                    (enhanced_cA, (enhanced_cH, enhanced_cV, enhanced_cD)),
                    "sym2",
                    mode="symmetric",
                )

                target_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
                if reconstructed.shape != target_shape:
                    if (
                        reconstructed.shape[0] > target_shape[0]
                        or reconstructed.shape[1] > target_shape[1]
                    ):
                        reconstructed = reconstructed[
                            : target_shape[0], : target_shape[1]
                        ]
                    else:
                        pad_h = max(0, target_shape[0] - reconstructed.shape[0])
                        pad_w = max(0, target_shape[1] - reconstructed.shape[1])
                        if pad_h > 0 or pad_w > 0:
                            reconstructed = np.pad(
                                reconstructed, ((0, pad_h), (0, pad_w)), mode="edge"
                            )

                reconstructed = np.clip(reconstructed, 0, 1)
                processed_batch[b, :, :, c] = reconstructed

            except Exception as e:  # noqa: BLE001 - matches notebook behaviour
                print(f"Wavelet processing failed for batch {b}, channel {c}: {e}")
                enhanced_original = channel_data * 1.02
                processed_batch[b, :, :, c] = np.clip(enhanced_original, 0, 1)

    return processed_batch.astype(np.float32)


# --------------------------------------------------------- decode + resize
def decode_and_resize(image_bytes):
    """
    Reproduce the decode/resize used by tf.keras.utils.image_dataset_from_directory
    with color_mode='grayscale', image_size=(64, 64).

    tf.image.resize defaults: method='bilinear', antialias=False. Matching this
    exactly matters - PIL/cv2 resamplers give slightly different pixel values.

    Returns float32 array (64, 64, 1) with values in [0, 255].
    """
    img = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])  # bilinear, no antialias
    return img.numpy()


def prepare_batch(image_bytes_list):
    """image bytes -> model-ready float32 batch (N, 64, 64, 1) in [0, 1]."""
    raw = np.stack([decode_and_resize(b) for b in image_bytes_list], axis=0)
    return apply_symlet2_preprocessing_numpy(raw)


# ------------------------------------------------------------- model load
def load_symmrnet(model_path=MODEL_FILENAME, verify=True):
    """
    Load the one and only deployed model. No fallback list, no silent
    substitution: if this file is missing or wrong, raise.
    """
    model = tf.keras.models.load_model(
        model_path, custom_objects=CUSTOM_OBJECTS, compile=False
    )

    if verify:
        got_shape = tuple(model.input_shape)
        got_params = int(model.count_params())
        if got_shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"Input shape mismatch: expected {EXPECTED_INPUT_SHAPE}, got {got_shape}"
            )
        if got_params != EXPECTED_PARAM_COUNT:
            raise ValueError(
                f"Parameter count mismatch: expected {EXPECTED_PARAM_COUNT:,}, "
                f"got {got_params:,} - this is NOT the Ch.5 Wavelet-Sym2 model."
            )
    return model
