"""Safe int8 quantization, V2 binary export, and a bit-exact NumPy reference."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x324E4E52
HEADER = struct.Struct("<8I")
PAYLOAD_BYTES = 51_112
MODEL_BYTES = HEADER.size + PAYLOAD_BYTES
INPUTS, HIDDEN, OUTPUTS = 784, 64, 10
INT32_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True)
class QuantizedModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    shift: int
    logit_scale: float = 1.0


def _validate(model: QuantizedModel) -> None:
    for name, shape, dtype in (
        ("w1", (HIDDEN, INPUTS), np.int8),
        ("b1", (HIDDEN,), np.int32),
        ("w2", (OUTPUTS, HIDDEN), np.int8),
        ("b2", (OUTPUTS,), np.int32),
    ):
        values = getattr(model, name)
        if values.shape != shape or values.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if not 1 <= model.shift <= 30:
        raise ValueError("The hidden arithmetic right shift must be in 1..30")
    # A bound for every possible image, including adversarial uint8 inputs.
    for weights, biases in ((model.w1, model.b1), (model.w2, model.b2)):
        bound = np.abs(weights.astype(np.int64)).sum(axis=1) * 255
        bound += np.abs(biases.astype(np.int64))
        if np.any(bound > INT32_MAX):
            raise ValueError("Model may overflow an RV32 signed accumulator")


def save_model(model: QuantizedModel, path: Path) -> None:
    """Export little-endian V2: input-major W1, B1, output-major W2, B2."""
    _validate(model)
    header = HEADER.pack(MAGIC, 2, INPUTS, HIDDEN, OUTPUTS, model.shift, PAYLOAD_BYTES, 0)
    payload = b"".join(
        values.astype(dtype, copy=False).tobytes(order="C")
        for values, dtype in (
            (model.w1.T, "i1"), (model.b1, "<i4"),
            (model.w2, "i1"), (model.b2, "<i4"),
        )
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload)


def load_model(path: Path) -> QuantizedModel:
    """Read a binary model, rejecting malformed headers and unsafe accumulators."""
    raw = Path(path).read_bytes()
    if len(raw) != MODEL_BYTES:
        raise ValueError(f"Model must contain exactly {MODEL_BYTES} bytes")
    magic, version, n_in, n_hidden, n_out, shift, payload_size, reserved = HEADER.unpack_from(raw)
    if (magic, version, n_in, n_hidden, n_out, payload_size, reserved) != (
        MAGIC, 2, INPUTS, HIDDEN, OUTPUTS, PAYLOAD_BYTES, 0
    ):
        raise ValueError("Unsupported or corrupt model header")
    offset = HEADER.size
    arrays: list[np.ndarray] = []
    for count, dtype, shape in (
        (HIDDEN * INPUTS, "i1", (INPUTS, HIDDEN)),
        (HIDDEN, "<i4", (HIDDEN,)),
        (OUTPUTS * HIDDEN, "i1", (OUTPUTS, HIDDEN)),
        (OUTPUTS, "<i4", (OUTPUTS,)),
    ):
        values = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
        arrays.append(values)
        offset += values.nbytes
    arrays[0] = arrays[0].T.copy()
    model = QuantizedModel(*arrays, shift=shift)
    _validate(model)
    return model


def forward_quantized(images: np.ndarray, model: QuantizedModel) -> np.ndarray:
    """Return exact int32 logits; arithmetic shift deliberately adds no rounding.

    Float64 GEMM is used only as an exact accelerator for the integer sums.
    All inputs and weights are integers, and validated accumulator bounds are
    far below 2**53, so every multiplication and partial sum is representable.
    This produces the same logits as int32 matrix multiplication and RV32IM.
    """
    _validate(model)
    images = np.asarray(images)
    if images.ndim == 1:
        images = images.reshape(1, -1)
    if images.ndim == 3 and images.shape[1:] == (28, 28):
        images = images.reshape(len(images), INPUTS)
    if images.ndim != 2 or images.shape[1] != INPUTS or images.dtype != np.uint8:
        raise ValueError("Images must be a uint8 array with shape (N,784) or (N,28,28)")
    scores = np.empty((len(images), OUTPUTS), dtype=np.int32)
    w1 = model.w1.astype(np.float64)
    w2 = model.w2.astype(np.float64)
    for start in range(0, len(images), 512):
        batch = images[start:start + 512].astype(np.float64)
        accumulator = (batch @ w1.T).astype(np.int32) + model.b1
        hidden = np.clip(accumulator >> model.shift, 0, 255)
        scores[start:start + len(batch)] = (hidden.astype(np.float64) @ w2.T).astype(np.int32) + model.b2
    return scores


def suggested_shift(params: dict[str, np.ndarray], calibration: np.ndarray) -> int:
    """Estimate a common shift from training-only activation statistics."""
    scales = np.maximum(np.max(np.abs(params["w1"]), axis=1) / 127, 1e-12)
    hidden = np.maximum(calibration.astype(np.float32) / 255 @ params["w1"].T + params["b1"], 0)
    accumulator = hidden * (255 / scales)
    high = max(float(np.percentile(accumulator, 99.99)), 1.0)
    return int(np.clip(np.ceil(np.log2(high / 255)), 1, 30))


def quantize(
    params: dict[str, np.ndarray], calibration: np.ndarray, shift: int
) -> QuantizedModel:
    """Fold per-neuron first-layer scales into second-layer weights.

    Calibration comes only from the training split. A mean output-bias
    correction compensates for quantization error, including the truncating
    hidden shift, without altering the assembly arithmetic.
    """
    if not 1 <= shift <= 30:
        raise ValueError("The hidden arithmetic right shift must be in 1..30")
    for name, shape in (("w1", (HIDDEN, INPUTS)), ("b1", (HIDDEN,)),
                        ("w2", (OUTPUTS, HIDDEN)), ("b2", (OUTPUTS,))):
        if params[name].shape != shape or not np.all(np.isfinite(params[name])):
            raise ValueError(f"Invalid floating-point parameter: {name}")
    calibration = np.asarray(calibration)
    if (calibration.ndim != 2 or calibration.shape[1] != INPUTS
            or len(calibration) == 0 or calibration.dtype != np.uint8):
        raise ValueError("Calibration must be nonempty uint8 pixels with shape (N,784)")
    first_scales = np.maximum(np.max(np.abs(params["w1"]), axis=1) / 127, 1e-12)
    w1 = np.clip(np.rint(params["w1"] / first_scales[:, None]), -127, 127).astype(np.int8)
    b1_float = np.rint(params["b1"] * 255 / first_scales)
    if not np.all(np.isfinite(b1_float)) or np.any(np.abs(b1_float) > INT32_MAX):
        raise ValueError("First-layer quantized biases overflow int32")
    b1 = b1_float.astype(np.int32)
    first_bound = np.abs(w1.astype(np.int64)).sum(axis=1) * 255 + np.abs(b1.astype(np.int64))
    if np.any(first_bound > INT32_MAX):
        raise ValueError("First-layer quantized accumulator may overflow int32")
    hidden_scales = first_scales * (2**shift / 255)
    scaled_w2 = params["w2"] * hidden_scales[None, :]
    if not np.all(np.isfinite(scaled_w2)):
        raise ValueError("Second-layer scales are not finite")
    logit_scale = max(float(np.max(np.abs(scaled_w2)) / 127), 1e-12)
    w2 = np.clip(np.rint(scaled_w2 / logit_scale), -127, 127).astype(np.int8)

    float_hidden = np.maximum(calibration.astype(np.float32) / 255 @ params["w1"].T + params["b1"], 0)
    accumulator = (calibration.astype(np.float64) @ w1.astype(np.float64).T).astype(np.int32) + b1
    integer_hidden = np.clip(accumulator >> shift, 0, 255)
    float_logits = float_hidden @ params["w2"].T + params["b2"]
    quant_logits_without_bias = integer_hidden.astype(np.float64) @ w2.astype(np.float64).T * logit_scale
    corrected_bias = np.mean(float_logits - quant_logits_without_bias, axis=0)
    b2_float = np.rint(corrected_bias / logit_scale)
    if not np.all(np.isfinite(b2_float)) or np.any(np.abs(b2_float) > INT32_MAX):
        raise ValueError("Second-layer quantized biases overflow int32")
    model = QuantizedModel(w1, b1, w2, b2_float.astype(np.int32), shift, logit_scale)
    _validate(model)
    return model
