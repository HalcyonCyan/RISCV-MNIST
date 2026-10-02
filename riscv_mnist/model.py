"""Validate the bundled integer model without scientific dependencies."""
from __future__ import annotations

import struct
from pathlib import Path

from .paths import ROOT

HEADER = struct.Struct("<8I")
MODEL_BYTES = 51144
PAYLOAD_BYTES = 51112


def validate_model(path: Path | None = None) -> dict[str, int]:
    """Check format, dimensions, shift, and conservative int32 MAC bounds."""
    model_path = Path(path) if path is not None else ROOT / "models" / "model.bin"
    raw = model_path.read_bytes()
    if len(raw) != MODEL_BYTES:
        raise ValueError(f"Model must contain exactly {MODEL_BYTES} bytes")
    magic, version, inputs, hidden, outputs, shift, size, reserved = HEADER.unpack_from(raw)
    if (magic, version, inputs, hidden, outputs, size, reserved) != (
        0x324E4E52, 2, 784, 64, 10, PAYLOAD_BYTES, 0
    ):
        raise ValueError("Invalid model header or unsupported architecture")
    if not 1 <= shift <= 30:
        raise ValueError("Hidden-layer shift must be between 1 and 30")

    # W1 is input-major: signed int8 W1[pixel, neuron]. Activations are uint8.
    weights1 = struct.unpack_from("<50176b", raw, 32)
    biases1 = struct.unpack_from("<64i", raw, 50208)
    weights2 = struct.unpack_from("<640b", raw, 50464)
    biases2 = struct.unpack_from("<10i", raw, 51104)
    for neuron, bias in enumerate(biases1):
        bound = abs(bias) + 255 * sum(abs(weights1[pixel * 64 + neuron]) for pixel in range(784))
        if bound > 2**31 - 1:
            raise ValueError(f"First-layer neuron {neuron} may overflow int32")
    for neuron, bias in enumerate(biases2):
        bound = abs(bias) + 255 * sum(abs(weight) for weight in weights2[neuron * 64:(neuron + 1) * 64])
        if bound > 2**31 - 1:
            raise ValueError(f"Output neuron {neuron} may overflow int32")
    return {"version": version, "inputs": inputs, "hidden": hidden,
            "outputs": outputs, "shift": shift, "bytes": len(raw)}
