"""Download checksum-verified MNIST archives and read their strict IDX format."""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mnist"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
CHECKSUMS = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def ensure_file(name: str) -> Path:
    """Validate cached data or atomically replace it with a verified download."""
    checksum = CHECKSUMS[name]
    destination = DATA_DIR / name
    if destination.is_file() and hashlib.md5(destination.read_bytes()).hexdigest() == checksum:
        return destination
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "riscv-mnist/2"})
    try:
        print(f"Downloading MNIST: {name}", flush=True)
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"MNIST checksum mismatch: {name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_mnist(split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    prefix = "train" if split == "train" else "t10k"
    image_path = ensure_file(f"{prefix}-images-idx3-ubyte.gz")
    label_path = ensure_file(f"{prefix}-labels-idx1-ubyte.gz")
    with gzip.open(image_path, "rb") as handle:
        image_bytes = handle.read()
    with gzip.open(label_path, "rb") as handle:
        label_bytes = handle.read()
    if len(image_bytes) < 16 or len(label_bytes) < 8:
        raise ValueError("Truncated MNIST header")
    magic, count, rows, columns = struct.unpack_from(">IIII", image_bytes)
    label_magic, label_count = struct.unpack_from(">II", label_bytes)
    expected_count = 60000 if split == "train" else 10000
    if (magic, count, rows, columns) != (2051, expected_count, 28, 28):
        raise ValueError("Invalid MNIST image header")
    if (label_magic, label_count) != (2049, count):
        raise ValueError("Invalid MNIST label header")
    if len(image_bytes) != 16 + count * 784 or len(label_bytes) != 8 + count:
        raise ValueError("Invalid MNIST payload length")
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(count, 784)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    if np.any(labels > 9):
        raise ValueError("Invalid MNIST label")
    return images, labels
