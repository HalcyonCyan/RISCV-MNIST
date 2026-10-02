"""Train a 784->64->10 MNIST MLP and export the RISC-V V2 integer model.

Run ``python -m training.train`` from the project directory. NumPy implements
the forward pass, cross-entropy gradient, backpropagation and Adam explicitly;
inference in the app runs the RISC-V assembly, not this training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Set before NumPy imports; fewer BLAS threads suit this compact network.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from .data import load_mnist
from .quantization import forward_quantized, quantize, save_model, suggested_shift


def float_scores(images: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    images = np.asarray(images).reshape(len(images), 784)
    scores = np.empty((len(images), 10), dtype=np.float32)
    for start in range(0, len(images), 1024):
        x = images[start:start + 1024].astype(np.float32) / 255
        hidden = np.maximum(x @ params["w1"].T + params["b1"], 0)
        scores[start:start + len(x)] = hidden @ params["w2"].T + params["b2"]
    return scores


def accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == labels))


def train(
    epochs: int = 25, batch_size: int = 128, seed: int = 20261002,
    output: Path = Path("artifacts/retrained"),
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    all_images, all_labels = load_mnist("train")
    all_images = all_images.reshape(len(all_images), 784)
    order = rng.permutation(len(all_images))
    # The public MNIST test set never participates in fitting, stopping or calibration.
    fitting, validation = order[:-5000], order[-5000:]
    x_train = all_images[fitting].astype(np.float32) / 255
    y_train = all_labels[fitting].astype(np.int64)
    val_images, val_labels = all_images[validation], all_labels[validation]
    calibration = all_images[fitting[:4096]]
    params = {
        "w1": (rng.standard_normal((64, 784)) * np.sqrt(2 / 784)).astype(np.float32),
        "b1": np.zeros(64, dtype=np.float32),
        "w2": (rng.standard_normal((10, 64)) * np.sqrt(2 / 64)).astype(np.float32),
        "b2": np.zeros(10, dtype=np.float32),
    }
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    best_params = {name: value.copy() for name, value in params.items()}
    best_accuracy = -1.0
    best_epoch = 0
    step = 0
    weight_decay = 1e-5
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        permutation = rng.permutation(len(x_train))
        # Decay towards the end for better held-out accuracy and quantization stability.
        rate = 0.0015 * (0.4 if epoch > epochs * 0.75 else 0.7 if epoch > epochs * 0.5 else 1.0)
        loss_sum = 0.0
        correct = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            x, y = x_train[indices], y_train[indices]
            hidden_linear = x @ params["w1"].T + params["b1"]
            hidden = np.maximum(hidden_linear, 0)
            logits = hidden @ params["w2"].T + params["b2"]
            logits -= logits.max(axis=1, keepdims=True)
            exponential = np.exp(logits)
            probabilities = exponential / exponential.sum(axis=1, keepdims=True)
            n = len(x)
            rows = np.arange(n)
            loss_sum -= float(np.log(np.maximum(probabilities[rows, y], 1e-12)).sum())
            correct += int(np.count_nonzero(np.argmax(probabilities, axis=1) == y))
            probabilities[rows, y] -= 1
            probabilities /= n
            hidden_gradient = probabilities @ params["w2"]
            hidden_gradient *= hidden_linear > 0
            gradients = {
                "w2": probabilities.T @ hidden + weight_decay * params["w2"],
                "b2": probabilities.sum(axis=0),
                "w1": hidden_gradient.T @ x + weight_decay * params["w1"],
                "b1": hidden_gradient.sum(axis=0),
            }
            step += 1
            first_correction, second_correction = 1 - 0.9**step, 1 - 0.999**step
            for name in params:
                first[name] *= 0.9
                first[name] += 0.1 * gradients[name]
                second[name] *= 0.999
                second[name] += 0.001 * gradients[name] ** 2
                params[name] -= rate * (first[name] / first_correction) / (
                    np.sqrt(second[name] / second_correction) + 1e-8
                )
        val_accuracy = accuracy(float_scores(val_images, params), val_labels)
        entry: dict[str, float | int] = {
            "epoch": epoch, "learning_rate": rate,
            "train_loss": loss_sum / len(x_train), "train_accuracy": correct / len(x_train),
            "validation_accuracy": val_accuracy, "seconds": time.perf_counter() - epoch_started,
        }
        print(
            f"Epoch {epoch:02d}/{epochs}: loss={entry['train_loss']:.4f} "
            f"train={entry['train_accuracy']:.2%} validation={val_accuracy:.2%} "
            f"({entry['seconds']:.1f}s)", flush=True,
        )
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            best_params = {name: value.copy() for name, value in params.items()}

    base_shift = suggested_shift(best_params, calibration)
    best_quantized_accuracy = -1.0
    quantized = None
    for shift in range(max(1, base_shift - 2), min(30, base_shift + 1) + 1):
        candidate = quantize(best_params, calibration, shift)
        candidate_accuracy = accuracy(forward_quantized(val_images, candidate), val_labels)
        print(f"Quantization shift={shift}: validation={candidate_accuracy:.2%}", flush=True)
        if candidate_accuracy > best_quantized_accuracy:
            best_quantized_accuracy, quantized = candidate_accuracy, candidate
    assert quantized is not None
    # Final, one-time report: no choice above depends on the test results.
    test_images, test_labels = load_mnist("test")
    float_test_accuracy = accuracy(float_scores(test_images, best_params), test_labels)
    quantized_test_accuracy = accuracy(forward_quantized(test_images, quantized), test_labels)
    model_dir = Path(output)
    model_dir.mkdir(parents=True, exist_ok=True)
    binary_path = model_dir / "model.bin"
    save_model(quantized, binary_path)
    metadata: dict[str, object] = {
        "format_version": 2, "architecture": [784, 64, 10], "activation": "ReLU",
        "layout": "W1 input-major int8; B1 int32; W2 output-major int8; B2 int32",
        "seed": seed, "epochs": epochs, "batch_size": batch_size, "best_epoch": best_epoch,
        "optimizer": "Adam", "weight_decay": weight_decay,
        "training_samples": len(fitting), "validation_samples": len(validation),
        "calibration_samples": len(calibration), "test_samples": len(test_labels),
        "float_validation_accuracy": best_accuracy,
        "quantized_validation_accuracy": best_quantized_accuracy,
        "float_test_accuracy": float_test_accuracy,
        "quantized_test_accuracy": quantized_test_accuracy,
        "shift": quantized.shift, "logit_scale": quantized.logit_scale,
        "integer_formula": "h=clip((x@W1.T+B1)>>shift,0,255); scores=h@W2.T+B2",
        "quantization": "per-hidden-neuron W1 int8; scale folded into W2; training-only bias calibration",
        "model_sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest(),
        "model_bytes": binary_path.stat().st_size,
        "training_seconds": time.perf_counter() - started,
        "evaluation_note": "Test data used only for final accuracy reporting; choices use training validation.",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Saved {binary_path}\nFloat test accuracy: {float_test_accuracy:.2%}\n"
        f"Integer test accuracy: {quantized_test_accuracy:.2%}\n"
        f"Elapsed: {metadata['training_seconds']:.1f}s", flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrained"),
                        help="Output directory (default: artifacts/retrained)")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.seed, args.output)


if __name__ == "__main__":
    main()
