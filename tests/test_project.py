"""Check the standalone API, preprocessing, and actual assembly predictions."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from PIL import Image, ImageDraw, ImageOps

from riscv_mnist.model import validate_model
from riscv_mnist.paths import ROOT
from riscv_mnist.preprocess import preprocess_image
from riscv_mnist.runtime import Prediction, RiscvSession
from riscv_mnist.server import Application, decode_input, make_handler


class PreprocessingTests(unittest.TestCase):
    def test_empty_backgrounds_and_polarity(self) -> None:
        for level in (0, 80, 180, 255):
            self.assertEqual(preprocess_image(Image.new("L", (100, 64), level)), bytes(784))
        source = Image.new("L", (150, 100), 255)
        ImageDraw.Draw(source).line((15, 15, 65, 15, 35, 80), fill=0, width=9)
        pixels = preprocess_image(source)
        self.assertEqual(len(pixels), 784)
        self.assertTrue(any(pixels))
        self.assertEqual(pixels, preprocess_image(ImageOps.invert(source)))

    def test_transparent_ink_and_center_of_mass(self) -> None:
        results = []
        for ink in (0, 255):
            image = Image.new("RGBA", (100, 120), (0, 0, 0, 0))
            ImageDraw.Draw(image).line((15, 15, 15, 85, 55, 85), fill=(ink, ink, ink, 255), width=8)
            results.append(preprocess_image(image))
        self.assertEqual(results[0], results[1])
        pixels = results[0]
        mass = sum(pixels)
        self.assertGreater(mass, 0)
        self.assertLessEqual(abs(sum((i % 28) * v for i, v in enumerate(pixels)) / mass - 13.5), 0.51)
        self.assertLessEqual(abs(sum((i // 28) * v for i, v in enumerate(pixels)) / mass - 13.5), 0.51)
        self.assertFalse(any(pixels[:28]) or any(pixels[-28:]))
        self.assertFalse(any(pixels[::28]) or any(pixels[27::28]))


class AssetAndInputTests(unittest.TestCase):
    def test_model_header_and_unsafe_accumulators(self) -> None:
        self.assertEqual(validate_model()["bytes"], 51144)
        original = (ROOT / "models" / "model.bin").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bin"
            for raw in (original[:-1], original + b"\0"):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    validate_model(path)
            unsafe = bytearray(original)
            struct.pack_into("<i", unsafe, 50208, 2**31 - 1)
            path.write_bytes(unsafe)
            with self.assertRaisesRegex(ValueError, "overflow"):
                validate_model(path)

    def test_input_formats_and_rejections(self) -> None:
        pixels = [0] * 784
        pixels[300] = 255
        self.assertEqual(decode_input({"pixels": pixels}), bytes(pixels))
        image = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(image).line((10, 10, 50, 10, 20, 55), fill=255, width=7)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
        self.assertEqual(decode_input({"image": url}), preprocess_image(image))
        for invalid in ({}, [], {"pixels": [0]}, {"pixels": [True] * 784},
                        {"pixels": [-1] * 784}, {"image": "data:image/png;base64,???"},
                        {"pixels": pixels, "image": url}):
            with self.subTest(invalid=str(invalid)[:50]), self.assertRaises(ValueError):
                decode_input(invalid)


class FakeSession:
    closed = False

    def __init__(self) -> None:
        self.images: list[bytes] = []

    def predict(self, pixels: bytes) -> Prediction:
        self.images.append(pixels)
        return Prediction(7, (0, 0, 0, 0, 0, 0, 0, 1000, 0, 0), 0.012)


class HttpApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = FakeSession()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Application(self.session)))
        self.worker = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05})
        self.worker.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=2)

    def get(self, route: str) -> object:
        with urllib.request.urlopen(self.url + route, timeout=5) as response:
            return json.load(response)

    def post(self, payload: object) -> object:
        request = urllib.request.Request(self.url + "/api/predict", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_samples_preserve_raw_pixels_through_api(self) -> None:
        self.assertTrue(self.get("/api/health")["ready"])
        sample = self.get("/api/sample?digit=7")
        result = self.post({"pixels": sample["pixels"]})
        self.assertEqual(result["digit"], 7)
        self.assertEqual(result["pixels"], sample["pixels"])
        self.assertEqual(self.session.images, [bytes(sample["pixels"])])
        self.assertEqual(result["elapsed_ms"], 12)
        self.assertAlmostEqual(sum(result["probabilities"]), 1)

    def test_empty_invalid_and_private_paths(self) -> None:
        for payload in ({"pixels": [0] * 784}, {"pixels": [255]}, {"pixels": [256] * 784}):
            with self.assertRaises(urllib.error.HTTPError) as failure:
                self.post(payload)
            self.assertEqual(failure.exception.code, 400)
        self.assertEqual(self.session.images, [])
        for route, expected in (("/api/sample?digit=22", 400), ("/models/model.bin", 404), ("/../app.py", 404)):
            with self.assertRaises(urllib.error.HTTPError) as failure:
                self.get(route)
            self.assertEqual(failure.exception.code, expected)
        with urllib.request.urlopen(self.url, timeout=5) as response:
            self.assertIn(b'id="canvas"', response.read())


@unittest.skipUnless((ROOT / "tools" / "rars.jar").is_file(), "Run python scripts/setup.py to prepare RARS")
class ActualAssemblyTests(unittest.TestCase):
    def test_all_ten_sample_scores_match_in_one_session(self) -> None:
        examples = json.loads((ROOT / "examples" / "digits.json").read_text(encoding="utf-8"))["samples"]
        with RiscvSession() as session:
            pid = session.pid
            results = session.predict_batch([bytes.fromhex(sample["hex"]) for sample in examples])
            self.assertEqual(session.pid, pid)
            for sample, result in zip(examples, results):
                self.assertEqual(list(result.scores), sample["expected_scores"])
                expected = max(range(10), key=lambda index: sample["expected_scores"][index])
                self.assertEqual(result.digit, expected)
        self.assertTrue(session.closed)
        self.assertIsNotNone(session._process.poll())


if __name__ == "__main__":
    unittest.main()
