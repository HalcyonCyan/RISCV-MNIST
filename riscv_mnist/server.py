"""Small local HTTP adapter; all predictions come from the RISC-V process."""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from PIL import Image, UnidentifiedImageError

from .paths import ROOT
from .preprocess import preprocess_image
from .runtime import RiscvSession

MAX_BODY = 512 * 1024
WEB_FILES = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}
MIME_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}


def decode_input(payload: object) -> bytes:
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("请求必须包含 image 或 pixels 中的一项。")
    if "pixels" in payload:
        values = payload["pixels"]
        if not isinstance(values, list) or len(values) != 784:
            raise ValueError("pixels 必须包含 784 个灰度值。")
        if any(type(value) is not int or not 0 <= value <= 255 for value in values):
            raise ValueError("灰度值必须是 0～255 的整数。")
        return bytes(values)
    data = payload.get("image")
    prefix = "data:image/png;base64,"
    if not isinstance(data, str) or not data.startswith(prefix) or len(data) > MAX_BODY:
        raise ValueError("image 必须是 PNG 画布的 data URL。")
    try:
        raw = base64.b64decode(data[len(prefix):], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or not 0 < image.width <= 1024 or not 0 < image.height <= 1024:
                raise ValueError("PNG 尺寸必须在 1～1024 像素之间。")
            image.load()
            return preprocess_image(image)
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取 PNG 图像。") from exc


class Application:
    def __init__(self, session: RiscvSession) -> None:
        self.session = session
        metadata = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
        self.scale = float(metadata["logit_scale"])
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("模型的 logit_scale 参数无效。")
        examples = json.loads((ROOT / "examples" / "digits.json").read_text(encoding="utf-8"))
        self.samples = {sample["digit"]: sample for sample in examples["samples"]}

    def predict(self, payload: object) -> dict[str, object]:
        pixels = decode_input(payload)
        if not any(pixels):
            raise ValueError("画板为空，请先写一个数字。")
        result = self.session.predict(pixels)
        peak = max(result.scores)
        weights = [math.exp((score - peak) * self.scale) for score in result.scores]
        total = sum(weights)
        return {"digit": result.digit, "scores": list(result.scores),
                "probabilities": [weight / total for weight in weights],
                "elapsed_ms": round(result.elapsed_seconds * 1000, 3), "pixels": list(pixels)}


def make_handler(application: Application) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RiscvMnist/1.0"

        def log_message(self, format: str, *args: object) -> None:
            # Avoid logging client-controlled paths; show useful errors in the page.
            pass

        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json_response(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.respond(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            route = urlsplit(self.path)
            if route.path == "/api/health":
                self.json_response(200, {"ready": not application.session.closed, "architecture": [784, 64, 10]})
            elif route.path == "/api/sample":
                try:
                    query = parse_qs(route.query)
                    values = query.get("digit", ["7"])
                    if len(values) != 1 or values[0] not in "0123456789" or len(values[0]) != 1:
                        raise ValueError("digit 必须是 0～9。")
                    sample = application.samples[int(values[0])]
                    self.json_response(200, {"digit": sample["digit"], "index": sample["index"],
                                             "pixels": list(bytes.fromhex(sample["hex"]))})
                except (ValueError, KeyError) as exc:
                    self.json_response(400, {"error": str(exc)})
            elif route.path in WEB_FILES:
                path = ROOT / "web" / WEB_FILES[route.path]
                self.respond(200, path.read_bytes(), MIME_TYPES[path.suffix])
            else:
                self.json_response(404, {"error": "页面或接口不存在。"})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/predict":
                self.json_response(404, {"error": "接口不存在。"})
                return
            if self.headers.get_content_type() != "application/json":
                self.json_response(415, {"error": "请发送 application/json。"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self.json_response(413, {"error": "请求为空或超过 512 KiB。"})
                    return
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("请求数据不完整。")
                payload = json.loads(raw)
                self.json_response(200, application.predict(payload))
            except (ValueError, UnicodeError) as exc:
                self.json_response(400, {"error": str(exc)})
            except TimeoutError:
                self.json_response(408, {"error": "请求读取超时。"})
            except (OSError, RuntimeError) as exc:
                self.json_response(503, {"error": f"识别进程异常，请重新启动程序：{exc}"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="RISC-V MNIST browser demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    try:
        print("Preparing RISC-V inference...", flush=True)
        with RiscvSession() as session:
            application = Application(session)
            with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(application)) as server:
                url = f"http://127.0.0.1:{args.port}"
                print(f"Ready: {url} (Ctrl+C to stop)", flush=True)
                if not args.no_browser:
                    threading.Timer(0.2, webbrowser.open, args=(url,)).start()
                try:
                    server.serve_forever(poll_interval=0.1)
                except KeyboardInterrupt:
                    print("\nStopping RISC-V inference...", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\nRun python scripts/setup.py first.\n")


if __name__ == "__main__":
    main()
