"""Keep the actual JVM, RARS program, and model alive between predictions."""
from __future__ import annotations

import os
import shutil
import math
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TextIO

from .paths import ROOT
from .model import validate_model


@dataclass(frozen=True)
class Prediction:
    digit: int
    scores: tuple[int, ...]
    elapsed_seconds: float


@lru_cache(maxsize=1)
def resolve_java() -> str:
    """Resolve JAVA_HOME or PATH and bypass Windows launcher indirection."""
    executable = "java.exe" if os.name == "nt" else "java"
    java_home = os.environ.get("JAVA_HOME")
    configured = Path(java_home) / "bin" / executable if java_home else None
    java = str(configured) if configured is not None and configured.is_file() else shutil.which("java")
    if java is None:
        raise RuntimeError("Java is missing. Install Java 8+ and set JAVA_HOME or add Java to PATH.")
    try:
        settings = subprocess.run(
            [java, "-XshowSettings:properties", "-version"], capture_output=True,
            stdin=subprocess.DEVNULL, text=True, errors="replace", timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Could not query Java. Check JAVA_HOME or PATH.") from exc
    if settings.returncode:
        raise RuntimeError("Java installation failed its version check")
    for line in (settings.stdout + settings.stderr).splitlines():
        name, separator, value = line.strip().partition(" = ")
        if separator and name == "java.home":
            actual = Path(value) / "bin" / executable
            if actual.is_file():
                return str(actual)
    return java


@dataclass(frozen=True)
class _ReaderEnd:
    error: str = ""


class RiscvSession:
    """One serialized, persistent RV32IM interpreter; never infer in Python.

    ``startup_seconds`` measures initialization through the READY response.
    Prediction timings measure individual requests after that initialization.
    Failed or timed-out sessions close their child and must be recreated.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        source_path: Path | None = None,
        timeout: float = 30,
        *,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self.timeout = float(timeout)
        self.startup_seconds = 0.0
        self._lock = threading.RLock()
        self._closed = False
        self._cancel_event = cancel_event
        self._reader_stop = threading.Event()
        self._stdout_overflow = threading.Event()
        self._stdout: queue.Queue[str | _ReaderEnd] = queue.Queue(maxsize=64)
        self._stderr: deque[str] = deque(maxlen=32)
        self._stderr_lock = threading.Lock()
        self._readers: list[threading.Thread] = []
        self._process: subprocess.Popen[str] | None = None
        started = time.perf_counter()
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("RISC-V session startup cancelled")
            jar = ROOT / "tools" / "rars.jar"
            model = Path(model_path) if model_path is not None else ROOT / "models" / "model.bin"
            source = Path(source_path) if source_path is not None else ROOT / "asm" / "mnist.s"
            for path in (jar, model, source):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing project file: {path}")
            validate_model(model)
            java = resolve_java()
            runtime_dir = ROOT / ".runtime"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            command = [
                java, "-Xmx512m", "-Djava.awt.headless=true",
                f"-Djava.io.tmpdir={runtime_dir}",
                f"-Djava.util.prefs.userRoot={runtime_dir / 'java-prefs'}",
                "-jar", str(jar), "nc", "me", "ae2", "se3", "sm",
                str(source.resolve()), "pa", str(model.resolve()), "-",
            ]
            self._process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert self._process.stdout is not None and self._process.stderr is not None
            for name, target, stream in (
                ("stdout", self._read_stdout, self._process.stdout),
                ("stderr", self._read_stderr, self._process.stderr),
            ):
                reader = threading.Thread(
                    target=target, args=(stream,), name=f"riscv-mnist-{self.pid}-{name}", daemon=True,
                )
                self._readers.append(reader)
                reader.start()
            ready = self._next_line(started + self.timeout, "startup")
            if ready != "READY":
                raise RuntimeError(self._with_stderr(f"Expected RARS READY, got: {ready}"))
            self.startup_seconds = time.perf_counter() - started
        except BaseException:
            self._shutdown(graceful=False)
            raise

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def closed(self) -> bool:
        return self._closed

    def _enqueue_stdout(self, value: str | _ReaderEnd) -> bool:
        try:
            self._stdout.put_nowait(value)
            return True
        except queue.Full:
            self._stdout_overflow.set()
            return False

    def _read_stdout(self, stream: TextIO) -> None:
        try:
            while not self._reader_stop.is_set():
                line = stream.readline(4097)
                if not line:
                    self._enqueue_stdout(_ReaderEnd())
                    return
                if len(line) > 4096:
                    self._enqueue_stdout(_ReaderEnd("RARS stdout line exceeded 4096 characters"))
                    return
                if not self._enqueue_stdout(line.rstrip("\r\n")):
                    return
        except (OSError, ValueError) as exc:
            self._enqueue_stdout(_ReaderEnd(str(exc)))

    def _read_stderr(self, stream: TextIO) -> None:
        try:
            while not self._reader_stop.is_set():
                # Both the number and size of stored chunks are bounded.
                chunk = stream.readline(1024)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr.append(chunk.rstrip("\r\n"))
        except (OSError, ValueError):
            return

    def _with_stderr(self, message: str) -> str:
        with self._stderr_lock:
            details = "\n".join(self._stderr).strip()
        return f"{message}\n{details}" if details else message

    def _next_line(self, deadline: float, phase: str) -> str:
        while True:
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise RuntimeError(f"RISC-V session {phase} cancelled")
            if self._stdout_overflow.is_set():
                raise RuntimeError(self._with_stderr("RARS produced too many unsolicited output lines"))
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise RuntimeError(self._with_stderr(f"RISC-V {phase} exceeded {self.timeout:g} seconds"))
            try:
                item = self._stdout.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            if isinstance(item, _ReaderEnd):
                assert self._process is not None
                code = self._process.poll()
                detail = item.error or f"RARS closed stdout (exit code {code})"
                raise RuntimeError(self._with_stderr(detail))
            return item

    @staticmethod
    def _validate_image(pixels: bytes) -> None:
        if not isinstance(pixels, bytes) or len(pixels) != 784:
            raise ValueError("Each image must be exactly 784 bytes")

    def _ensure_open(self) -> subprocess.Popen[str]:
        if self._closed or self._process is None:
            raise RuntimeError("RISC-V session is closed")
        if self._process.poll() is not None:
            code = self._process.returncode
            self._shutdown(graceful=False)
            raise RuntimeError(self._with_stderr(f"RARS exited with code {code}"))
        return self._process

    def _predict_locked(self, pixels: bytes) -> Prediction:
        process = self._ensure_open()
        started = time.perf_counter()
        try:
            assert process.stdin is not None
            process.stdin.write(pixels.hex() + "\n")
            process.stdin.flush()
            line = self._next_line(started + self.timeout, "inference")
            parts = line.split()
            if len(parts) != 12 or parts[0] != "PRED":
                raise RuntimeError(self._with_stderr(f"Unexpected RARS output: {line}"))
            try:
                digit, *scores = [int(value) for value in parts[1:]]
            except ValueError as exc:
                raise RuntimeError(f"Invalid RARS prediction: {line}") from exc
            if not 0 <= digit <= 9 or any(not -(2**31) <= score < 2**31 for score in scores):
                raise RuntimeError("RARS returned an invalid digit or score")
            if digit != max(range(10), key=lambda index: scores[index]):
                raise RuntimeError("RARS digit does not match its output scores")
            return Prediction(digit, tuple(scores), time.perf_counter() - started)
        except BaseException as exc:
            self._shutdown(graceful=False)
            if isinstance(exc, (BrokenPipeError, OSError, ValueError)):
                raise RuntimeError(self._with_stderr(f"Could not exchange data with RARS: {exc}")) from exc
            raise

    def predict(self, pixels: bytes) -> Prediction:
        self._validate_image(pixels)
        with self._lock:
            return self._predict_locked(pixels)

    def predict_batch(self, images: list[bytes]) -> list[Prediction]:
        if not images:
            raise ValueError("At least one 28x28 image is required")
        for pixels in images:
            self._validate_image(pixels)
        with self._lock:
            return [self._predict_locked(pixels) for pixels in images]

    def _shutdown(self, *, graceful: bool) -> None:
        self._closed = True
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            if graceful and process.stdin is not None:
                try:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                except (OSError, ValueError):
                    pass
            else:
                process.terminate()
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self._reader_stop.set()
        for reader in self._readers:
            if reader.is_alive():
                reader.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._shutdown(graceful=True)

    def __enter__(self) -> RiscvSession:
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

