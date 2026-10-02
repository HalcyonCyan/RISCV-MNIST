"use strict";

(() => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const preview = document.getElementById("preview");
  const previewCtx = preview.getContext("2d");
  const recognize = document.getElementById("recognize");
  const clear = document.getElementById("clear");
  const sample = document.getElementById("sample");
  const sampleDigit = document.getElementById("sample-digit");
  const hint = document.getElementById("canvas-hint");
  const status = document.getElementById("status");
  const resultDigit = document.getElementById("result-digit");
  const resultState = document.getElementById("result-state");
  const elapsed = document.getElementById("elapsed");
  const connection = document.getElementById("connection");
  const connectionDot = document.getElementById("connection-dot");
  const classes = document.getElementById("classes");
  let hasInk = false;
  let busy = false;
  let originalPixels = null;
  let activePointer = null;
  let lastPoint = null;

  const rows = Array.from({ length: 10 }, (_, digit) => {
    const row = document.createElement("li");
    row.className = "class-row";
    const label = document.createElement("span");
    label.className = "class-label";
    label.textContent = String(digit);
    const track = document.createElement("span");
    track.className = "bar-track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    track.append(fill);
    const value = document.createElement("span");
    value.className = "class-value";
    value.textContent = "—";
    row.append(label, track, value);
    classes.append(row);
    return { row, fill, value };
  });

  function message(text, error = false) {
    status.textContent = text;
    status.classList.toggle("error", error);
  }

  function setBusy(value, label = "正在识别…") {
    busy = value;
    [recognize, clear, sample, sampleDigit].forEach((control) => { control.disabled = value; });
    document.body.classList.toggle("busy", value);
    canvas.setAttribute("aria-busy", String(value));
    if (value) { resultState.textContent = label; }
  }

  function resetResult() {
    resultDigit.textContent = "—";
    resultState.textContent = "等待识别";
    elapsed.textContent = "— ms";
    previewCtx.fillStyle = "black";
    previewCtx.fillRect(0, 0, 28, 28);
    rows.forEach(({ row, fill, value }) => {
      row.classList.remove("winner");
      row.removeAttribute("title");
      fill.style.width = "0%";
      value.textContent = "—";
    });
  }

  function clearCanvas() {
    if (busy) { return; }
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasInk = false;
    originalPixels = null;
    activePointer = null;
    hint.hidden = false;
    resetResult();
    resultState.textContent = "等待输入";
    message("画板已清空。写一个数字，按 Enter 识别。");
  }

  function paintPixels(targetContext, pixels) {
    const image = targetContext.createImageData(28, 28);
    pixels.forEach((value, i) => {
      const offset = i * 4;
      image.data[offset] = value;
      image.data[offset + 1] = value;
      image.data[offset + 2] = value;
      image.data[offset + 3] = 255;
    });
    targetContext.putImageData(image, 0, 0);
  }

  function validPixels(pixels) {
    return Array.isArray(pixels) && pixels.length === 784 && pixels.every((value) => Number.isInteger(value) && value >= 0 && value <= 255);
  }

  async function request(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 60000);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      let data;
      try { data = await response.json(); }
      catch { throw new Error("服务返回了无法读取的结果，请检查后端运行状态。"); }
      if (!response.ok) { throw new Error(data.error || data.message || `请求失败（${response.status}）。`); }
      return data;
    } catch (error) {
      if (error.name === "AbortError") { throw new Error("请求超时。请稍后重试，或查看后端终端。"); }
      if (error instanceof TypeError) { throw new Error("无法连接本地服务，请确认后端正在运行。"); }
      throw error;
    } finally { window.clearTimeout(timeout); }
  }

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height };
  }

  function stroke(from, to) {
    ctx.strokeStyle = "white";
    ctx.fillStyle = "white";
    ctx.lineWidth = 24;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(to.x, to.y, 12, 0, Math.PI * 2);
    ctx.fill();
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (busy || activePointer !== null || (event.pointerType === "mouse" && event.button !== 0)) { return; }
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    activePointer = event.pointerId;
    canvas.setPointerCapture(activePointer);
    originalPixels = null;
    hasInk = true;
    hint.hidden = true;
    resetResult();
    lastPoint = point(event);
    stroke(lastPoint, lastPoint);
    message("写好后点击「识别数字」，或按 Enter。");
  });

  canvas.addEventListener("pointermove", (event) => {
    if (busy || event.pointerId !== activePointer) { return; }
    const next = point(event);
    stroke(lastPoint, next);
    lastPoint = next;
  });

  function endStroke(event) {
    if (event.pointerId === activePointer) {
      if (canvas.hasPointerCapture(event.pointerId)) { canvas.releasePointerCapture(event.pointerId); }
      activePointer = null;
      lastPoint = null;
    }
  }
  canvas.addEventListener("pointerup", endStroke);
  canvas.addEventListener("pointercancel", endStroke);
  canvas.addEventListener("lostpointercapture", endStroke);

  async function loadSample() {
    if (busy) { return; }
    setBusy(true, "正在载入…");
    message("正在载入 MNIST 样本…");
    try {
      const data = await request(`/api/sample?digit=${encodeURIComponent(sampleDigit.value)}`);
      if (!validPixels(data.pixels)) { throw new Error("样本数据不完整，请重试。"); }
      const thumbnail = document.createElement("canvas");
      thumbnail.width = 28;
      thumbnail.height = 28;
      paintPixels(thumbnail.getContext("2d"), data.pixels);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(thumbnail, 0, 0, canvas.width, canvas.height);
      originalPixels = data.pixels.slice();
      hasInk = true;
      hint.hidden = true;
      resetResult();
      paintPixels(previewCtx, data.pixels);
      message(`已载入数字 ${data.digit} 的真实 MNIST 样本，点击「识别数字」。`);
    } catch (error) {
      resultState.textContent = "载入失败";
      message(error.message, true);
    } finally { setBusy(false); }
  }

  async function predict() {
    if (busy) { return; }
    if (!hasInk) { message("画板还是空的。请先写一个数字，或载入样本。", true); canvas.focus(); return; }
    setBusy(true);
    message("正在读取数字…");
    try {
      const body = originalPixels ? { pixels: originalPixels } : { image: canvas.toDataURL("image/png") };
      const data = await request("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!Number.isInteger(data.digit) || data.digit < 0 || data.digit > 9 || !validPixels(data.pixels) || !Array.isArray(data.scores) || data.scores.length !== 10 || !data.scores.every(Number.isFinite) || !Array.isArray(data.probabilities) || data.probabilities.length !== 10 || !data.probabilities.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) || !Number.isFinite(data.elapsed_ms) || data.elapsed_ms < 0) {
        throw new Error("识别结果不完整，请重试。");
      }
      resultDigit.textContent = String(data.digit);
      resultState.textContent = "识别完成";
      elapsed.textContent = `${data.elapsed_ms.toFixed(1)} ms`;
      paintPixels(previewCtx, data.pixels);
      rows.forEach(({ row, fill, value }, digit) => {
        const percentage = data.probabilities[digit] * 100;
        fill.style.width = `${percentage}%`;
        value.textContent = `${percentage.toFixed(1)}%`;
        row.classList.toggle("winner", digit === data.digit);
        row.title = `数字 ${digit}：原始分数 ${data.scores[digit]}`;
      });
      message(`识别结果：${data.digit}。可以继续书写，或清空后再试一次。`);
    } catch (error) {
      resultState.textContent = "识别失败";
      message(error.message, true);
    } finally { setBusy(false); }
  }

  recognize.addEventListener("click", predict);
  clear.addEventListener("click", clearCanvas);
  sample.addEventListener("click", loadSample);
  document.addEventListener("keydown", (event) => {
    if (busy || event.repeat || event.ctrlKey || event.altKey || event.metaKey || event.target instanceof HTMLSelectElement) { return; }
    if (event.key === "Escape") { event.preventDefault(); clearCanvas(); }
    if (event.key === "Enter" && !(event.target instanceof HTMLButtonElement) && !(event.target instanceof HTMLAnchorElement)) { event.preventDefault(); predict(); }
  });

  clearCanvas();
  message("在画板上写一个数字，或试一个 MNIST 样本。");
  request("/api/health").then((data) => {
    if (!data.ready) { throw new Error("服务尚未准备好，请查看后端终端。"); }
    connection.textContent = "本地服务就绪";
    connectionDot.classList.add("ready");
  }).catch((error) => {
    connection.textContent = "连接未就绪";
    connectionDot.classList.add("error");
    message(error.message, true);
  });
})();
