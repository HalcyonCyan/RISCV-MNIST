"""Normalize handwritten images with Pillow; no neural-network math here."""
from __future__ import annotations

from statistics import median

from PIL import Image, ImageOps


def preprocess_image(image: Image.Image) -> bytes:
    """Return 784 uint8 pixels: light ink, zero background, centered at 13.5.

    Handles either foreground polarity and transparent PNGs. An empty or nearly
    uniform image returns an all-zero input. The crop fits inside a 20x20 box.
    """
    image = ImageOps.exif_transpose(image)
    if image.width == 0 or image.height == 0:
        return bytes(784)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        luminance = rgba.convert("L").tobytes()
        alpha = rgba.getchannel("A").tobytes()
        visible = [value for value, opacity in zip(luminance, alpha) if opacity > 32]
        background = 0 if visible and median(visible) > 127 else 255
        opaque = Image.new("RGBA", rgba.size, (background, background, background, 255))
        opaque.alpha_composite(rgba)
        image = opaque.convert("RGB")

    gray = list(image.convert("L").tobytes())
    width, height = image.size
    edges = gray[:width] + gray[-width:]
    edges.extend(gray[y * width] for y in range(height))
    edges.extend(gray[y * width + width - 1] for y in range(height))
    if median(edges) > 127:
        gray = [255 - value for value in gray]
        edges = [255 - value for value in edges]
    baseline = median(edges)
    foreground = [max(0.0, value - baseline) for value in gray]
    peak = max(foreground)
    if peak < 12:
        return bytes(784)
    threshold = max(12.0, peak * 0.08)
    left, right, top, bottom = width, -1, height, -1
    for index, value in enumerate(foreground):
        if value > threshold:
            x, y = index % width, index // width
            left, right = min(left, x), max(right, x)
            top, bottom = min(top, y), max(bottom, y)
    if right < left:
        return bytes(784)
    crop_width, crop_height = right - left + 1, bottom - top + 1
    crop = bytes(round(foreground[y * width + x] * 255.0 / peak)
                 if foreground[y * width + x] > threshold else 0
                 for y in range(top, bottom + 1) for x in range(left, right + 1))
    scale = 20.0 / max(crop_width, crop_height)
    target = (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
    resized = Image.frombytes("L", (crop_width, crop_height), crop).resize(
        target, Image.Resampling.LANCZOS)
    output = Image.new("L", (28, 28), 0)
    output.paste(resized, ((28 - target[0]) // 2, (28 - target[1]) // 2))
    pixels = output.tobytes()
    mass = sum(pixels)
    if not mass:
        return bytes(784)
    dx = round(13.5 - sum(value * (index % 28) for index, value in enumerate(pixels)) / mass)
    dy = round(13.5 - sum(value * (index // 28) for index, value in enumerate(pixels)) / mass)
    # Pillow clips translated ink at the border; it never wraps to the other side.
    centered = Image.new("L", (28, 28), 0)
    centered.paste(output, (dx, dy))
    return centered.tobytes()
