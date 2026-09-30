"""Draws web/favicon.ico (a glowing orb with sound bars) without image libraries."""
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(n):
    ss = 4  # supersampling
    m = n * ss
    y, x = (np.mgrid[0:m, 0:m] + 0.5) / m * 2 - 1  # -1..1, y down
    r = np.hypot(x, y)
    rgba = np.zeros((m, m, 4))
    # Disc with a violet → blue radial gradient and a soft highlight.
    disc = r <= 0.94
    t = np.clip(np.hypot(x + 0.35, y + 0.4) / 1.6, 0, 1)[..., None]
    col = (1 - t) * np.array([0.72, 0.55, 1.0]) + t * np.array([0.28, 0.20, 0.78])
    rgba[..., :3] = col
    rgba[..., 3] = disc
    # Five white sound bars.
    heights = [0.28, 0.52, 0.72, 0.52, 0.28]
    for i, h in enumerate(heights):
        cx = (i - 2) * 0.27
        bar = (np.abs(x - cx) <= 0.085) & (np.abs(y) <= h)
        cap = np.hypot(x - cx, np.abs(y) - h) <= 0.085
        rgba[(bar | cap) & disc, :3] = 1.0
    img = rgba.reshape(n, ss, n, ss, 4).mean(axis=(1, 3))
    img[..., :3] /= np.maximum(img[..., 3:], 1e-6)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def ico(sizes):
    images = []
    for n in sizes:
        px = render(n)
        bgra = px[::-1, :, [2, 1, 0, 3]].tobytes()  # bottom-up BGRA
        mask_row = ((n + 31) // 32) * 4
        header = struct.pack("<IiiHHIIiiII", 40, n, n * 2, 1, 32, 0, len(bgra), 0, 0, 0, 0)
        images.append(header + bgra + b"\0" * mask_row * n)
    out = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    for n, data in zip(sizes, images):
        out += struct.pack("<BBBBHHII", n % 256, n % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return out + b"".join(images)


if __name__ == "__main__":
    path = os.path.join(ROOT, "web", "favicon.ico")
    with open(path, "wb") as f:
        f.write(ico([16, 32, 48, 64]))
    print("wrote", path)
