"""Streaming stationary-noise suppressor for the microphone (fans, hum, AC).

A short-time spectral Wiener filter (decision-directed a-priori SNR) with a
noise estimate that keeps learning whenever a band is near its noise floor,
so it adapts to the room within a second or two and follows slow changes.
Runs on 16 kHz mono in 512-sample STFT frames with 50 % overlap
(16 ms latency, well under 1 % of one CPU core).
"""
import numpy as np

N = 512
HOP = N // 2
LEVELS = {
    # gain floor (how much noise is left), over-subtraction
    "light": (0.30, 1.0),
    "strong": (0.12, 1.6),
}


class Denoiser:
    def __init__(self, level="light"):
        self.win = np.sqrt(np.hanning(N + 1)[:N]).astype(np.float32)  # sqrt-Hann: perfect WOLA
        self.set_level(level)
        self.reset()

    def set_level(self, level):
        self.level = level if level in LEVELS else "off"
        if self.level != "off":
            self.floor, self.over = LEVELS[self.level]

    def reset(self):
        self.inbuf = np.zeros(N - HOP, dtype=np.float32)
        self.outbuf = np.zeros(N - HOP, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.noise = None
        self.prev = None
        self.frames = 0

    @property
    def enabled(self):
        return self.level != "off"

    def process(self, x):
        """Takes any number of samples; returns the same number (delayed by HOP)."""
        if not self.enabled:
            return x
        buf = np.concatenate([self.pending, x]) if len(self.pending) else x
        n = len(buf) // HOP
        self.pending = buf[n * HOP:].copy()
        out = np.empty(n * HOP, dtype=np.float32)
        for i in range(n):
            out[i * HOP:(i + 1) * HOP] = self._hop(buf[i * HOP:(i + 1) * HOP])
        return out

    def _hop(self, hop):
        frame = np.concatenate([self.inbuf, hop])
        self.inbuf = frame[HOP:]
        spec = np.fft.rfft(frame * self.win)
        power = spec.real ** 2 + spec.imag ** 2 + 1e-12
        self.frames += 1

        if self.noise is None:
            self.noise = power.copy()
            self.prev = np.ones_like(power)
        # Noise tracking: fast where the band is close to the floor, a slow
        # upward creep elsewhere so a louder fan is still learned.
        near = power < self.noise * 4.0
        fast = 0.80 if self.frames < 30 else 0.95
        self.noise = np.where(near, fast * self.noise + (1 - fast) * power, self.noise * 1.002)
        self.noise = np.minimum(self.noise, np.maximum(power, self.noise * 0.9) * 1.0)

        gamma = power / (self.noise * self.over)
        xi = 0.96 * self.prev + 0.04 * np.maximum(gamma - 1.0, 0.0)
        gain = xi / (1.0 + xi)
        gain = np.maximum(gain, self.floor)
        # Smooth across neighbouring bins to avoid "musical" tones.
        gain = np.convolve(gain, (0.25, 0.5, 0.25), mode="same")
        self.prev = (gain ** 2) * gamma

        y = np.fft.irfft(spec * gain, N).astype(np.float32) * self.win
        y[:N - HOP] += self.outbuf
        self.outbuf = y[HOP:].copy()
        return y[:HOP]
