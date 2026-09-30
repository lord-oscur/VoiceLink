"""Speaker verification for the "Only my voice" microphone filter.

WeSpeaker ResNet34 (VoxCeleb) through sherpa-onnx gives a 256-number voice
print per utterance; an utterance is accepted when its cosine similarity to
the enrolled print is high enough. About 20 ms of CPU per second of audio.
"""
import logging
import threading

import numpy as np

from .config import model_path

log = logging.getLogger("speaker")

MODEL = "speaker/wespeaker_resnet34.onnx"


def threshold(strictness):
    """Strictness 0..100 -> cosine threshold (50 -> 0.80)."""
    s = min(100.0, max(0.0, float(strictness)))
    return 0.62 + 0.0036 * s


class SpeakerID:
    def __init__(self):
        self.lock = threading.Lock()
        self.ex = None
        self.error = None

    def available(self):
        import os

        return os.path.exists(model_path(MODEL))

    def _load(self):
        if self.ex is None:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path(MODEL), num_threads=1)
            self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            log.info("Speaker verification model loaded (%d-d)", self.ex.dim)
        return self.ex

    def embed(self, audio):
        """float32 16 kHz mono -> unit-length voice print (np.ndarray)."""
        with self.lock:
            ex = self._load()
            s = ex.create_stream()
            s.accept_waveform(16000, np.ascontiguousarray(audio, dtype=np.float32))
            s.input_finished()
            e = np.asarray(ex.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def enroll(self, audio):
        """Averages prints of ~3 s pieces for a steadier reference."""
        piece = 3 * 16000
        parts = [audio[i:i + piece] for i in range(0, len(audio) - 16000, piece)] or [audio]
        prints = np.stack([self.embed(p) for p in parts])
        e = prints.mean(axis=0)
        return e / (np.linalg.norm(e) + 1e-9)

    @staticmethod
    def similarity(a, b):
        return float(np.dot(a, b))
