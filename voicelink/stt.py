"""Speech-to-text with NVIDIA Parakeet TDT 0.6B v3 (25 European languages,
punctuated output) running on ONNX Runtime through onnx-asr."""
import logging
import time

import numpy as np
import onnxruntime as ort

from .config import model_path
from .gpu import providers_for

log = logging.getLogger("stt")


class SpeechToText:
    def __init__(self, cfg):
        import onnx_asr

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg["cpu_threads"])
        so.log_severity_level = 3
        self.providers = providers_for(cfg["device"])
        t = time.perf_counter()
        self.model = onnx_asr.load_model(
            cfg["model"], model_path(cfg["model_dir"]),
            quantization=cfg["quantization"] or None,
            providers=self.providers, sess_options=so,
        )
        self.model.recognize(np.zeros(16000, dtype=np.float32))
        device = self.providers[0] if isinstance(self.providers[0], str) else self.providers[0][0]
        log.info("Parakeet ready on %s in %.1fs", device, time.perf_counter() - t)

    def transcribe(self, audio):
        """audio: float32 mono 16 kHz. Returns cleaned text (may be empty)."""
        if len(audio) < 16000 * 0.25:
            audio = np.pad(audio, (0, int(16000 * 0.25) - len(audio)))
        text = self.model.recognize(audio.astype(np.float32, copy=False))
        return " ".join(str(text).split())
