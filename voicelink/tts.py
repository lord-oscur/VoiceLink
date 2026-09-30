"""Text-to-speech with Kokoro-82M (ONNX). 24 kHz mono float32 output."""
import logging
import time

import numpy as np
import onnxruntime as ort

from .config import model_path
from .gpu import providers_for

log = logging.getLogger("tts")

SAMPLE_RATE = 24000

# Kokoro voice ids start with a letter that encodes their language.
_VOICE_LANG = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
    "i": "it", "p": "pt-br", "j": "ja", "z": "cmn",
}
_VOICE_ISO = {"a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi", "i": "it", "p": "pt", "j": "ja", "z": "zh"}


def voice_language(voice):
    return _VOICE_ISO.get(voice[:1], "en")


class TextToSpeech:
    def __init__(self, cfg):
        from kokoro_onnx import Kokoro

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg["cpu_threads"])
        so.log_severity_level = 3
        self.providers = providers_for(cfg["device"], cudnn_heuristic=True)
        t = time.perf_counter()
        sess = ort.InferenceSession(model_path(cfg["model_file"]), sess_options=so, providers=self.providers)
        self.kokoro = Kokoro.from_session(sess, model_path(cfg["voices_file"]))
        self.voices = self.kokoro.get_voices()
        self.voice_map = {k: v for k, v in cfg["voices"].items() if v in self.voices}
        self.speed = float(cfg["speed"])
        self.synthesize("Ready.", "en")
        log.info("Kokoro ready on %s in %.1fs (%d voices)", sess.get_providers()[0],
                 time.perf_counter() - t, len(self.voices))

    def pick_voice(self, lang, override=None):
        if override and override in self.voices:
            return override
        return self.voice_map.get(lang) or self.voice_map.get("en") or "af_heart"

    def synthesize(self, text, lang="en", voice=None, speed=None):
        """Returns float32 audio at SAMPLE_RATE (empty array if nothing to say)."""
        voice = self.pick_voice(lang, voice)
        espeak_lang = _VOICE_LANG.get(voice[:1], "en-us")
        try:
            audio, _ = self.kokoro.create(text, voice=voice, speed=speed or self.speed, lang=espeak_lang)
        except ValueError as e:
            log.debug("TTS skipped %r: %s", text, e)
            return np.zeros(0, dtype=np.float32)
        return np.asarray(audio, dtype=np.float32)
