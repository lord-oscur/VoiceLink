"""Text-to-speech with Kokoro-82M (ONNX). 24 kHz mono float32 output."""
import json
import logging
import os
import time

import numpy as np
import onnxruntime as ort

from .config import model_path
from .gpu import providers_for

_META_FILE = os.path.join(os.path.dirname(__file__), "voices_meta.json")

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
        self.male_map = {k: v for k, v in cfg.get("voices_male", {}).items() if v in self.voices}
        self.speed = float(cfg["speed"])
        self.catalog = self._catalog()
        self.meta = {v["id"]: v for v in self.catalog}
        self.synthesize("Ready.", "en")
        log.info("Kokoro ready on %s in %.1fs (%d voices)", sess.get_providers()[0],
                 time.perf_counter() - t, len(self.voices))

    def _catalog(self):
        """Voice list for the UI: language, accent, gender and approximate sounding age."""
        meta = {}
        if os.path.exists(_META_FILE):
            with open(_META_FILE, encoding="utf-8") as f:
                meta = json.load(f)
        defaults = set(self.voice_map.values()) | set(self.male_map.values())
        out = []
        for v in self.voices:
            m = meta.get(v, {})
            out.append({
                "id": v, "name": v.split("_", 1)[-1].replace("_", " ").title(),
                "group": v[0], "lang": voice_language(v),
                "gender": m.get("gender") or ("female" if v[1:2] == "f" else "male"),
                "age": m.get("age"), "f0": m.get("f0"), "default": v in defaults,
            })
        return out

    def pick_voice(self, lang, override=None, gender=None):
        if override and override in self.voices:
            return override
        if gender == "male":
            voice = self.male_map.get(lang)
            if voice:
                return voice
        elif gender == "female" or gender is None:
            voice = self.voice_map.get(lang)
            if voice and (gender is None or voice[1] == "f"):
                return voice
        return self.voice_map.get(lang) or self.voice_map.get("en") or "af_heart"

    def speak(self, text, lang="en", voice=None, speed=None, gender=None, fx=None):
        """synthesize() plus the voice effects from the UI (see voicefx.py)."""
        from . import voicefx

        voice = self.pick_voice(lang, voice, gender)
        meta = self.meta.get(voice, {})
        p = voicefx.plan(fx, meta.get("gender", "female"), meta.get("f0"))
        if p is None:
            return self.synthesize(text, lang, voice, speed)
        rate = min(2.0, max(0.5, (speed or self.speed) * p["rate"]))
        audio = self.synthesize(text, lang, voice, rate)
        return voicefx.apply(audio, SAMPLE_RATE, p)

    def synthesize(self, text, lang="en", voice=None, speed=None, gender=None):
        """Returns float32 audio at SAMPLE_RATE (empty array if nothing to say)."""
        voice = self.pick_voice(lang, voice, gender)
        espeak_lang = _VOICE_LANG.get(voice[:1], "en-us")
        try:
            audio, _ = self.kokoro.create(text, voice=voice, speed=speed or self.speed, lang=espeak_lang)
        except ValueError as e:
            log.debug("TTS skipped %r: %s", text, e)
            return np.zeros(0, dtype=np.float32)
        return np.asarray(audio, dtype=np.float32)
