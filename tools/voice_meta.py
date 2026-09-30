"""Measures every Kokoro voice and writes voicelink/voices_meta.json.

For each voice, a neutral sentence in its language is synthesized and its
median pitch measured. Within one gender, a lower median pitch is heard as
older, so the pitch gives an approximate "sounding age" order for the UI.
A few character voices get a fixed age instead (the Santas).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voicelink.config import load_config  # noqa: E402
from voicelink.tts import SAMPLE_RATE, TextToSpeech, voice_language  # noqa: E402

import parselmouth  # noqa: E402
from parselmouth.praat import call  # noqa: E402

SENTENCES = {
    "en": "Hello there, this is how my voice sounds when I read a normal sentence.",
    "es": "Hola, así es como suena mi voz cuando leo una frase normal.",
    "fr": "Bonjour, voici comment sonne ma voix quand je lis une phrase normale.",
    "it": "Ciao, questo è il suono della mia voce quando leggo una frase normale.",
    "pt": "Olá, é assim que a minha voz soa quando leio uma frase normal.",
    "hi": "नमस्ते, जब मैं एक सामान्य वाक्य पढ़ती हूँ तो मेरी आवाज़ ऐसी लगती है।",
    "ja": "こんにちは、これは普通の文を読むときの私の声です。",
    "zh": "你好，这是我读一个普通句子时的声音。",
}
FIXED_AGE = {"am_santa": 68, "em_santa": 68, "pm_santa": 68}


def median_f0(audio):
    s = parselmouth.Sound(audio.astype(np.float64), SAMPLE_RATE)
    pitch = s.to_pitch_ac(0.01, 60, 500)
    return call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")


def measure():
    tts = TextToSpeech(load_config()["tts"])
    meta = {}
    for v in sorted(tts.voices):
        lang = voice_language(v)
        audio = tts.synthesize(SENTENCES[lang], lang, v, 1.0)
        f0 = median_f0(audio)
        meta[v] = {"lang": lang, "gender": "female" if v[1] == "f" else "male", "f0": round(float(f0), 1)}
        print(f"{v:16} {lang} {meta[v]['gender']:6} {f0:6.1f} Hz")
    return meta


def main():
    path = os.path.join(ROOT, "voicelink", "voices_meta.json")
    if os.path.exists(path) and "--remeasure" not in sys.argv:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = measure()
    # Approximate sounding age from pitch within each gender: the typical
    # range maps to 22-45; clear outliers (a "male" voice at 290 Hz) go lower.
    for gender in ("female", "male"):
        voices = [v for v in meta if meta[v]["gender"] == gender and v not in FIXED_AGE]
        f0s = np.array([meta[v]["f0"] for v in voices])
        lo, hi = np.percentile(f0s, 10), np.percentile(f0s, 90)
        for v in voices:
            x = (meta[v]["f0"] - lo) / max(hi - lo, 1)
            meta[v]["age"] = int(np.clip(round(45 - 23 * x), 10, 55))
    for v, age in FIXED_AGE.items():
        if v in meta:
            meta[v]["age"] = age
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("wrote voicelink/voices_meta.json")


if __name__ == "__main__":
    main()
