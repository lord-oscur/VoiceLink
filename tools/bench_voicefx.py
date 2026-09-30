"""Checks the voice-effects chain: speed and resulting pitch of every preset.

    .venv\\Scripts\\python tools\\bench_voicefx.py [voice ...]

Writes one WAV per preset to data/fx_samples/ so they can be listened to.
"""
import os
import sys
import time
import wave

import numpy as np
import parselmouth

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voicelink import voicefx  # noqa: E402
from voicelink.config import ROOT, load_config  # noqa: E402
from voicelink.tts import SAMPLE_RATE, TextToSpeech, voice_language  # noqa: E402

TEXT = {"en": "I have been waiting for this moment all day, and now it is finally here.",
        "es": "He estado esperando este momento todo el día, y por fin ha llegado."}


def f0(audio):
    pitch = parselmouth.Sound(audio.astype(np.float64), SAMPLE_RATE).to_pitch_ac(0.01, 60, 600)
    v = pitch.selected_array["frequency"]
    v = v[v > 0]
    return float(np.median(v)) if len(v) else 0.0


def save(path, audio):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    voices = sys.argv[1:] or ["af_heart", "am_michael", "ef_dora", "em_alex"]
    out_dir = os.path.join(ROOT, "data", "fx_samples")
    os.makedirs(out_dir, exist_ok=True)
    tts = TextToSpeech(load_config()["tts"])
    cases = [("age", a) for a in voicefx.AGES if a not in ("none", "adult")] + \
            [("emotion", e) for e in voicefx.EMOTIONS if e != "none"]
    for voice in voices:
        lang = voice_language(voice)
        base = tts.synthesize(TEXT.get(lang, TEXT["en"]), lang, voice, 1.0)
        f_base = f0(base)
        save(os.path.join(out_dir, f"{voice}_0_base.wav"), base)
        print(f"\n{voice}: base {f_base:.0f} Hz, {len(base) / SAMPLE_RATE:.2f} s")
        for kind, name in cases:
            fx = {kind: name, "intensity": 100}
            t = time.perf_counter()
            audio = tts.speak(TEXT.get(lang, TEXT["en"]), lang, voice, 1.0, None, fx)
            ms = (time.perf_counter() - t) * 1000
            dur = len(audio) / SAMPLE_RATE
            fv = f0(audio)
            peak = float(np.max(np.abs(audio)))
            print(f"  {kind:7} {name:8} {fv:5.0f} Hz ({12 * np.log2(max(fv, 1) / f_base):+5.1f} st)  "
                  f"{dur:.2f} s  peak {peak:.2f}  total {ms:4.0f} ms")
            save(os.path.join(out_dir, f"{voice}_{kind}_{name}.wav"), audio)
    # DSP cost alone (no synthesis) for a 4 s sentence.
    audio = tts.synthesize(TEXT["en"], "en", "af_heart", 1.0)
    p = voicefx.plan({"pitch": 3, "timbre": 30, "age": "senior", "emotion": "sad", "intensity": 100}, "female", 187)
    t = time.perf_counter()
    for _ in range(5):
        voicefx.apply(audio, SAMPLE_RATE, p)
    print(f"\nDSP only (pitch+timbre+senior+sad, {len(audio) / SAMPLE_RATE:.1f} s): "
          f"{(time.perf_counter() - t) / 5 * 1000:.0f} ms")
    print("Samples in", out_dir)


if __name__ == "__main__":
    main()
