"""Benchmark for the microphone noise filter (voicelink/denoise.py).

Synthesizes EN/ES sentences with Kokoro, mixes in fan-like noise (pink noise
plus mains hum) at several SNRs, and measures for each filter level:
  * word error rate of Parakeet on the result,
  * how often Silero VAD fires on the noise alone (false starts),
  * CPU cost of the filter.

    python tools/bench_denoise.py
"""
import os
import sys
import time
from difflib import SequenceMatcher

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from selftest import make_voice  # noqa: E402
from voicelink.config import load_config, model_path  # noqa: E402
from voicelink.denoise import Denoiser  # noqa: E402
from voicelink.stt import SpeechToText  # noqa: E402
from voicelink.textproc import normalized_words  # noqa: E402
from voicelink.vad import FRAME, SileroVAD  # noqa: E402

SENTENCES = [
    ("en-us", "am_michael", "Can you remind me to call the dentist tomorrow at nine in the morning?"),
    ("en-us", "af_heart", "The quarterly report shows a small increase in sales across every region."),
    ("en-us", "bm_george", "Please turn off the lights in the kitchen and lock the back door."),
    ("es", "em_alex", "Mañana tengo una reunión muy importante con el equipo de ventas."),
    ("es", "ef_dora", "¿Me puedes recomendar una receta sencilla para la cena de esta noche?"),
    ("es", "em_alex", "El tren sale de la estación central a las siete y cuarto."),
]
SNRS = (30, 10, 5, 0, -5)
LEVELS = ("off", "light", "strong")
LEAD_S = 2.0      # the filter hears the room before the user talks, as in real use


def fan_noise(n, rng):
    white = rng.standard_normal(n + 1024)
    f = np.fft.rfftfreq(len(white), 1 / 16000)
    shape = 1 / np.sqrt(np.maximum(f, 40))                      # pink
    shape *= 1 + 2.5 * np.exp(-((f - 180) / 60) ** 2)            # blade-pass hump
    x = np.fft.irfft(np.fft.rfft(white) * shape, len(white))[:n]
    t = np.arange(n) / 16000
    x = x / np.std(x) + 0.35 * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t + 1))
    return (x / np.std(x)).astype(np.float32)


def wer(ref, hyp):
    r, h = normalized_words(ref), normalized_words(hyp)
    sm = SequenceMatcher(None, r, h, autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return errors / max(1, len(r))


def main():
    cfg = load_config()
    say = make_voice()
    stt = SpeechToText(cfg["stt"])
    thr = cfg["vad"]["threshold"]
    rng = np.random.default_rng(7)
    speech = [(text, say(text, voice, lang).astype(np.float32) / 32767) for lang, voice, text in SENTENCES]
    lead = int(LEAD_S * 16000)

    print(f"{'SNR':>5} {'filter':>7} {'WER':>6} {'VAD false %':>12} {'ms per s':>9}")
    for snr in SNRS:
        for level in LEVELS:
            errs, cost, dur = [], 0.0, 0.0
            for text, clean in speech:
                noise = fan_noise(lead + len(clean) + 8000, rng)
                gain = np.sqrt(np.mean(clean ** 2)) / (10 ** (snr / 20))
                mix = noise * gain
                mix[lead:lead + len(clean)] += clean
                d = Denoiser(level)
                t = time.perf_counter()
                out = d.process(mix)
                cost += time.perf_counter() - t
                dur += len(mix) / 16000
                errs.append(wer(text, stt.transcribe(out[lead:])))
            # False starts: 20 s of noise alone at the same level.
            noise = fan_noise(20 * 16000, rng) * gain
            out = Denoiser(level).process(noise)[lead:]
            vad = SileroVAD(model_path("vad/silero_vad.onnx"))
            probs = [vad(out[i:i + FRAME]) for i in range(0, len(out) - FRAME, FRAME)]
            false = 100 * np.mean(np.array(probs) >= thr)
            print(f"{snr:>4}dB {level:>7} {np.mean(errs):>6.1%} {false:>11.1f}% {1000 * cost / dur:>8.1f}")
        print()


if __name__ == "__main__":
    main()
