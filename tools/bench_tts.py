"""Benchmark Kokoro TTS variants.

    python tools/bench_tts.py cpu:4:kokoro-v1.0.onnx cpu:8:kokoro-v1.0.int8.onnx cuda:4:kokoro-v1.0.onnx
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voicelink.config import model_path  # noqa: E402

import onnxruntime as ort  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402

try:
    ort.preload_dlls()
except Exception:
    pass

TEXTS = [
    ("Sure!", "af_heart", "en-us"),
    ("Sure, here is a quick idea.", "af_heart", "en-us"),
    ("That is a great question, and honestly the short answer is that it depends on how much time you have.",
     "af_heart", "en-us"),
    ("Claro, te cuento una idea rápida para empezar hoy mismo.", "ef_dora", "es"),
]

for spec in sys.argv[1:] or ["cpu:4:kokoro-v1.0.onnx"]:
    dev, threads, fname = spec.split(":")
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    providers = ([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
                 if dev == "cuda" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(model_path("kokoro/" + fname), sess_options=so, providers=providers)
    k = Kokoro.from_session(sess, model_path("kokoro/voices-v1.0.bin"))
    k.create("Warm up sentence number one.", voice="af_heart", lang="en-us")
    t = time.perf_counter()
    for _ in range(5):
        k.tokenizer.phonemize(TEXTS[2][0], "en-us")
    print(f"{spec}: phonemize long sentence {(time.perf_counter()-t)/5*1000:.0f} ms")
    for text, voice, lang in TEXTS:
        best = 1e9
        for _ in range(2):
            t = time.perf_counter()
            audio, sr = k.create(text, voice=voice, lang=lang, speed=1.05)
            best = min(best, time.perf_counter() - t)
        print(f"  {text[:34]!r:38} synth {best*1000:6.0f} ms  audio {len(audio)/sr:4.1f}s  RTF {best/(len(audio)/sr):.3f}")
