"""Benchmark Parakeet TDT 0.6B v3 (onnx-asr) on CPU / GPU."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voicelink.config import model_path  # noqa: E402

import onnxruntime as ort  # noqa: E402
import onnx_asr  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402

try:
    ort.preload_dlls()
except Exception:
    pass

k = Kokoro(model_path("kokoro/kokoro-v1.0.onnx"), model_path("kokoro/voices-v1.0.bin"))


def speech(text, voice, lang):
    audio, sr = k.create(text, voice=voice, lang=lang)
    idx = np.round(np.arange(0, len(audio), sr / 16000)).astype(int)
    return audio[idx[idx < len(audio)]].astype(np.float32)


samples = {
    "en 2s": speech("Hey, what time is it right now?", "am_michael", "en-us"),
    "en 7s": speech("Hello there, this is a short test of the speech recognizer. "
                    "I would like to know what the weather will be like tomorrow.", "am_michael", "en-us"),
    "es 4s": speech("Hola, ¿me puedes recomendar una película para ver esta noche?", "em_alex", "es"),
}

providers_list = [["CPUExecutionProvider"]]
if "CUDAExecutionProvider" in ort.get_available_providers():
    providers_list.insert(0, ["CUDAExecutionProvider", "CPUExecutionProvider"])
if "DmlExecutionProvider" in ort.get_available_providers():
    providers_list.insert(0, ["DmlExecutionProvider", "CPUExecutionProvider"])

for quant in sys.argv[1:] or ["int8"]:
    for providers in providers_list:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        t = time.perf_counter()
        m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", model_path("parakeet-tdt-0.6b-v3"),
                                quantization=quant, providers=providers, sess_options=so)
        load = time.perf_counter() - t
        m.recognize(samples["en 2s"])
        for name, audio in samples.items():
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                text = m.recognize(audio)
                best = min(best, time.perf_counter() - t)
            print(f"{providers[0]:22} {quant or 'fp32':5} load {load:4.1f}s  {name}  {best*1000:5.0f} ms  {text!r}")
