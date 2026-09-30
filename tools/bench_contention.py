"""Does TTS on the CPU slow down LM Studio token generation?

    python tools/bench_contention.py
"""
import json
import os
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voicelink.config import load_config, model_path  # noqa: E402

cfg = load_config()["llm"]
BASE = cfg["base_url"].rstrip("/")


def model_id():
    with urllib.request.urlopen(BASE + "/api/v0/models", timeout=5) as r:
        return next(m["id"] for m in json.loads(r.read())["data"] if m.get("state") == "loaded")


def llm_rate(model):
    body = {"model": model, "stream": True, "max_tokens": 120, "temperature": 0.7, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "Tell me a short story about a lighthouse keeper."}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first, n = None, 0
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and "[DONE]" not in line:
                d = json.loads(line[5:])["choices"][0]["delta"]
                if d.get("content"):
                    n += 1
                    first = first or time.perf_counter()
    end = time.perf_counter()
    return first - t0, n / (end - first)


def main():
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    model = model_id()
    print("model:", model)
    ttft, rate = llm_rate(model)
    print(f"idle CPU      : ttft {ttft*1000:.0f} ms, {rate:.1f} tok/s")
    for threads in (8, 6, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        sess = ort.InferenceSession(model_path("kokoro/kokoro-v1.0.onnx"), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        k = Kokoro.from_session(sess, model_path("kokoro/voices-v1.0.bin"))
        stop = threading.Event()
        stats = []

        def loop():
            while not stop.is_set():
                t = time.perf_counter()
                a, sr = k.create("This is a background sentence to keep the speech engine busy.",
                                 voice="af_heart", lang="en-us")
                stats.append((time.perf_counter() - t) / (len(a) / sr))
        th = threading.Thread(target=loop)
        th.start()
        time.sleep(1)
        ttft, rate = llm_rate(model)
        stop.set()
        th.join()
        print(f"TTS {threads} threads: ttft {ttft*1000:.0f} ms, {rate:.1f} tok/s  (TTS RTF {min(stats):.2f})")


if __name__ == "__main__":
    main()
