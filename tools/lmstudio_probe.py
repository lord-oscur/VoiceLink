"""Probe an LM Studio server: list loaded models and measure which
"no thinking" strategy works (time to first spoken token).

Usage:  python tools/lmstudio_probe.py [base_url] [model]
Only uses the Python standard library.
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1234"
MODEL = sys.argv[2] if len(sys.argv) > 2 else None


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())


def loaded_model():
    try:
        for m in get("/api/v0/models")["data"]:
            if m.get("state") == "loaded" and m.get("type") in ("llm", "vlm"):
                return m["id"]
    except Exception:
        pass
    return get("/v1/models")["data"][0]["id"]


def run(label, messages, extra=None):
    body = {"model": MODEL, "stream": True, "max_tokens": 200, "temperature": 0.7,
            "messages": messages}
    body.update(extra or {})
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_content = None
    reasoning_chars = 0
    content = ""
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0].get("delta", {})
            reasoning_chars += len(delta.get("reasoning_content") or delta.get("reasoning") or "")
            c = delta.get("content") or ""
            if c and first_content is None and c.strip():
                first_content = time.perf_counter() - t0
            content += c
    total = time.perf_counter() - t0
    fc = f"{first_content:.2f}s" if first_content else "never"
    print(f"[{label}] first content: {fc}  total: {total:.2f}s  reasoning chars: {reasoning_chars}")
    print("   ->", content.strip().replace("\n", " ")[:160])


if __name__ == "__main__":
    MODEL = MODEL or loaded_model()
    print("Model:", MODEL)
    sys_msg = {"role": "system", "content": "You are a friendly voice assistant. Reply in one or two short sentences."}
    user = {"role": "user", "content": "Hi! What's a good name for a cat?"}
    run("baseline", [sys_msg, user])
    run("reasoning_effort=none", [sys_msg, user], {"reasoning_effort": "none"})
    run("prefill empty think", [sys_msg, user, {"role": "assistant", "content": "<think>\n\n</think>\n\n"}])
