"""Download every model the voice pipeline needs into ./models (about 1 GB).

    python tools/download_models.py

Files that already exist are skipped, so it is safe to run repeatedly.
"""
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

FILES = {
    "vad/silero_vad.onnx":
        "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
    "kokoro/kokoro-v1.0.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "kokoro/voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
PARAKEET_FILES = ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]


def fetch(url, dest):
    rel = os.path.relpath(dest, ROOT)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  ok   {rel}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  get  {rel}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r       {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    if total:
        print()
    os.replace(tmp, dest)


def fetch_parakeet():
    target = os.path.join(MODELS, "parakeet-tdt-0.6b-v3")
    if all(os.path.exists(os.path.join(target, f)) for f in PARAKEET_FILES):
        print("  ok   models/parakeet-tdt-0.6b-v3")
        return
    print(f"  get  models/parakeet-tdt-0.6b-v3  (Hugging Face: {PARAKEET_REPO}, ~670 MB)")
    from huggingface_hub import snapshot_download
    snapshot_download(PARAKEET_REPO, local_dir=target, allow_patterns=PARAKEET_FILES)


def main():
    print("Downloading models into", MODELS)
    try:
        for rel, url in FILES.items():
            fetch(url, os.path.join(MODELS, rel))
        fetch_parakeet()
    except Exception as e:
        print(f"\nDownload failed: {e}\nCheck your internet connection and run setup.bat again.")
        sys.exit(1)
    print("All models ready.")


if __name__ == "__main__":
    main()
