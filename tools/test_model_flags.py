"""Sanity-check model family / type / rating inference."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voicelink.llm import _infer_flags  # noqa: E402

SAMPLES = [
    {"key": "qwen3-moe-6x1.7b", "display_name": "Qwen3 MoE 6x1.7B",
     "architecture": "qwen3moe", "num_experts": 6},
    {"key": "qwen3.8-flash-next-mtp", "display_name": "Qwen3.8 Flash Next MTP",
     "architecture": "qwen4exp"},
    {"key": "dirty-muse-nsfw", "display_name": "Dirty Muse Writer NSFW",
     "architecture": "gemma2"},
    {"key": "ternary-bonsai-2-27b-abliterated",
     "display_name": "Ternary Bonsai 2 27B Abliterated PQ2 0",
     "architecture": "qwen35"},
    {"key": "huihui-qwen3.5-4b-claude-4.6-opus-abliterated",
     "display_name": "Huihui Qwen3.5 4B Claude 4.6 Opus Abliterated",
     "architecture": "qwen35"},
    {"key": "gemma-4-e4b-uncensored", "display_name": "Gemma 4 E4B Uncensored",
     "architecture": "gemma4"},
    {"key": "dirty-muse-writer-v01-uncensored-erotica-nsfw",
     "display_name": "Dirty Muse Writer v01 Uncensored Erotica NSFW",
     "architecture": "llama"},
    {"key": "qwen3-moe-6x1.7b-10.2b-shining-madness-uncensored-i1",
     "display_name": "Qwen3 MoE 6x1.7B Shining Madness Uncensored",
     "architecture": "qwen3moe"},
    {"key": "qwen3.5-4b-emperoai-qwen3.8-distill-heretic-abliterated-mtp",
     "display_name": "Qwen3.5 4B EmperoAI Heretic Abliterated MTP",
     "architecture": "qwen35"},
    {"key": "DIRTY-MUSE-WRITER-V01-UNCENSORED-EROTICA-NSFW",
     "display_name": "DIRTY MUSE WRITER", "architecture": "llama"},
    {"key": "Qwen3.5-4B-EmperoAI-Qwen3.8-Distill-Heretic-Abliterated-MTP",
     "display_name": "Heretic Abliterated", "architecture": "qwen35"},
    {"key": "ternary-bonsai-8b", "display_name": "Ternary Bonsai 8B",
     "architecture": "qwen3"},
    {"key": "qwen3.5-2b", "display_name": "Qwen3.5 2B", "architecture": "qwen35"},
]


def main():
    failed = 0
    for m in SAMPLES:
        f = _infer_flags(m)
        print(f"{m['display_name']!r:55} fam={f['families']} kind={f['kind']} "
              f"nsfw={f['nsfw']} ablit={f['abliterated']}")
        blob = (m["key"] + " " + m["display_name"]).lower()
        if ("uncensored" in blob or "nsfw" in blob) and not f["nsfw"]:
            print("  FAIL: expected +18 / nsfw")
            failed += 1
        if "abliterated" in blob and not f["abliterated"]:
            print("  FAIL: expected Not Filtered / abliterated")
            failed += 1
    if failed:
        raise SystemExit(f"{failed} flag check(s) failed")
    print("ok")


if __name__ == "__main__":
    main()
