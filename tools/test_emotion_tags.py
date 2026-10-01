"""Mood-span tag parser (no models required).

    .venv\\Scripts\\python tools/test_emotion_tags.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voicelink.textproc import EmotionSpanFilter, strip_emotion_tags  # noqa: E402


def collect(parts):
    f = EmotionSpanFilter()
    out = []
    for p in parts:
        out.extend(f.feed(p))
    out.extend(f.flush())
    merged = []
    for mood, text in out:
        if merged and merged[-1][0] == mood:
            merged[-1] = (mood, merged[-1][1] + text)
        else:
            merged.append((mood, text))
    return merged


def eq(got, want, label):
    if got != want:
        raise SystemExit(f"FAIL {label}\n  got  {got!r}\n  want {want!r}")
    print("ok ", label)


eq(collect(["He said ", "<angry>Get out!</angry>", " and left."]),
   [(None, "He said "), ("angry", "Get out!"), (None, " and left.")],
   "span around dialogue")

eq(collect(["<ang", "ry>No!</ang", "ry> Fine."]),
   [("angry", "No!"), (None, " Fine.")],
   "tags split across tokens")

eq(collect(["<sad>I said <angry>No!</angry> and wept.</sad>"]),
   [("sad", "I said "), ("angry", "No!"), ("sad", " and wept.")],
   "inner tag wins")

eq(collect(["<WHISPER>Stay quiet.</whisper>"]),
   [("whisper", "Stay quiet.")],
   "case insensitive")

eq(collect(["<p>Hello</p> there"]),
   [(None, "Hello there")],
   "unknown tags stripped, not spoken")

eq(strip_emotion_tags("He said <angry>Get out!</angry> and left."),
   "He said Get out! and left.",
   "strip for the chat bubble")

print("all emotion tag tests passed")
