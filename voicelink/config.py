"""Configuration: built-in defaults deep-merged with the user's config.json."""
import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.json")
MODELS_DIR = os.path.join(ROOT, "models")

DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, quick-witted voice assistant having a real-time spoken conversation. "
    "Everything you write is read aloud by a text-to-speech engine, so: "
    "keep replies short and natural (usually one to three sentences) unless the user asks for detail; "
    "never use markdown, bullet lists, headings, code blocks, tables, emojis or special symbols, "
    "except the mood tags described at the end of this prompt; "
    "write numbers, dates and units the way a person would say them. "
    "Reply in the language the user is speaking unless a later instruction names a specific language — then stay in that language. "
    "The user may interrupt you while you talk; an assistant message ending with an em dash means "
    "you were cut off at that point and the user did not hear the rest, so continue naturally."
)

DEFAULTS = {
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
        "open_browser": True,
    },
    "llm": {
        "base_url": "http://127.0.0.1:1234",
        "model": "auto",
        "api_key": "lm-studio",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_tokens": 512,
        "disable_thinking": True,
        "history_messages": 40,
        "auto_title": True,  # name new conversations with the LLM after the first reply
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
    "stt": {
        "model": "nemo-parakeet-tdt-0.6b-v3",
        "model_dir": "parakeet-tdt-0.6b-v3",
        "quantization": "int8",
        "device": "cpu",
        "cpu_threads": 4,
        "languages": ["en"],
        "live_transcript": True,
        "live_interval_ms": 1000,
        "noise_filter": "light",  # off | light | strong (the browser's own suppression runs first)
    },
    "tts": {
        "device": "cpu",
        "model_file": "kokoro/kokoro-v1.0.onnx",
        "voices_file": "kokoro/voices-v1.0.bin",
        "cpu_threads": 4,
        "speed": 1.05,
        "voices": {
            "en": "af_heart",
            "es": "ef_dora",
            "fr": "ff_siwis",
            "it": "if_sara",
            "pt": "pf_dora",
            "hi": "hf_alpha",
        },
        # Used by "Auto" when the voice panel's gender filter is set to male.
        "voices_male": {
            "en": "am_michael",
            "es": "em_alex",
            "it": "im_nicola",
            "pt": "pm_alex",
            "hi": "hm_omega",
        },
        "first_chunk_min_chars": 12,
        "first_chunk_wait_ms": 600,
        "chunk_max_chars": 220,
    },
    "vad": {
        "threshold": 0.5,
        "neg_threshold": 0.35,
        "min_speech_ms": 160,
        "spec_silence_ms": 250,
        "end_silence_ms": 800,
        "preroll_ms": 350,
        "max_utterance_s": 30,
        "barge_in_ms": 280,
        "barge_in_threshold": 0.7,
        "echo_check": True,
        "verify_max_ms": 1500,
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=CONFIG_PATH):
    user = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
    cfg = _merge(DEFAULTS, user)
    if os.environ.get("VOICELINK_PORT"):
        cfg["server"]["port"] = int(os.environ["VOICELINK_PORT"])
    return cfg


def model_path(rel):
    return rel if os.path.isabs(rel) else os.path.join(MODELS_DIR, rel)
