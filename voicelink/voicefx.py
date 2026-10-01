"""Post-TTS voice effects: pitch, timbre, pitch range, volume, age and emotion.

Three layers are combined into one set of parameters, in this order:
    1. the user's controls (pitch, timbre, range, volume; rate = Kokoro speed),
    2. an age preset (child ... senior),
    3. an emotion preset, scaled by its intensity.
Pitch, formants (timbre / vocal-tract size) and pitch range are changed in one
pass with Praat's PSOLA-based "Change gender" (praat-parselmouth, ~10 ms per
second of audio). Rate is applied by Kokoro itself (cleaner than stretching).
Breathiness, whisper, tremor, drive and volume are cheap numpy stages.
"""
import logging
import math

import numpy as np

log = logging.getLogger("voicefx")

try:
    import parselmouth
    from parselmouth.praat import call as praat
except ImportError:  # effects degrade to rate/volume only
    parselmouth = None

# target_f0: (female, male) median pitch in Hz; younger presets never lower the pitch.
# shift: (female, male) semitones for the older presets (male voices rise a little with age).
AGES = {
    "none":    {},
    "child":   {"target_f0": (300, 285), "formant": 1.22, "range": 1.30, "rate": 0.92, "breath": 0.02},
    "preteen": {"target_f0": (250, 230), "formant": 1.14, "range": 1.20, "rate": 0.97},
    "teen":    {"target_f0": (215, 128), "formant": 1.05, "range": 1.10, "rate": 1.03},
    "adult":   {},
    "mature":  {"shift": (-1.5, -0.5), "formant": 0.98, "range": 0.90, "rate": 0.95, "breath": 0.06,
                "tremor": (0.10, 5.5), "tremolo": 0.03},
    "senior":  {"shift": (-2.5, 0.8), "formant": 0.95, "range": 0.80, "rate": 0.86, "breath": 0.16,
                "tremor": (0.35, 5.5), "tremolo": 0.08},
}

# Values at 100 % intensity. shift in semitones, volume in dB.
EMOTIONS = {
    "none":    {},
    "happy":   {"shift": 2.0, "range": 1.35, "rate": 1.06, "volume": 1.5, "formant": 1.03},
    "excited": {"shift": 3.0, "range": 1.60, "rate": 1.12, "volume": 3.0, "formant": 1.04},
    "calm":    {"shift": -1.0, "range": 0.75, "rate": 0.92, "volume": -1.5, "breath": 0.03},
    "sad":     {"shift": -2.0, "range": 0.60, "rate": 0.86, "volume": -3.0, "formant": 0.98, "breath": 0.07},
    "angry":   {"shift": 1.0, "range": 1.30, "rate": 1.08, "volume": 4.0, "formant": 1.02, "drive": 0.6},
    "fearful": {"shift": 2.5, "range": 1.20, "rate": 1.10, "breath": 0.05, "tremor": (0.5, 7.0), "tremolo": 0.06},
    "whisper": {"volume": -6.0, "rate": 0.70},  # half volume, slower; no extra DSP
}


def _num(v, default, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v)) if math.isfinite(v) else default


def _pair(v, default):
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            return default
    if isinstance(v, (int, float)):
        return (float(v), float(v))
    return default


def merge_preset(base, over):
    """Built-in preset plus the user's saved edits for that chip."""
    out = dict(base or {})
    if not isinstance(over, dict):
        return out
    for key, val in over.items():
        if key in ("target_f0", "shift", "tremor") and isinstance(base.get(key), (list, tuple)):
            out[key] = _pair(val, out.get(key))
        elif key in ("formant", "range", "rate", "breath", "tremolo", "volume", "drive",
                     "whisper", "shift") and not isinstance(val, (list, tuple)):
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                pass
    return out


def _preset(table, name, custom):
    return merge_preset(table.get(name) or {}, (custom or {}).get(name) if isinstance(custom, dict) else None)


def plan(fx, gender="female", voice_f0=None):
    """Turns the UI settings into DSP parameters. Returns None when nothing changes."""
    fx = fx or {}
    male = gender == "male"
    ratio = 2 ** (_num(fx.get("pitch"), 0, -12, 12) / 12)
    formant = 1.25 ** (_num(fx.get("timbre"), 0, -100, 100) / 100)
    rng = _num(fx.get("range"), 100, 0, 250) / 100
    volume = _num(fx.get("volume"), 0, -18, 12)
    rate = 1.0
    breath = whisper = drive = tremolo = 0.0
    tremor = (0.0, 5.5)

    age = _preset(AGES, fx.get("age") or "none", fx.get("ageCustom"))
    if "target_f0" in age:
        target = age["target_f0"][1 if male else 0]
        base = voice_f0 or (120 if male else 190)
        ratio *= max(1.0, target / base)
    if "shift" in age:
        sh = age["shift"]
        sem = sh[1 if male else 0] if isinstance(sh, (list, tuple)) else sh
        ratio *= 2 ** (float(sem) / 12)
    formant *= age.get("formant", 1.0)
    rng *= age.get("range", 1.0)
    rate *= age.get("rate", 1.0)
    breath += age.get("breath", 0.0)
    tremolo += age.get("tremolo", 0.0)
    if "tremor" in age:
        tremor = _pair(age["tremor"], tremor)

    emo_name = fx.get("emotion") or "none"
    emo = _preset(EMOTIONS, emo_name, fx.get("emoCustom"))
    k = _num(fx.get("intensity"), 60, 0, 100) / 100
    if emo_name == "whisper":
        # Normal Kokoro audio only: half volume and slower. No noise or STFT mix
        # (those sounded like a short echo).
        volume += _num(emo.get("volume"), -6.0, -18, 6)
        rate *= _num(emo.get("rate"), 0.70, 0.5, 1.2)
    elif emo and k:
        ratio *= 2 ** (emo.get("shift", 0.0) * k / 12)
        formant *= emo.get("formant", 1.0) ** k
        rng *= emo.get("range", 1.0) ** k
        rate *= emo.get("rate", 1.0) ** k
        volume += emo.get("volume", 0.0) * k
        breath += emo.get("breath", 0.0) * k
        whisper = emo.get("whisper", 0.0) * k
        drive = emo.get("drive", 0.0) * k
        tremolo += emo.get("tremolo", 0.0) * k
        if "tremor" in emo:
            t = _pair(emo["tremor"], (0.0, 5.5))
            tremor = (tremor[0] + t[0] * k, t[1])

    p = {"ratio": ratio, "formant": formant, "range": rng, "rate": rate, "volume": volume,
         "breath": min(breath, 1.0), "whisper": min(whisper, 1.0), "drive": drive,
         "tremor": tremor, "tremolo": min(tremolo, 0.5)}
    neutral = (abs(ratio - 1) < 1e-3 and abs(formant - 1) < 1e-3 and abs(rng - 1) < 1e-3 and abs(rate - 1) < 1e-3
               and abs(volume) < 0.05 and not (breath or whisper or drive or tremolo or tremor[0]))
    return None if neutral else p


def apply(audio, sr, p, rng=None):
    """Applies a plan from `plan()` to float32 mono audio (rate is not handled here)."""
    if p is None or len(audio) < sr // 20:
        return audio
    x = audio.astype(np.float64)
    peak_in = float(np.max(np.abs(x))) or 1.0
    if parselmouth is not None and (abs(p["ratio"] - 1) > 1e-3 or abs(p["formant"] - 1) > 1e-3
                                    or abs(p["range"] - 1) > 1e-3):
        x = _change_gender(x, sr, p["ratio"], p["formant"], p["range"])
    if p["tremor"][0] or p["tremolo"]:
        x = _tremor(x, sr, p["tremor"][0], p["tremor"][1], p["tremolo"])
    if p["breath"] or p["whisper"]:
        x = _breathy(x, sr, p["breath"], p["whisper"], rng or np.random.default_rng())
    if p["drive"]:
        k = 1 + 4 * p["drive"]
        x = np.tanh(k * x / peak_in) / np.tanh(k) * peak_in
    if p["volume"]:
        x *= 10 ** (p["volume"] / 20)
    return _soft_limit(x).astype(np.float32)


def _change_gender(x, sr, ratio, formant, rng):
    try:
        snd = parselmouth.Sound(x, sampling_frequency=sr)
        pitch = snd.to_pitch_ac(0.01, 60, 600)
        median = praat(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
        if not median or not math.isfinite(median):
            if abs(formant - 1) < 1e-3:
                return x
            median = 0.0
        new_median = median * ratio if median else 0.0
        out = praat(snd, "Change gender", 60, 600, formant, new_median, rng, 1.0)
        y = out.values[0]
        return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    except Exception as e:  # never lose a sentence to an effect
        log.debug("Change gender failed: %s", e)
        return x


def _tremor(x, sr, depth_st, freq, tremolo):
    """Vibrato through a modulated delay line (+ amplitude tremolo)."""
    t = np.arange(len(x)) / sr
    phase = 2 * np.pi * freq * t
    if depth_st:
        # peak frequency deviation = 2*pi*f*D  ->  D for a deviation of depth_st semitones
        dev = 2 ** (depth_st / 12) - 1
        delay = dev / (2 * np.pi * freq) * (1 + np.sin(phase))
        x = np.interp(t - delay, t, x, left=0.0)
    if tremolo:
        x = x * (1 - tremolo * 0.5 * (1 + np.sin(phase + 0.7)))
    return x


def _whisper_air(x, sr, amount, rng):
    """Quiet whisper: duck the voice and add envelope-followed hiss.

    A random-phase STFT copy of the voice sounds like a short echo, so this
    uses high-passed noise shaped by the amplitude envelope instead.
    """
    amount = min(1.0, max(0.0, amount))
    win = max(8, sr // 250)
    env = np.convolve(np.abs(x), np.ones(win) / win, mode="same")
    env /= float(np.max(env)) + 1e-9
    noise = rng.standard_normal(len(x))
    hiss = np.empty_like(noise)
    hiss[0] = noise[0]
    hiss[1:] = noise[1:] - 0.94 * noise[:-1]
    peak = float(np.max(np.abs(hiss))) or 1.0
    air = (hiss / peak) * env * (0.10 + 0.20 * amount)
    return x * (1.0 - 0.70 * amount) + air


def _breathy(x, sr, breath, whisper, rng, n=512, hop=128):
    """Aspiration noise that follows the voice's spectral envelope.
    Whisper is a quieter, slower, non-echoing mix (see `_whisper_air`)."""
    if whisper:
        return _whisper_air(x, sr, whisper, rng)
    win = np.hanning(n)
    pad = n
    xp = np.pad(x, (pad, pad + hop))
    frames = np.lib.stride_tricks.sliding_window_view(xp, n)[::hop] * win
    spec = np.fft.rfft(frames, axis=1)
    f = np.fft.rfftfreq(n, 1 / sr)
    weight = np.clip((f - 300) / 2200, 0, 1)
    noisy = np.abs(spec) * weight * np.exp(1j * rng.uniform(0, 2 * np.pi, spec.shape))
    yf = np.fft.irfft(noisy, n=n, axis=1) * win
    y = np.zeros(len(xp))
    for i, fr in enumerate(yf):
        y[i * hop:i * hop + n] += fr
    y = y[pad:pad + len(x)] / (np.sum(win ** 2) / hop)
    return x * (1 - 0.3 * breath) + y * (1.6 * breath)


def _soft_limit(x, knee=0.8):
    a = np.abs(x)
    over = a > knee
    if over.any():
        x = x.copy()
        x[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return x
