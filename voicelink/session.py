"""One live conversation over a WebSocket.

Audio in  : binary frames of int16 mono 16 kHz PCM (any length).
Audio out : binary frames  [uint32 turn][uint32 seg][int16 mono 24 kHz PCM].
Control   : JSON text frames in both directions (see README, "Protocol").

Turn taking
  * Silero VAD runs on every 32 ms frame.
  * After `spec_silence_ms` of silence a *speculative* turn starts: STT, the
    LLM request and TTS all run, but nothing is sent to the browser.
  * After `end_silence_ms` the turn is *committed* and its buffered output is
    released. If the user resumes speaking before that, it is cancelled.
Barge-in
  * While the assistant is talking, speech ducks the playback immediately.
  * Once `barge_in_ms` of speech is confirmed the assistant is interrupted -
    after an optional STT "echo check" that ignores the assistant's own voice
    leaking into the microphone (speakers without headphones).
  * Only the part of the reply the user actually heard is kept in history.
"""
import asyncio
import json
import logging
import struct
import time
from collections import deque
from difflib import SequenceMatcher

import numpy as np

from . import attachments as att
from .llm import LLMError
from .textproc import (LANGUAGE_NAMES, EmotionSpanFilter, MOOD_TAG_HINT, SentenceChunker, clean_for_speech,
                       confident_language, detect_language, has_words, normalized_words, strip_emotion_tags)
from .tts import SAMPLE_RATE as TTS_SR, voice_language
from .denoise import Denoiser
from .speaker import threshold as speaker_threshold
from .vad import FRAME, SAMPLE_RATE, SileroVAD

log = logging.getLogger("session")
FRAME_MS = FRAME * 1000.0 / SAMPLE_RATE
_NOISE_WORDS = {"", "uh", "um", "hmm", "mm", "ah", "eh", "oh", "mhm"}
# Said while listening; they must not stop the assistant.
_BACKCHANNEL = set("uh um hmm mm mhm ah eh oh yeah yes yep ok okay right sure i see "
                   "sí si vale ajá aha claro bueno ya".split())
# Always interrupt, even if they also appear in the assistant's own sentence.
_STOP_WORDS = set("stop wait hold pause enough quiet shut para espera alto basta cállate silencio "
                  "momento arrête attends aspetta pare".split())
_VERIFY_KEEP_MS = 1000  # audio kept when the echo-check window slides
_ENROLL_MS = 10000      # speech needed for a voice print
_LOOP_LEAD_S = 0.25     # speakers reach the mic a little after loopback captures them
_QUIET = 10 ** (-50 / 20)


def now():
    return time.perf_counter()


def _max_corr(mic, played, max_lag):
    """Peak normalized correlation, allowing the mic to lag the speakers."""
    n = min(len(mic), len(played))
    if n < 16:
        return 0.0
    best = -1.0
    step = max(1, SAMPLE_RATE // 200)  # 5 ms
    for lag in range(0, min(max_lag, n // 2), step):
        a, b = mic[lag:], played[:n - lag]
        a0, b0 = a - a.mean(), b - b.mean()
        denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-9
        best = max(best, float(np.dot(a0, b0) / denom))
    return best


def _ts():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


_GEN_KEYS = {"temperature": (0.0, 2.0), "top_p": (0.0, 1.0), "top_k": (0, 500), "repeat_penalty": (0.5, 2.0),
             "max_tokens": (16, 32768), "history": (2, 400)}


def _clean_gen(gen):
    """Per-request LLM settings from the model panel; unset values use config.json."""
    out = {}
    for key, (lo, hi) in _GEN_KEYS.items():
        v = gen.get(key)
        if v is None or v == "":
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        v = min(hi, max(lo, v))
        out[key] = int(v) if key in ("top_k", "max_tokens", "history") else v
    if gen.get("thinking") is not None:
        out["thinking"] = bool(gen["thinking"])
    return out


class Turn:
    def __init__(self, tid, audio=None, text=None, speech_end=None):
        self.id = tid
        self.audio = audio
        self.user_text = text
        self.speech_end = speech_end or now()
        self.committed = asyncio.Event()
        self.outbox = asyncio.Queue()
        self.task = None
        self.subtasks = []
        self.segments = []          # (text, est_start, duration)
        self.play_until = 0.0       # estimated end of playback in the browser
        self.done = False
        self.interrupted = False
        self.cancelled = False
        self.hist_idx = None
        self.lang = None
        self.user_lang = None
        self.attachments = []
        self.llm_raw = ""           # assistant tokens including mood tags
        self.metrics = {}
        if speech_end:
            self.metrics["start_ms"] = round((now() - speech_end) * 1000)

    def emit(self, item):
        self.outbox.put_nowait(item)

    def cancel(self):
        self.cancelled = True
        for t in [self.task, *self.subtasks]:
            if t and not t.done():
                t.cancel()

    def heard_text(self, at):
        words = []
        for text, start, dur in self.segments:
            if at >= start + dur:
                words.append(text)
            elif at > start:
                w = text.split()
                n = int(len(w) * (at - start) / max(dur, 1e-3))
                if n:
                    words.append(" ".join(w[:n]))
                break
            else:
                break
        return " ".join(words)

    def text_near(self, at, before=4.0, after=0.5):
        """Text of the segments playing around time `at` (for the echo check)."""
        return " ".join(t for t, s, d in self.segments if s <= at + after and s + d >= at - before)


class Session:
    def __init__(self, ws, engines):
        self.ws = ws
        self.eng = engines
        self.cfg = engines.cfg
        self.vc = self.cfg["vad"]
        self.vad = SileroVAD(engines.vad_file)
        self.out = asyncio.Queue()
        self.history = []
        self.conv_id = None
        self.titled = False
        self.incognito = False
        self._skip_persist = False
        self.labels = []
        self.project_id = None
        self.pending = {}           # attachments waiting for the next user turn (id -> record)
        self.gen = {}
        self.system_prompt = None
        self.langs = self.cfg["stt"]["languages"] or None
        self.lang = (self.langs or ["en"])[0]
        self.output_lang = "voice"
        self.headphones = False
        self.voice = None
        self.gender = None
        self.fx = None
        self.speed = None
        self.mic_on = True
        self.pcm = np.zeros(0, dtype=np.float32)
        self.denoiser = Denoiser(self.cfg["stt"].get("noise_filter", "light"))
        self.playback_gate = False  # ignore speech that is coming out of this PC's speakers
        self.loop_device = ""
        self.only_me = False        # accept only the enrolled voice
        self.voiceprint = None
        self.only_me_thr = speaker_threshold(50)
        self.enroll = None          # speech frames while recording a voice print
        self.utt_t0 = 0.0
        self.preroll = deque(maxlen=max(1, int(self.vc["preroll_ms"] / FRAME_MS)))
        self.turn = None            # committed assistant turn (may still be playing)
        self.spec = None            # speculative (uncommitted) turn
        self.seq = 0
        self.state = None
        self.ducked = False
        self._reset_speech()
        self.writer = asyncio.create_task(self._writer())
        self.live_task = None
        self.verify_task = None

    # ------------------------------------------------------------ plumbing

    def send(self, msg):
        self.out.put_nowait(msg)

    async def _writer(self):
        while True:
            item = await self.out.get()
            if self.ws.closed:
                return
            try:
                if isinstance(item, (bytes, bytearray)):
                    await self.ws.send_bytes(item)
                else:
                    await self.ws.send_str(json.dumps(item))
            except (ConnectionResetError, RuntimeError):
                return

    async def close(self):
        for t in (self.turn, self.spec):
            if t:
                t.cancel()
        for t in (self.live_task, self.verify_task, self.writer):
            if t and not t.done():
                t.cancel()
        self.eng.loopback.release(id(self))

    def _set_state(self, state):
        if state != self.state:
            self.state = state
            self.send({"type": "state", "state": state})

    def _refresh_state(self):
        if self.in_speech and not self.verifying:
            self._set_state("user_speaking")
        elif self.assistant_active():
            t = self.turn
            self._set_state("speaking" if t.segments and now() < t.play_until + 0.2 else "thinking")
        else:
            self._set_state("listening")

    def assistant_active(self):
        t = self.turn
        if t is None or t.cancelled or not t.committed.is_set():
            return False
        return (not t.done) or now() < t.play_until

    async def _stt(self, audio):
        loop = asyncio.get_running_loop()
        self.eng.stt_jobs += 1
        try:
            return await loop.run_in_executor(self.eng.stt_pool, self.eng.stt.transcribe, audio)
        finally:
            self.eng.stt_jobs -= 1

    # --------------------------------------------------------- client input

    async def on_json(self, msg):
        kind = msg.get("type")
        if kind == "config":
            if "headphones" in msg:
                self.headphones = bool(msg["headphones"])
            if "voice" in msg:
                self.voice = msg["voice"] or None
            if "gender" in msg:
                self.gender = msg["gender"] if msg["gender"] in ("male", "female") else None
            if "fx" in msg:
                self.fx = msg["fx"] if isinstance(msg["fx"], dict) else None
            if "speed" in msg:
                self.speed = float(msg["speed"]) if msg["speed"] else None
            if "mic" in msg:
                self.mic_on = bool(msg["mic"])
                if not self.mic_on:
                    self._abort_speech()
            if msg.get("noise") in ("off", "light", "strong") and msg["noise"] != self.denoiser.level:
                self.denoiser.set_level(msg["noise"])
                self.denoiser.reset()
            if "playback_gate" in msg or "loopback" in msg:
                self.playback_gate = bool(msg.get("playback_gate", self.playback_gate))
                self.loop_device = str(msg.get("loopback", self.loop_device) or "")
                if self.playback_gate:
                    await asyncio.get_running_loop().run_in_executor(
                        None, self.eng.loopback.acquire, id(self), self.loop_device)
                else:
                    self.eng.loopback.release(id(self))
            if "only_me" in msg:
                self.only_me = bool(msg["only_me"])
            if "only_me_strictness" in msg:
                self.only_me_thr = speaker_threshold(msg["only_me_strictness"] or 50)
            if "voiceprint" in msg:
                vp = msg["voiceprint"]
                self.voiceprint = None
                if isinstance(vp, list) and len(vp) >= 64:
                    v = np.asarray(vp, dtype=np.float32)
                    self.voiceprint = v / (np.linalg.norm(v) + 1e-9)
            if "gen" in msg and isinstance(msg["gen"], dict):
                self.gen = _clean_gen(msg["gen"])
            if "system_prompt" in msg:
                self.system_prompt = (msg["system_prompt"] or "").strip() or None
            if "languages" in msg and isinstance(msg["languages"], list):
                allowed = [x for x in msg["languages"] if isinstance(x, str) and x in LANGUAGE_NAMES]
                if "en" not in allowed:
                    allowed = ["en"] + allowed
                if allowed:
                    self.langs = allowed
                    if self.lang not in allowed:
                        self.lang = allowed[0]
            if "output_lang" in msg:
                v = msg["output_lang"]
                if v in ("voice", "auto") or v in LANGUAGE_NAMES:
                    self.output_lang = v
            if "incognito" in msg:
                self.incognito = bool(msg["incognito"])
                if self.incognito:
                    self._skip_persist = True
                    log.info("Incognito on: this chat will not be written to disk")
        elif kind == "text":
            text = (msg.get("text") or "").strip()
            if text or self.pending:
                self._interrupt("typed message")
                self._cancel_spec()
                turn = self._new_turn(text=text)
                self._commit(turn)
        elif kind == "attach_add":
            a = msg.get("attachment") or {}
            if a.get("kind") in ("image", "document", "audio", "video") and a.get("id"):
                self.pending[str(a["id"])] = {k: a.get(k) for k in (
                    "id", "kind", "name", "data", "text", "frames", "duration", "truncated") if a.get(k) is not None}
        elif kind == "attach_remove":
            self.pending.pop(str(msg.get("id")), None)
        elif kind == "attach_clear":
            self.pending.clear()
        elif kind == "enroll_start":
            if not self.eng.speaker.available():
                self.send({"type": "enroll_error", "message": "The voice model is missing - run setup.bat."})
                return
            self._interrupt("voice enrollment")
            self._abort_speech()
            self.enroll = []
            self.send({"type": "enroll_progress", "pct": 0})
        elif kind == "enroll_cancel":
            self.enroll = None
        elif kind == "playback_idle":
            t = self.turn
            if t and t.id == msg.get("turn") and t.done:
                t.play_until = min(t.play_until, now())
            self._refresh_state()
        elif kind == "interrupt":
            self._interrupt("button")
            self._refresh_state()
        elif kind == "reset":
            self._interrupt("reset")
            self._cancel_spec()
            self.history = []
            self.conv_id = None
            self.titled = False
            self.labels = []
            self.project_id = None
            if "incognito" in msg:
                self.incognito = bool(msg["incognito"])
            self._skip_persist = self.incognito
            self.send({"type": "reset_done", "incognito": self.incognito})
            self._refresh_state()
        elif kind == "conv_meta":
            if "labels" in msg and isinstance(msg["labels"], list):
                self.labels = [str(x).strip()[:40] for x in msg["labels"] if str(x).strip()]
            if "project_id" in msg:
                self.project_id = msg["project_id"] or None
            if self.conv_id and not self.incognito:
                try:
                    conv = self.eng.store.update(self.conv_id, labels=self.labels,
                                                 project_id=self.project_id)
                    self.send({"type": "conv_saved", "id": conv["id"], "title": conv.get("title"),
                               "labels": conv.get("labels") or [], "project_id": conv.get("project_id")})
                except (KeyError, OSError, ValueError) as e:
                    log.warning("Could not update conversation meta: %s", e)
        elif kind == "open":
            try:
                conv = self.eng.store.get(msg.get("id"))
            except (KeyError, OSError, ValueError):
                self.send({"type": "error", "message": "That conversation no longer exists."})
                return
            self._interrupt("open conversation")
            self._cancel_spec()
            self.history = conv.get("messages", [])
            self.conv_id = conv["id"]
            self.titled = True
            self.incognito = False
            self._skip_persist = False
            self.labels = list(conv.get("labels") or [])
            self.project_id = conv.get("project_id") or None
            shown = []
            for m in self.history:
                row = dict(m)
                if m.get("attachments"):
                    row["attachments"] = [att.public(a) for a in m["attachments"]]
                if row.get("role") == "assistant" and isinstance(row.get("content"), str):
                    row["content"] = strip_emotion_tags(row["content"])
                shown.append(row)
            self.send({"type": "conversation", "id": conv["id"], "title": conv.get("title"),
                       "messages": shown, "labels": self.labels, "project_id": self.project_id})
            self._refresh_state()

    def on_audio(self, data):
        if not self.mic_on:
            return
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) * (1.0 / 32768.0)
        x = self.denoiser.process(x)
        self.pcm = np.concatenate([self.pcm, x]) if len(self.pcm) else x
        n = len(self.pcm) // FRAME
        for i in range(n):
            self._frame(self.pcm[i * FRAME:(i + 1) * FRAME])
        self.pcm = self.pcm[n * FRAME:]
        if self.state in ("speaking", "thinking") and not self.assistant_active():
            self._refresh_state()

    # ------------------------------------------------------------ VAD logic

    def _reset_speech(self):
        self.in_speech = False
        self.verifying = False
        self.end_pending = False
        self.run_ms = 0.0
        self.silence_ms = 0.0
        self.utt = []
        self.utt_ms = 0.0
        self.last_live = 0.0

    def _abort_speech(self):
        self._cancel_spec()
        if self.verify_task and not self.verify_task.done():
            self.verify_task.cancel()
        self._reset_speech()
        self.preroll.clear()
        self._unduck()
        self._refresh_state()

    def _duck(self):
        if not self.ducked:
            self.ducked = True
            self.send({"type": "duck"})

    def _unduck(self):
        if self.ducked:
            self.ducked = False
            self.send({"type": "unduck"})

    def _frame(self, frame):
        p = self.vad(frame)
        vc = self.vc
        if self.enroll is not None:
            self._enroll_frame(frame, p)
            return
        if not self.in_speech:
            self.preroll.append(frame)
            active = self.assistant_active()
            thr = vc["barge_in_threshold"] if active and not self.headphones else vc["threshold"]
            need = vc["barge_in_ms"] if active else vc["min_speech_ms"]
            if p >= thr:
                self.run_ms += FRAME_MS
                if active:
                    self._duck()
            elif p < vc["neg_threshold"]:
                self.run_ms = max(0.0, self.run_ms - FRAME_MS)
                if self.run_ms == 0 and self.ducked:
                    self._unduck()
            if self.run_ms >= need:
                self._speech_start(active)
            return

        self.utt.append(frame)
        self.utt_ms += FRAME_MS
        if p >= vc["neg_threshold"]:
            if self.silence_ms and p >= vc["threshold"]:
                self._cancel_spec()
            if p >= vc["threshold"] or not self.silence_ms:
                self.silence_ms = 0.0
        else:
            self.silence_ms += FRAME_MS

        if self.verifying:
            if self.silence_ms >= vc["end_silence_ms"]:
                self.end_pending = True
            return

        if self.silence_ms >= vc["end_silence_ms"] or self.utt_ms >= vc["max_utterance_s"] * 1000:
            self._speech_end()
            return
        if self.silence_ms >= vc["spec_silence_ms"] and self.spec is None:
            self.spec = self._new_turn(audio=self._utt_audio(),
                                       speech_end=now() - self.silence_ms / 1000.0)
        self._maybe_live()

    def _utt_audio(self):
        return np.concatenate(self.utt) if self.utt else np.zeros(0, dtype=np.float32)

    def _speech_start(self, assistant_active):
        self.in_speech = True
        self.utt = list(self.preroll)
        self.utt_ms = len(self.utt) * FRAME_MS
        self.preroll.clear()
        self.run_ms = 0.0
        self.silence_ms = 0.0
        self.last_live = now()
        self.utt_t0 = now() - self.utt_ms / 1000.0
        gate = self._gate(assistant_active)
        if gate:
            self.verifying = True
            self.verify_task = asyncio.create_task(self._verify_speech(assistant_active, gate))
        elif assistant_active:
            self._interrupt("barge-in")
        self._refresh_state()

    def _gate(self, assistant_active):
        """Which checks new speech must pass before it counts, or None."""
        if self.only_me and self.voiceprint is not None:
            return "speaker"
        echo = assistant_active and not self.headphones and self.vc["echo_check"]
        playback = self.playback_gate and self.eng.loopback.playing()
        if echo and playback:
            return "echo+playback"
        return "echo" if echo else "playback" if playback else None

    # ------------------------------------------------------ voice enrollment

    def _enroll_frame(self, frame, p):
        if p >= self.vc["threshold"]:
            self.enroll.append(frame)
            got = len(self.enroll) * FRAME_MS
            if len(self.enroll) % 8 == 0:
                self.send({"type": "enroll_progress", "pct": min(99, round(100 * got / _ENROLL_MS))})
            if got >= _ENROLL_MS:
                audio = np.concatenate(self.enroll)
                self.enroll = None
                asyncio.create_task(self._finish_enroll(audio))

    async def _finish_enroll(self, audio):
        try:
            vp = await asyncio.get_running_loop().run_in_executor(None, self.eng.speaker.enroll, audio)
        except Exception as e:
            log.exception("Voice enrollment failed")
            self.send({"type": "enroll_error", "message": str(e) or type(e).__name__})
            return
        self.voiceprint = vp
        log.info("Voice print recorded (%d s of speech)", len(audio) // SAMPLE_RATE)
        self.send({"type": "enrolled", "voiceprint": [round(float(x), 5) for x in vp]})

    def _speech_end(self):
        speech_end = now() - self.silence_ms / 1000.0
        turn = self.spec or self._new_turn(audio=self._utt_audio(), speech_end=speech_end)
        self.spec = None
        self._reset_speech()
        self._commit(turn)

    def _barge_in_verdict(self, text, trimmed=False, ref=None):
        """True = the user said something new, None = not sure yet (echo,
        backchannel, or the same words coming out of the speakers)."""
        words = normalized_words(text)
        if trimmed:
            words = words[1:]  # the window may start mid-word
        if ref is None:
            ref = set(normalized_words(self.turn.text_near(now()))) if self.turn else set()
        if any(w in _STOP_WORDS and w not in ref for w in words):
            return True
        # Garbled echo ("for its" -> "fruit") looks like a new word, so a
        # word close to one the assistant is saying counts as echo, and it
        # takes two new words making up half of what was heard.
        novel = [w for w in words if len(w) > 1 and w not in _BACKCHANNEL and w not in ref
                 and not any(SequenceMatcher(None, w, r).ratio() >= 0.6 for r in ref)]
        if len(novel) >= 2 and len(novel) * 2 >= len(words):
            return True
        return None

    async def _verify_speech(self, assistant_active, gate):
        """Holds new speech until it passes `gate`:
        * "speaker": its voice print must match the enrolled one.
        * "echo" / "playback": keep transcribing it as it grows and accept it
          only for words that are not what the assistant is saying (echo) or
          what the PC is playing (loopback transcript).
        Echo and videos keep the VAD busy for long stretches, so once the
        audio exceeds verify_max_ms the window slides forward instead of
        giving up; otherwise words said in the middle would be thrown away."""
        vc = self.vc
        keep = max(1, int(_VERIFY_KEEP_MS / FRAME_MS))
        trimmed = False
        text = ""
        kind = None
        try:
            if gate == "speaker":
                verdict, text = await self._speaker_verdict()
                if verdict is None:
                    return
            else:
                while True:
                    heard_ms = self.utt_ms
                    text = await self._stt(self._utt_audio())
                    if not self.in_speech:
                        return
                    ref = set()
                    if "echo" in gate and self.turn:
                        ref |= set(normalized_words(self.turn.text_near(now())))
                    kind = None
                    if "playback" in gate:
                        played = self._loopback_for(self._utt_audio())
                        play_text = ""
                        if len(played) >= SAMPLE_RATE // 4 and self._tail_rms(played) >= _QUIET:
                            play_text = await self._stt(played)
                            if not self.in_speech:
                                return
                            ref |= set(normalized_words(play_text))
                        kind = self._playback_kind(self._utt_audio(), played, text, play_text)
                    verdict = self._barge_in_verdict(text, trimmed, ref)
                    if kind == "quiet" and self._has_real_words(text) and self._tail_rms(self._utt_audio()) >= 10 ** (-40 / 20):
                        verdict = True
                    elif kind == "copy":
                        words = normalized_words(text)
                        if any(w in _STOP_WORDS and w not in ref for w in words):
                            verdict = True
                        else:
                            verdict = None if not self.end_pending else False
                    log.debug("%s check %.0f ms kind=%s -> %s: %r", gate, heard_ms, kind, verdict, text)
                    if verdict or self.end_pending:
                        break
                    # Long echo or a video: leave CPU for the TTS.
                    step = (1000 if trimmed else 600) if "playback" in gate else (500 if trimmed else 250)
                    while self.in_speech and not self.end_pending and self.utt_ms < heard_ms + step:
                        await asyncio.sleep(0.03)
                    if not self.in_speech:
                        return
                    if self.utt_ms >= vc["verify_max_ms"] and not self.end_pending:
                        dropped = len(self.utt) - keep
                        if dropped > 0:
                            self.utt_t0 += dropped * FRAME / SAMPLE_RATE
                        self.utt = self.utt[-keep:]
                        self.utt_ms = len(self.utt) * FRAME_MS
                        trimmed = True
        except asyncio.CancelledError:
            return
        if not verdict:
            log.info("Speech ignored (%s%s): %r", gate, f"/{kind}" if kind else "", text)
            self._reset_speech()
            self._unduck()
            self._refresh_state()
            return
        log.info("Speech accepted (%s)%s", gate, f": {text!r}" if text else "")
        self.verifying = False
        if assistant_active:
            self._interrupt("barge-in")
        if self.end_pending:
            self._speech_end()
        self._refresh_state()

    async def _speaker_verdict(self):
        """(accepted, "") once there is enough audio, or (None, "") if the speech vanished."""
        while self.in_speech and not self.end_pending and self.utt_ms < 1500:
            await asyncio.sleep(0.03)
        if not self.in_speech:
            return None, ""
        audio = self._utt_audio()
        vp = await asyncio.get_running_loop().run_in_executor(None, self.eng.speaker.embed, audio)
        if not self.in_speech:
            return None, ""
        score = float(np.dot(vp, self.voiceprint))
        thr = self.only_me_thr - (0.06 if len(audio) < SAMPLE_RATE else 0.0)  # short clips score lower
        ok = score >= thr
        self.send({"type": "speaker", "score": round(score, 3), "threshold": round(thr, 3), "accepted": ok})
        log.debug("voice match %.2f (needs %.2f) -> %s", score, thr, ok)
        return ok, ""

    def _loopback_for(self, audio):
        # Align to the microphone clock, not wall time: STT can take hundreds of
        # milliseconds, during which the speakers may already have gone quiet.
        t1 = (self.utt_t0 + len(audio) / SAMPLE_RATE) if self.utt_t0 else now()
        dur = max(len(audio) / SAMPLE_RATE, 0.3)
        return self.eng.loopback.window(t1 - dur - _LOOP_LEAD_S, t1)

    @staticmethod
    def _tail_rms(x, seconds=0.5):
        if x is None or len(x) == 0:
            return 0.0
        n = min(len(x), int(seconds * SAMPLE_RATE))
        t = x[-n:]
        return float(np.sqrt(np.mean(t * t)))

    def _playback_kind(self, mic, played, mic_text="", play_text=""):
        """'copy' = the mic is the speakers, 'quiet' = speakers silent, 'mix' = both."""
        play_rms = self._tail_rms(played)
        mic_rms = self._tail_rms(mic)
        if play_rms < _QUIET:
            return "quiet"
        mic_w = {w for w in normalized_words(mic_text) if len(w) > 1 and w not in _BACKCHANNEL}
        play_w = {w for w in normalized_words(play_text) if len(w) > 1}
        if mic_w and play_w:
            shared = sum(1 for w in mic_w if w in play_w or any(SequenceMatcher(None, w, r).ratio() >= 0.6 for r in play_w))
            if shared >= 0.3 * len(mic_w):
                return "copy"
        # A video leaking into the mic is quieter than (or similar to) the speakers
        # themselves. Speech over a video is louder.
        if mic_rms <= play_rms * 0.85:
            return "copy"
        corr = _max_corr(mic.astype(np.float64), played.astype(np.float64), int(0.35 * SAMPLE_RATE))
        if corr > 0.18:
            return "copy"
        return "mix"

    @staticmethod
    def _has_real_words(text):
        words = [w for w in normalized_words(text) if len(w) > 1 and w not in _BACKCHANNEL]
        return len(words) >= 1 and (len(words) >= 2 or len(words[0]) >= 4)

    def _maybe_live(self):
        sc = self.cfg["stt"]
        if not sc["live_transcript"] or self.silence_ms > 0 or self.utt_ms > 12000:
            return
        if (self.live_task and not self.live_task.done()) or self.eng.stt_jobs:
            return
        if now() - self.last_live < sc["live_interval_ms"] / 1000.0 or self.utt_ms < 800:
            return
        self.last_live = now()
        self.live_task = asyncio.create_task(self._live(self._utt_audio()))

    async def _live(self, audio):
        try:
            text = await self._stt(audio)
        except asyncio.CancelledError:
            return
        if self.in_speech and not self.verifying and text:
            log.debug("partial: %r", text)
            self.send({"type": "partial", "text": text})

    # --------------------------------------------------------------- turns

    def _new_turn(self, audio=None, text=None, speech_end=None):
        self.seq += 1
        turn = Turn(self.seq, audio=audio, text=text, speech_end=speech_end)
        turn.attachments = list(self.pending.values())
        turn.task = asyncio.create_task(self._run_turn(turn))
        return turn

    def _cancel_spec(self):
        if self.spec:
            log.debug("Speculative turn %d cancelled (user kept talking)", self.spec.id)
            self.spec.cancel()
            self.spec = None

    def _commit(self, turn):
        if self.turn and self.turn is not turn:
            if self.assistant_active():
                self._interrupt("new turn")
        self._unduck()
        self.turn = turn
        turn.metrics["commit_ms"] = round((now() - turn.speech_end) * 1000)
        turn.committed.set()
        self._refresh_state()

    def _interrupt(self, reason):
        t = self.turn
        if not t or not self.assistant_active():
            return
        t.interrupted = True
        heard = t.heard_text(now())
        t.cancel()
        self.send({"type": "stop", "turn": t.id})
        self.ducked = False
        log.info("Interrupted turn %d (%s); heard: %r", t.id, reason, heard[-80:])
        self._store_assistant(t, (heard + " —") if heard else "")

    def _store_assistant(self, turn, text):
        if turn.hist_idx is not None and turn.hist_idx < len(self.history):
            if text:
                self.history[turn.hist_idx]["content"] = text
            else:
                self.history.pop(turn.hist_idx)
                turn.hist_idx = None
        elif text:
            self.history.append({"role": "assistant", "content": text, "ts": _ts()})
            turn.hist_idx = len(self.history) - 1
        self._persist()
        if (text and not turn.interrupted and not self.titled and not self.incognito
                and not self._skip_persist and self.cfg["llm"].get("auto_title", True)):
            self.titled = True
            asyncio.create_task(self._auto_title(self.conv_id))

    def _persist(self):
        if not self.history or self.incognito or self._skip_persist:
            return
        if not self.conv_id:
            self.conv_id = self.eng.store.new_id()
        try:
            conv = self.eng.store.save(self.conv_id, self.history, model=self.eng.llm.model,
                                      labels=self.labels, project_id=self.project_id)
        except OSError as e:
            log.warning("Could not save the conversation: %s", e)
            return
        self.send({"type": "conv_saved", "id": conv["id"], "title": conv["title"],
                   "labels": conv.get("labels") or [], "project_id": conv.get("project_id")})

    async def _auto_title(self, cid):
        """Names a new conversation after its first exchange (in the background)."""
        convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in self.history[:2])
        prompt = [{"role": "system", "content": "You write very short titles for chat conversations."},
                  {"role": "user", "content": "Write a title of 2 to 6 words, in the language of the "
                   "conversation, for this exchange. Reply with the title only, no quotes.\n\n" + convo}]
        try:
            title = await self.eng.llm.complete(prompt, max_tokens=24, temperature=0.3)
        except Exception as e:
            log.debug("Auto title failed: %s", e)
            return
        title = title.strip().strip("\"'«»“”*#.").splitlines()[0].strip() if title.strip() else ""
        if not title or cid is None or cid != self.conv_id or self.incognito:
            return
        try:
            conv = self.eng.store.rename(cid, title[:80])
        except (KeyError, OSError, ValueError):
            return
        self.send({"type": "conv_saved", "id": cid, "title": conv["title"]})

    def _history_with_user(self, text, attachments=None):
        hist = [dict(m) for m in self.history]
        if hist and hist[-1]["role"] == "user":
            hist[-1]["content"] = (hist[-1]["content"] + " " + text).strip()
            if attachments:
                hist[-1]["attachments"] = list(hist[-1].get("attachments") or []) + list(attachments)
        else:
            m = {"role": "user", "content": text, "ts": _ts()}
            if attachments:
                m["attachments"] = list(attachments)
            hist.append(m)
        return hist

    def _locked_lang(self):
        """ISO language the model and TTS must stay in, or None to follow speech."""
        if self.output_lang == "voice":
            return voice_language(self.voice) if self.voice else None
        if self.output_lang in LANGUAGE_NAMES:
            return self.output_lang
        return None

    def _messages(self, hist):
        limit = int(self.gen.get("history") or self.cfg["llm"]["history_messages"])
        keep = hist[-limit:]
        while keep and keep[0]["role"] != "user":
            keep = keep[1:]
        system = (self.system_prompt or self.cfg["llm"]["system_prompt"]).rstrip()
        if MOOD_TAG_HINT not in system:
            system = system + "\n\n" + MOOD_TAG_HINT
        lock = self._locked_lang()
        if lock:
            name = LANGUAGE_NAMES.get(lock, lock)
            system += (
                f"\n\nAlways reply in {name} only. Do not switch languages "
                "even if the user writes in another language."
            )
        vision = bool(((self.eng.llm.info or {}).get("caps") or {}).get("vision"))
        out = [{"role": "system", "content": system}]
        for m in keep:
            content = m["content"]
            if m.get("attachments"):
                content = att.llm_content(content, m["attachments"], vision)
            out.append({"role": m["role"], "content": content})
        return out

    async def _run_turn(self, turn):
        try:
            if turn.user_text is None:
                t0 = now()
                turn.user_text = await self._stt(turn.audio)
                turn.metrics["stt_ms"] = round((now() - t0) * 1000)
            text = turn.user_text.strip()
            typed_with_files = turn.audio is None and turn.attachments
            if not typed_with_files and (not has_words(text) or text.lower().strip(".,!? ") in _NOISE_WORDS):
                await turn.committed.wait()
                log.info("Ignored empty/noise utterance %r", text)
                if self.turn is turn:
                    self.turn = None
                self.send({"type": "ignored", "text": text})
                self._refresh_state()
                return

            hist = self._history_with_user(text, turn.attachments)
            sender = asyncio.create_task(self._sender(turn))
            tts_q = asyncio.Queue()
            tts = asyncio.create_task(self._tts_worker(turn, tts_q))
            turn.subtasks += [sender, tts]

            messages = self._messages(hist)
            turn.user_lang = confident_language(text, self.langs)
            reply_lang = self._locked_lang() or (
                turn.user_lang if turn.user_lang and len(self.langs or ()) != 1 else None)
            if reply_lang:
                name = LANGUAGE_NAMES.get(reply_lang, reply_lang)
                last = messages[-1]["content"]
                if isinstance(last, list):
                    last = [dict(last[0], text=f"{last[0]['text']}\n\n(Reply in {name} only.)")] + last[1:]
                else:
                    last = f"{last}\n\n(Reply in {name} only.)"
                messages[-1] = {"role": "user", "content": last}

            tc = self.cfg["tts"]
            chunker = SentenceChunker(tc["first_chunk_min_chars"], tc["chunk_max_chars"],
                                      tc["first_chunk_wait_ms"] / 1000.0)
            spans = EmotionSpanFilter()
            mood = None

            def pump(visible, next_mood, t=None):
                nonlocal mood
                if not visible:
                    return
                if next_mood != mood:
                    for piece in chunker.flush():
                        tts_q.put_nowait((piece, mood))
                    mood = next_mood
                for piece in chunker.feed(visible, t):
                    turn.metrics.setdefault("text_ready_ms", round((now() - turn.speech_end) * 1000))
                    tts_q.put_nowait((piece, mood))

            t0 = now()
            async for delta in self.eng.llm.stream(messages, self.gen):
                if "ttft_ms" not in turn.metrics:
                    turn.metrics["ttft_ms"] = round((now() - t0) * 1000)
                turn.llm_raw += delta
                for next_mood, visible in spans.feed(delta):
                    if visible:
                        turn.emit({"type": "assistant_delta", "turn": turn.id, "text": visible})
                        pump(visible, next_mood, now())
            for next_mood, visible in spans.flush():
                if visible:
                    turn.emit({"type": "assistant_delta", "turn": turn.id, "text": visible})
                    pump(visible, next_mood, now())
            for piece in chunker.flush():
                tts_q.put_nowait((piece, mood))
            turn.metrics["llm_ms"] = round((now() - t0) * 1000)
            tts_q.put_nowait(None)
            await tts
            turn.emit({"type": "turn_done", "turn": turn.id, "metrics": turn.metrics})
            turn.emit(None)
            await sender
            turn.done = True
            if not turn.interrupted:
                stored = turn.llm_raw.strip() or " ".join(s for s, _, _ in turn.segments)
                self._store_assistant(turn, stored)
            log.info("Turn %d %s", turn.id, turn.metrics)
            self._refresh_state()
        except asyncio.CancelledError:
            raise
        except LLMError as e:
            await turn.committed.wait()
            log.error("LLM error: %s", e)
            self.send({"type": "error", "message": str(e)})
            turn.done = True
            self._refresh_state()
        except Exception as e:
            log.exception("Turn %d failed", turn.id)
            if turn.committed.is_set():
                self.send({"type": "error", "message": f"{type(e).__name__}: {e}"})
            turn.done = True

    async def _tts_worker(self, turn, q):
        loop = asyncio.get_running_loop()
        seg = 0
        while True:
            item = await q.get()
            if item is None:
                return
            chunk, mood = item if isinstance(item, tuple) else (item, None)
            speech = clean_for_speech(chunk)
            if not has_words(speech):
                continue
            if turn.lang is None:
                turn.lang = self._locked_lang() or detect_language(
                    speech, self.langs, default=turn.user_lang or self.lang)
                self.lang = turn.lang
            fx = dict(self.fx or {})
            if mood:
                fx["emotion"] = mood
            t0 = now()
            audio = await loop.run_in_executor(self.eng.tts_pool, self.eng.tts.speak, speech, turn.lang,
                                               self.voice, self.speed, self.gender, fx)
            if not len(audio):
                continue
            log.debug("turn %d seg %d: %.0f ms for %.2fs audio %s: %r", turn.id, seg,
                      (now() - t0) * 1000, len(audio) / TTS_SR, mood or "none", speech)
            if seg == 0:
                turn.metrics["tts_first_ms"] = round((now() - t0) * 1000)
            tail = speech.rstrip("\"')]»”")[-1:]
            pause = 0.22 if tail in ".!?…" else 0.08 if tail in ",;:—–" else 0.03
            audio = np.concatenate([audio, np.zeros(int(pause * TTS_SR), dtype=np.float32)])
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            turn.emit({"type": "seg", "turn": turn.id, "seg": seg, "text": speech,
                       "duration": round(len(audio) / TTS_SR, 3), "lang": turn.lang,
                       "emotion": mood or None})
            turn.emit((seg, speech, len(audio) / TTS_SR, struct.pack("<II", turn.id, seg) + pcm))
            seg += 1

    async def _sender(self, turn):
        """Releases a turn's output to the browser once it is committed."""
        await turn.committed.wait()
        if turn.cancelled:
            return
        self.history = self._history_with_user(turn.user_text.strip(), turn.attachments)
        for a in turn.attachments:
            self.pending.pop(str(a.get("id")), None)
        self._persist()
        log.info("User (turn %d): %r%s", turn.id, turn.user_text.strip(),
                 f" + {len(turn.attachments)} attachment(s)" if turn.attachments else "")
        self.send({"type": "user", "turn": turn.id, "text": turn.user_text.strip(),
                   "stt_ms": turn.metrics.get("stt_ms"),
                   "attachments": [att.public(a) for a in turn.attachments]})
        self._refresh_state()
        while True:
            item = await turn.outbox.get()
            if item is None:
                return
            if isinstance(item, tuple):
                seg, text, dur, payload = item
                t = now()
                if "first_audio_ms" not in turn.metrics:
                    turn.metrics["first_audio_ms"] = round((t - turn.speech_end) * 1000)
                start = max(t + 0.05, turn.play_until)
                turn.segments.append((text, start, dur))
                turn.play_until = start + dur
                self.send(payload)
                self._refresh_state()
            else:
                self.send(item)
