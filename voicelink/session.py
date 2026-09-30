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

from .llm import LLMError
from .textproc import (LANGUAGE_NAMES, SentenceChunker, clean_for_speech, confident_language, detect_language,
                       has_words, normalized_words)
from .tts import SAMPLE_RATE as TTS_SR
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


def now():
    return time.perf_counter()


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
        self.langs = self.cfg["stt"]["languages"] or None
        self.lang = (self.langs or ["en"])[0]
        self.headphones = False
        self.voice = None
        self.gender = None
        self.fx = None
        self.speed = None
        self.mic_on = True
        self.pcm = np.zeros(0, dtype=np.float32)
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
        elif kind == "text":
            text = (msg.get("text") or "").strip()
            if text:
                self._interrupt("typed message")
                self._cancel_spec()
                turn = self._new_turn(text=text)
                self._commit(turn)
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
            self.history.clear()
            self.send({"type": "reset_done"})
            self._refresh_state()

    def on_audio(self, data):
        if not self.mic_on:
            return
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) * (1.0 / 32768.0)
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
        if assistant_active:
            if self.headphones or not self.vc["echo_check"]:
                self._interrupt("barge-in")
            else:
                self.verifying = True
                self.verify_task = asyncio.create_task(self._verify_barge_in())
        self._refresh_state()

    def _speech_end(self):
        speech_end = now() - self.silence_ms / 1000.0
        turn = self.spec or self._new_turn(audio=self._utt_audio(), speech_end=speech_end)
        self.spec = None
        self._reset_speech()
        self._commit(turn)

    def _barge_in_verdict(self, text, trimmed=False):
        """True = real interruption, None = not sure yet (echo/backchannel)."""
        words = normalized_words(text)
        if trimmed:
            words = words[1:]  # the window may start mid-word
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

    async def _verify_barge_in(self):
        """Echo check: keep transcribing the candidate speech as it grows and
        only interrupt for genuine new words.  Echo keeps the VAD busy for
        long stretches, so once the audio exceeds verify_max_ms the window
        slides forward instead of giving up; otherwise words the user says
        in the middle of the echo would be thrown away."""
        vc = self.vc
        keep = max(1, int(_VERIFY_KEEP_MS / FRAME_MS))
        trimmed = False
        try:
            while True:
                heard_ms = self.utt_ms
                text = await self._stt(self._utt_audio())
                if not self.in_speech:
                    return
                verdict = self._barge_in_verdict(text, trimmed)
                log.debug("echo check %.0f ms -> %s: %r", heard_ms, verdict, text)
                if verdict or self.end_pending:
                    break
                step = 500 if trimmed else 250  # long echo: leave CPU for the TTS
                while self.in_speech and not self.end_pending and self.utt_ms < heard_ms + step:
                    await asyncio.sleep(0.03)
                if not self.in_speech:
                    return
                if self.utt_ms >= vc["verify_max_ms"] and not self.end_pending:
                    self.utt = self.utt[-keep:]
                    self.utt_ms = len(self.utt) * FRAME_MS
                    trimmed = True
        except asyncio.CancelledError:
            return
        if not verdict:
            log.info("Barge-in ignored (echo/backchannel): %r", text)
            self._reset_speech()
            self._unduck()
            self._refresh_state()
            return
        log.info("Barge-in confirmed: %r", text)
        self.verifying = False
        self._interrupt("barge-in")
        if self.end_pending:
            self._speech_end()
        self._refresh_state()

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
        if turn.hist_idx is not None:
            if text:
                self.history[turn.hist_idx]["content"] = text
            else:
                self.history.pop(turn.hist_idx)
                turn.hist_idx = None
        elif text:
            self.history.append({"role": "assistant", "content": text})
            turn.hist_idx = len(self.history) - 1

    def _history_with_user(self, text):
        hist = [dict(m) for m in self.history]
        if hist and hist[-1]["role"] == "user":
            hist[-1]["content"] += " " + text
        else:
            hist.append({"role": "user", "content": text})
        return hist

    def _messages(self, hist):
        keep = hist[-int(self.cfg["llm"]["history_messages"]):]
        while keep and keep[0]["role"] != "user":
            keep = keep[1:]
        return [{"role": "system", "content": self.cfg["llm"]["system_prompt"]}] + keep

    async def _run_turn(self, turn):
        try:
            if turn.user_text is None:
                t0 = now()
                turn.user_text = await self._stt(turn.audio)
                turn.metrics["stt_ms"] = round((now() - t0) * 1000)
            text = turn.user_text.strip()
            if not has_words(text) or text.lower().strip(".,!? ") in _NOISE_WORDS:
                await turn.committed.wait()
                log.info("Ignored empty/noise utterance %r", text)
                if self.turn is turn:
                    self.turn = None
                self.send({"type": "ignored", "text": text})
                self._refresh_state()
                return

            hist = self._history_with_user(text)
            sender = asyncio.create_task(self._sender(turn))
            tts_q = asyncio.Queue()
            tts = asyncio.create_task(self._tts_worker(turn, tts_q))
            turn.subtasks += [sender, tts]

            messages = self._messages(hist)
            turn.user_lang = confident_language(text, self.langs)
            if turn.user_lang and len(self.langs or ()) != 1:
                name = LANGUAGE_NAMES.get(turn.user_lang, turn.user_lang)
                messages[-1] = {"role": "user", "content": f"{messages[-1]['content']}\n\n(Reply in {name}.)"}

            tc = self.cfg["tts"]
            chunker = SentenceChunker(tc["first_chunk_min_chars"], tc["chunk_max_chars"],
                                      tc["first_chunk_wait_ms"] / 1000.0)
            t0 = now()
            async for delta in self.eng.llm.stream(messages):
                if "ttft_ms" not in turn.metrics:
                    turn.metrics["ttft_ms"] = round((now() - t0) * 1000)
                turn.emit({"type": "assistant_delta", "turn": turn.id, "text": delta})
                for chunk in chunker.feed(delta, now()):
                    turn.metrics.setdefault("text_ready_ms", round((now() - turn.speech_end) * 1000))
                    tts_q.put_nowait(chunk)
            for chunk in chunker.flush():
                tts_q.put_nowait(chunk)
            turn.metrics["llm_ms"] = round((now() - t0) * 1000)
            tts_q.put_nowait(None)
            await tts
            turn.emit({"type": "turn_done", "turn": turn.id, "metrics": turn.metrics})
            turn.emit(None)
            await sender
            turn.done = True
            if not turn.interrupted:
                self._store_assistant(turn, " ".join(s for s, _, _ in turn.segments))
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
            chunk = await q.get()
            if chunk is None:
                return
            speech = clean_for_speech(chunk)
            if not has_words(speech):
                continue
            if turn.lang is None:
                turn.lang = detect_language(speech, self.langs, default=turn.user_lang or self.lang)
                self.lang = turn.lang
            t0 = now()
            audio = await loop.run_in_executor(self.eng.tts_pool, self.eng.tts.speak, speech, turn.lang,
                                               self.voice, self.speed, self.gender, self.fx)
            if not len(audio):
                continue
            log.debug("turn %d seg %d: %.0f ms for %.2fs audio: %r", turn.id, seg,
                      (now() - t0) * 1000, len(audio) / TTS_SR, speech)
            if seg == 0:
                turn.metrics["tts_first_ms"] = round((now() - t0) * 1000)
            tail = speech.rstrip("\"')]»”")[-1:]
            pause = 0.22 if tail in ".!?…" else 0.08 if tail in ",;:—–" else 0.03
            audio = np.concatenate([audio, np.zeros(int(pause * TTS_SR), dtype=np.float32)])
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            turn.emit({"type": "seg", "turn": turn.id, "seg": seg, "text": speech,
                       "duration": round(len(audio) / TTS_SR, 3), "lang": turn.lang})
            turn.emit((seg, speech, len(audio) / TTS_SR, struct.pack("<II", turn.id, seg) + pcm))
            seg += 1

    async def _sender(self, turn):
        """Releases a turn's output to the browser once it is committed."""
        await turn.committed.wait()
        if turn.cancelled:
            return
        self.history = self._history_with_user(turn.user_text.strip())
        log.info("User (turn %d): %r", turn.id, turn.user_text.strip())
        self.send({"type": "user", "turn": turn.id, "text": turn.user_text.strip(),
                   "stt_ms": turn.metrics.get("stt_ms")})
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
