"""End-to-end test of a running VoiceLink server, without a browser or microphone.

Synthesizes user speech with Kokoro, streams it over the WebSocket in real
time (like a microphone), and checks turn-taking, latency and barge-in.

    python tools/selftest.py                 # all scenarios
    python tools/selftest.py --only=3,5      # just some
    python tools/selftest.py ws://host:port/ws
"""
import asyncio
import json
import os
import struct
import sys
import time

import aiohttp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voicelink.config import load_config, model_path  # noqa: E402

_args = [a for a in sys.argv[1:] if not a.startswith("--only=")]
ONLY = {int(x) for a in sys.argv[1:] if a.startswith("--only=") for x in a[7:].split(",") if x}
URL = _args[0] if _args else "ws://127.0.0.1:%d/ws" % load_config()["server"]["port"]
FRAME = 512


def make_voice():
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model_path("kokoro/kokoro-v1.0.onnx"), sess_options=so,
                                providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(sess, model_path("kokoro/voices-v1.0.bin"))

    def say(text, voice="am_michael", lang="en-us"):
        audio, sr = k.create(text, voice=voice, lang=lang)
        idx = np.round(np.arange(0, len(audio), sr / 16000)).astype(int)
        pcm = audio[idx[idx < len(audio)]]
        return (np.clip(pcm, -1, 1) * 32767 * 0.8).astype("<i2")
    return say


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.events = []
        self.audio = {}
        self.t0 = time.perf_counter()
        self.first_audio = {}
        self.echo_gain = 0.0          # >0: feed received speech back into the "mic"
        self.echo = np.zeros(0, dtype=np.float32)

    def t(self):
        return time.perf_counter() - self.t0

    async def reader(self):
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                turn, seg = struct.unpack("<II", msg.data[:8])
                self.audio.setdefault(turn, 0)
                self.audio[turn] += (len(msg.data) - 8) / 2 / 24000
                self.first_audio.setdefault(turn, self.t())
                if self.echo_gain:
                    pcm = np.frombuffer(msg.data[8:], dtype="<i2").astype(np.float32)
                    idx = np.arange(0, len(pcm), 1.5).astype(int)
                    self.echo = np.concatenate([self.echo, pcm[idx] * self.echo_gain])
            else:
                m = json.loads(msg.data)
                self.events.append((self.t(), m))
                if m["type"] == "stop":
                    self.echo = np.zeros(0, dtype=np.float32)
                if m["type"] in ("user", "stop", "error", "ignored", "turn_done"):
                    print(f"  [{self.t():6.2f}s] {m['type']:9} " +
                          json.dumps({k: v for k, v in m.items() if k != "type"}, ensure_ascii=False)[:150])

    async def stream(self, pcm):
        # Paced against the wall clock: asyncio.sleep on Windows has ~15 ms
        # granularity, so sleeping a fixed 32 ms per frame would drift.
        t_next = time.perf_counter()
        for i in range(0, len(pcm), FRAME):
            frame = pcm[i:i + FRAME]
            if len(self.echo):
                e, self.echo = self.echo[:len(frame)], self.echo[len(frame):]
                mixed = frame.astype(np.float32)
                mixed[:len(e)] += e
                frame = np.clip(mixed, -32768, 32767).astype("<i2")
            await self.ws.send_bytes(frame.tobytes())
            t_next += FRAME / 16000
            delay = t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

    async def silence(self, seconds):
        await self.stream(np.zeros(int(seconds * 16000), dtype="<i2"))

    def wait_for(self, kind, after=0.0, **match):
        for t, m in self.events:
            if t >= after and m["type"] == kind and all(m.get(k) == v for k, v in match.items()):
                return t, m
        return None

    async def until(self, kind, after, timeout):
        for _ in range(int(timeout / 0.1)):
            if self.wait_for(kind, after=after):
                return True
            await self.silence(0.1)
        return False

    async def until_audio(self, after):
        while not self.first_audio or max(self.first_audio.values()) < after:
            await self.silence(0.1)


async def main():
    print("Synthesizing test utterances...")
    say = make_voice()
    q1 = say("Hi there! Can you give me one fun fact about octopuses?")
    q2 = say("Hola, ¿qué me recomiendas cenar esta noche?", "em_alex", "es")
    q3 = say("Tell me a long story about a dragon who learns to cook.")
    q5 = say("Tell me a long story about a lighthouse keeper and his cat.")
    barge = say("Wait, stop. Just tell me a short joke instead.", "am_adam")
    results = {}

    async with aiohttp.ClientSession() as http:
        for attempt in range(120):
            try:
                ws = await http.ws_connect(URL)
                break
            except aiohttp.ClientError:
                if attempt == 0:
                    print(f"Waiting for the server at {URL} ...")
                await asyncio.sleep(1)
        else:
            sys.exit(f"Could not connect to {URL}")
        c = Client(ws)
        reader = asyncio.create_task(c.reader())

        async def config(**kw):
            await ws.send_str(json.dumps({"type": "config", **kw}))

        async def turn(pcm, label):
            start = c.t()
            await c.stream(pcm)
            speech_end = c.t()
            await c.until("turn_done", start, 30)
            done, user = c.wait_for("turn_done", after=start), c.wait_for("user", after=start)
            if not (done and user):
                print(f"  !! {label}: no complete turn")
                return None
            tid = done[1]["turn"]
            v2v = c.first_audio.get(tid, 0) - speech_end
            print(f"  => {label}: heard {user[1]['text']!r}; voice-to-voice {v2v:.2f}s; "
                  f"{c.audio.get(tid, 0):.1f}s of reply audio; metrics {done[1]['metrics']}")
            results[label] = v2v
            await c.silence(c.audio.get(tid, 0) + 0.5)
            return tid

        async def scenario1():
            print("\n1) English question")
            await config(headphones=True)
            await turn(q1, "english")

        async def scenario2():
            print("\n2) Spanish question (voice should switch to Spanish)")
            await config(headphones=True)
            tid = await turn(q2, "spanish")
            seg = [m for _, m in c.events if m["type"] == "seg" and m.get("turn") == tid]
            print("     reply language:", seg[0]["lang"] if seg else "?")

        async def scenario3():
            print("\n3) Barge-in (headphones): interrupt a long answer")
            await config(headphones=True)
            start = c.t()
            await c.stream(q3)
            await c.until_audio(start)
            await c.silence(1.5)
            t_barge = c.t()
            await c.stream(barge)
            stop_evt = c.wait_for("stop", after=t_barge)
            if stop_evt:
                print(f"  => stop received {stop_evt[0] - t_barge:.2f}s after the user started talking")
                results["barge_in_s"] = stop_evt[0] - t_barge
            else:
                print("  !! no stop event")
            await c.until("turn_done", t_barge, 25)
            u = c.wait_for("user", after=t_barge)
            print("  => new user turn:", repr(u[1]["text"]) if u else None)
            await c.silence(2)

        async def scenario4():
            print("\n4) 3 s of silence must not create a turn")
            t4 = c.t()
            await c.silence(3)
            print("  =>", "OK" if not c.wait_for("user", after=t4) else "FAIL: spurious turn")

        async def scenario5():
            print("\n5) Speakers, no echo cancellation: the assistant's voice leaks into the mic")
            await config(headphones=False)
            c.echo_gain = 0.35
            t5 = c.t()
            await c.stream(q5)
            await c.until_audio(t5)
            await c.silence(4)
            false_stop = c.wait_for("stop", after=t5)
            print("  => echo alone:", "FAIL: assistant interrupted itself" if false_stop
                  else "OK, no self-interruption")
            results["echo_self_interrupt"] = 1.0 if false_stop else 0.0
            if not len(c.echo):
                print("  !! reply already finished; barge-in over echo not tested")
                c.echo_gain = 0.0
                return
            t_barge = c.t()
            await c.stream(barge)
            await c.until("stop", t_barge, 4)
            stop_evt = c.wait_for("stop", after=t_barge)
            print("  => real barge-in over echo:",
                  f"stop after {stop_evt[0] - t_barge:.2f}s" if stop_evt else "FAIL: not interrupted")
            if stop_evt:
                results["barge_in_echo_s"] = stop_evt[0] - t_barge
            c.echo_gain = 0.0
            await c.until("turn_done", t_barge, 20)

        for n, fn in enumerate((scenario1, scenario2, scenario3, scenario4, scenario5), 1):
            if not ONLY or n in ONLY:
                await fn()
        reader.cancel()
        await ws.close()
    print("\nSummary:", {k: round(v, 2) for k, v in results.items()})


if __name__ == "__main__":
    asyncio.run(main())
