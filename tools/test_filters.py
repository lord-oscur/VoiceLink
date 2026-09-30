"""In-process test of the microphone filters (no browser, no speakers, no LLM).

Drives a real Session with synthesized speech in real time and checks which
utterances become user turns:
  * playback filter: a "video" that is also in the (simulated) loopback feed
    must be ignored; the user talking after or over it must not be;
  * "Only my voice": after enrolling one voice, other voices are ignored.

    python tools/test_filters.py
"""
import asyncio
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from selftest import make_voice  # noqa: E402
from voicelink import session as sess_mod  # noqa: E402
from voicelink.config import load_config  # noqa: E402
from voicelink.server import Engines  # noqa: E402
from voicelink.session import Session, Turn  # noqa: E402

FRAME = 512
RATE = 16000
LOOP_LEAD = 0.12   # loopback hears the video this long before the microphone does


class FakeWS:
    closed = False

    def __init__(self):
        self.msgs = []

    async def send_str(self, s):
        import json
        self.msgs.append(json.loads(s))

    async def send_bytes(self, b):
        pass


class FakeLoopback:
    """Plays `signal` starting at perf_counter time `t0`."""

    def __init__(self):
        self.signal = np.zeros(0, dtype=np.float32)
        self.t0 = 0.0
        self.error = None

    def play(self, x):
        self.signal = x
        self.t0 = time.perf_counter()

    def window(self, t0, t1):
        a = max(0, int((t0 - self.t0) * RATE))
        b = min(len(self.signal), int((t1 - self.t0) * RATE))
        return self.signal[a:b] if b > a else np.zeros(0, dtype=np.float32)

    def playing(self, seconds=1.5):
        t = time.perf_counter()
        x = self.window(t - seconds, t)
        return len(x) > 0 and float(np.sqrt(np.mean(x * x))) > 10 ** (-55 / 20)

    def acquire(self, *a):
        pass

    def release(self, *a):
        pass

    def close(self):
        pass


class Probe(Session):
    """Records committed turns instead of running the LLM."""

    def __init__(self, ws, eng):
        self.committed = []
        super().__init__(ws, eng)

    def _new_turn(self, audio=None, text=None, speech_end=None):
        self.seq += 1
        return Turn(self.seq, audio=audio, text=text, speech_end=speech_end)

    def _commit(self, turn):
        self.committed.append(turn)
        self._refresh_state()


async def stream(s, x):
    """Real-time microphone stream."""
    t = time.perf_counter()
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    for i in range(0, len(pcm), FRAME):
        s.on_audio(pcm[i:i + FRAME].tobytes())
        t += FRAME / RATE
        await asyncio.sleep(max(0.0, t - time.perf_counter()))


def silence(sec):
    return np.zeros(int(sec * RATE), dtype=np.float32)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-7s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    eng = Engines(cfg)
    eng.loopback = FakeLoopback()
    say = make_voice()
    f32 = lambda text, voice, lang="en-us": say(text, voice, lang).astype(np.float32) / 32767  # noqa: E731
    rng = np.random.default_rng(3)
    room = lambda n: (rng.standard_normal(n) * 0.002).astype(np.float32)  # noqa: E731

    video = f32("And that is why the rocket had to be rebuilt from scratch before the second launch window.", "bf_emma")
    video2 = f32("Scientists say the new telescope will change how we see the early universe.", "em_alex", "es")
    user = f32("Hey, can you tell me what time it is in Tokyo right now?", "am_michael")
    user2 = f32("Please add milk and eggs to my shopping list.", "am_michael")
    other = f32("Could you turn the music down a little bit please?", "af_heart")

    results = []

    def check(name, ok, detail=""):
        results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    async def run(s, mic, loop=None, wait=2.0):
        before = len(s.committed)
        if loop is not None:
            # The video starts playing slightly before the microphone hears it.
            eng.loopback.play(np.concatenate([loop, silence(3)]))
            await asyncio.sleep(LOOP_LEAD)
        await stream(s, np.concatenate([silence(0.3), mic, silence(wait)]) + room(len(mic) + int((wait + 0.3) * RATE)))
        return len(s.committed) - before

    ws = FakeWS()
    s = Probe(ws, eng)
    await s.on_json({"type": "config", "noise": "light"})

    # 1. Baseline
    n = await run(s, user)
    check("no filter: user speech becomes a turn", n == 1, f"turns={n}")
    n = await run(s, 0.5 * video)
    check("no filter: a video is (wrongly) taken as a turn", n == 1, f"turns={n}")

    # 2. Playback filter
    await s.on_json({"type": "config", "playback_gate": True})
    n = await run(s, 0.5 * video, loop=video)
    check("playback filter: video alone is ignored", n == 0, f"turns={n}")
    n = await run(s, 0.4 * video2, loop=video2)
    check("playback filter: another video is ignored", n == 0, f"turns={n}")
    mix = np.concatenate([0.5 * video, silence(1.2), user])
    n = await run(s, mix, loop=np.concatenate([video, silence(1.2 + len(user) / RATE)]))
    check("playback filter: user speaking right after the video is heard", n >= 1, f"turns={n}")
    over = user.copy()
    over[:len(video2[:len(user)])] += 0.3 * video2[:len(user)]
    n = await run(s, over, loop=video2[:len(user)])
    check("playback filter: user speaking over a quieter video is heard", n >= 1, f"turns={n}")
    eng.loopback.play(np.zeros(0, dtype=np.float32))
    n = await run(s, user2)
    check("playback filter: nothing playing -> normal turn", n == 1, f"turns={n}")
    await s.on_json({"type": "config", "playback_gate": False})

    # 3. Only my voice
    enroll = np.concatenate([f32(t, "am_michael") for t in (
        "I am recording my voice so the assistant can recognise me.",
        "When I speak, it should answer; when a video or another person talks, it should stay quiet.",
        "Today is a good day to try something new, and tomorrow I will tell you how it went.",
        "One, two, three, four, five.")])
    await s.on_json({"type": "enroll_start"})
    await stream(s, np.concatenate([enroll, silence(1.0)]))
    for _ in range(50):
        if any(m["type"] == "enrolled" for m in ws.msgs):
            break
        await asyncio.sleep(0.1)
    vp = next((m["voiceprint"] for m in ws.msgs if m["type"] == "enrolled"), None)
    check("only me: enrollment produced a voice print", vp is not None,
          f"dim={len(vp) if vp else 0}, progress msgs={sum(m['type'] == 'enroll_progress' for m in ws.msgs)}")
    await s.on_json({"type": "config", "only_me": True, "only_me_strictness": 50, "voiceprint": vp})

    def last_score():
        m = [x for x in ws.msgs if x["type"] == "speaker"]
        return f"score={m[-1]['score']:.2f}" if m else "no score"

    n = await run(s, user2)
    check("only me: my voice is accepted", n == 1, f"turns={n} {last_score()}")
    n = await run(s, other)
    check("only me: another person is ignored", n == 0, f"turns={n} {last_score()}")
    n = await run(s, 0.5 * video)
    check("only me: a video voice is ignored", n == 0, f"turns={n} {last_score()}")
    n = await run(s, 0.5 * video2)
    check("only me: a male video voice is ignored", n == 0, f"turns={n} {last_score()}")
    short = f32("Yes, do it.", "am_michael")
    n = await run(s, short)
    check("only me: a short reply of mine is accepted", n == 1, f"turns={n} {last_score()}")

    await s.close()
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
