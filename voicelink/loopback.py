"""What this PC is playing, captured with WASAPI loopback (soundcard).

Used as a reference so that speech coming out of the speakers (a video, a
call, music, and also our own replies) is not taken as the user talking.
One capture thread is shared by every session and runs only while at least
one session has the "ignore sound playing on this PC" filter on.
"""
import logging
import threading
import time
import warnings

import numpy as np

log = logging.getLogger("loopback")

RATE = 16000
KEEP_S = 30
BLOCK = 800          # 50 ms
QUIET = 10 ** (-55 / 20)


def speakers():
    """[{name, default}] for the output devices loopback can capture."""
    try:
        import soundcard as sc

        default = sc.default_speaker().name
        return [{"name": s.name, "default": s.name == default} for s in sc.all_speakers()]
    except Exception as e:  # no audio stack, unusual drivers
        log.warning("Cannot list output devices: %s", e)
        return []


class Loopback:
    def __init__(self):
        self.lock = threading.Lock()
        self.users = {}              # session id -> wanted device name ("" = default)
        self.device = None
        self.thread = None
        self.stop_flag = threading.Event()
        self.buf = np.zeros(RATE * KEEP_S, dtype=np.float32)
        self.written = 0             # total samples captured
        self.t_end = 0.0             # perf_counter time of the newest sample
        self.error = None

    # -------------------------------------------------------------- control

    def acquire(self, owner, device=""):
        with self.lock:
            self.users[owner] = device or ""
            self._update()

    def release(self, owner):
        with self.lock:
            self.users.pop(owner, None)
            self._update()

    def _update(self):
        want = next(iter(self.users.values()), None) if self.users else None
        if want == self.device and (want is None or (self.thread and self.thread.is_alive())):
            return
        self._stop()
        self.device = want
        if want is not None:
            self.stop_flag.clear()
            self.thread = threading.Thread(target=self._run, args=(want,), name="loopback", daemon=True)
            self.thread.start()

    def _stop(self):
        if self.thread:
            self.stop_flag.set()
            self.thread.join(timeout=2)
            self.thread = None

    def close(self):
        with self.lock:
            self.users.clear()
            self._stop()
            self.device = None

    def _run(self, name):
        try:
            import soundcard as sc

            spk = next((s for s in sc.all_speakers() if s.name == name), None) if name else None
            spk = spk or sc.default_speaker()
            mic = sc.get_microphone(spk.id, include_loopback=True)
            log.info("Listening to what the PC plays on: %s", spk.name)
            self.error = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "data discontinuity" when the device idles
                with mic.recorder(samplerate=RATE, channels=1, blocksize=BLOCK) as rec:
                    while not self.stop_flag.is_set():
                        x = rec.record(numframes=BLOCK)[:, 0].astype(np.float32)
                        self._push(x)
        except Exception as e:
            self.error = str(e) or type(e).__name__
            log.warning("Loopback capture failed: %s", self.error)

    def _push(self, x):
        n = len(x)
        pos = self.written % len(self.buf)
        end = pos + n
        if end <= len(self.buf):
            self.buf[pos:end] = x
        else:
            k = len(self.buf) - pos
            self.buf[pos:] = x[:k]
            self.buf[:n - k] = x[k:]
        self.written += n
        self.t_end = time.perf_counter()

    # --------------------------------------------------------------- access

    @property
    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def window(self, t0, t1):
        """Samples played between perf_counter times t0 and t1 (clipped to what is kept)."""
        if not self.running or not self.written:
            return np.zeros(0, dtype=np.float32)
        newest = self.written
        oldest = max(0, newest - len(self.buf) + RATE)   # keep a margin from the writer
        a = int(newest - (self.t_end - t0) * RATE)
        b = int(newest - (self.t_end - t1) * RATE)
        a, b = max(a, oldest), min(b, newest)
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        idx = np.arange(a, b) % len(self.buf)
        return self.buf[idx]

    def playing(self, seconds=1.5):
        """True if something audible came out of the speakers recently."""
        t = time.perf_counter()
        x = self.window(t - seconds, t)
        return len(x) > 0 and float(np.sqrt(np.mean(x * x))) > QUIET
