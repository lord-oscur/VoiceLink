"""Streaming Silero VAD (v5/v6 ONNX) - one instance per audio stream."""
import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
FRAME = 512          # 32 ms at 16 kHz, the window size the model expects
_CONTEXT = 64


class SileroVAD:
    _session = None

    def __init__(self, model_file):
        if SileroVAD._session is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            SileroVAD._session = ort.InferenceSession(model_file, sess_options=so,
                                                      providers=["CPUExecutionProvider"])
        self.sess = SileroVAD._session
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame):
        """frame: float32 array of exactly FRAME samples. Returns speech probability."""
        x = np.concatenate([self._context, frame.reshape(1, -1)], axis=1)
        out, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -_CONTEXT:]
        return float(out[0, 0])
