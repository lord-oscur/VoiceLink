"""Execution-provider selection for ONNX Runtime.

"cpu" is the default for every component: in the target setup LM Studio
already owns the GPU, and sharing it slows token generation. "cuda" only
works with the optional onnxruntime-gpu install (setup.bat gpu)."""
import logging

import onnxruntime as ort

log = logging.getLogger("gpu")
_preloaded = False


def _preload():
    global _preloaded
    if not _preloaded:
        _preloaded = True
        try:
            ort.preload_dlls()
        except Exception as e:  # CPU-only build or missing CUDA wheels
            log.debug("preload_dlls: %s", e)


def providers_for(device, cudnn_heuristic=False):
    device = (device or "cpu").lower()
    if device in ("cuda", "gpu", "auto") and "CUDAExecutionProvider" in ort.get_available_providers():
        _preload()
        opts = {"cudnn_conv_algo_search": "HEURISTIC"} if cudnn_heuristic else {}
        return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
    if device in ("cuda", "gpu"):
        log.warning("CUDA requested but onnxruntime-gpu is not installed; using CPU")
    return ["CPUExecutionProvider"]
