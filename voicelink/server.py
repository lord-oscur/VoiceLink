"""HTTP + WebSocket server: serves the browser UI and hosts voice sessions."""
import asyncio
import json
import logging
import os
import socket
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from aiohttp import web

from . import __version__
from .config import ROOT, load_config, model_path
from .llm import LMStudioClient
from .session import Session

log = logging.getLogger("server")
WEB_DIR = os.path.join(ROOT, "web")


class Engines:
    """Models shared by every session (loaded once)."""

    def __init__(self, cfg):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        from .stt import SpeechToText
        from .tts import TextToSpeech

        self.cfg = cfg
        self.vad_file = model_path("vad/silero_vad.onnx")
        missing = [p for p in (self.vad_file, model_path(cfg["tts"]["model_file"]),
                               model_path(cfg["tts"]["voices_file"])) if not os.path.exists(p)]
        if missing:
            raise SystemExit("Missing model files: %s\nRun setup.bat first." % ", ".join(missing))
        self.stt_pool = ThreadPoolExecutor(1, thread_name_prefix="stt")
        self.stt_jobs = 0
        self.tts_pool = ThreadPoolExecutor(1, thread_name_prefix="tts")
        self.stt = SpeechToText(cfg["stt"])
        self.tts = TextToSpeech(cfg["tts"])
        self.llm = None


async def index(request):
    return web.FileResponse(os.path.join(WEB_DIR, "index.html"),
                            headers={"Cache-Control": "no-store"})


async def revalidate_static(request, response):
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"


async def status(request):
    eng = request.app["engines"]
    llm = await eng.llm.status()
    return web.json_response({
        "version": __version__,
        "llm": llm,
        "voices": eng.tts.voices,
        "voice_map": eng.tts.voice_map,
        "speed": eng.tts.speed,
        "languages": eng.cfg["stt"]["languages"],
    })


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)
    eng = request.app["engines"]
    session = Session(ws, eng)
    peer = request.remote
    log.info("Client connected (%s)", peer)
    llm = await eng.llm.status()
    session.send({"type": "ready", "llm": llm, "voices": eng.tts.voices,
                  "voice_map": eng.tts.voice_map, "speed": eng.tts.speed})
    session._refresh_state()
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                session.on_audio(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    await session.on_json(json.loads(msg.data))
                except (ValueError, KeyError, TypeError) as e:
                    log.warning("Bad client message %r: %s", msg.data[:200], e)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        await session.close()
        log.info("Client disconnected (%s)", peer)
    return ws


async def on_startup(app):
    eng = app["engines"]
    app["http"] = aiohttp.ClientSession()
    eng.llm = LMStudioClient(eng.cfg["llm"], app["http"])
    st = await eng.llm.status()
    if st["ok"]:
        log.info("LM Studio OK at %s - model: %s", st["base_url"], st["model"])
    else:
        log.warning("LM Studio not reachable at %s (%s). Start its server; the UI will retry.",
                    st["base_url"], st["error"])


async def on_cleanup(app):
    await app["http"].close()


def main():
    debug = bool(os.environ.get("VOICELINK_DEBUG"))
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("aiohttp.access", "asyncio", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    cfg = load_config()
    host, port = cfg["server"]["host"], int(cfg["server"]["port"])
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    want_browser = cfg["server"]["open_browser"] and not os.environ.get("VOICELINK_NO_BROWSER")
    if not _port_free(host, port):
        log.error("Port %d is already in use - VoiceLink is probably already running (%s).", port, url)
        if want_browser:
            _open(url)
        raise SystemExit(1)

    t = time.perf_counter()
    log.info("VoiceLink %s - loading speech models...", __version__)
    engines = Engines(cfg)
    log.info("Models loaded in %.1fs", time.perf_counter() - t)

    app = web.Application()
    app["engines"] = engines
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.on_response_prepare.append(revalidate_static)
    app.router.add_get("/", index)
    app.router.add_get("/api/status", status)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", WEB_DIR)

    async def open_browser(app):
        if want_browser:
            asyncio.get_running_loop().call_later(0.5, _open, url)

    app.on_startup.append(open_browser)
    log.info("Open %s  (Ctrl+C to stop)", url)
    web.run_app(app, host=host, port=port, print=None, access_log=None)


def _port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host if host not in ("", "0.0.0.0") else "0.0.0.0", port))
            return True
        except OSError:
            return False


def _open(url):
    chrome = [os.path.expandvars(p) for p in (
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
    )]
    for exe in chrome:
        if os.path.exists(exe):
            webbrowser.register("chrome", None, webbrowser.BackgroundBrowser(exe))
            webbrowser.get("chrome").open(url)
            return
    webbrowser.open(url)
