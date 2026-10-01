"""HTTP + WebSocket server: serves the browser UI and hosts voice sessions."""
import asyncio
import io
import json
import logging
import mimetypes
import os
import socket
import time
import wave
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
from aiohttp import web

from . import __version__, loopback
from .config import ROOT, load_config, model_path
from .history import ConversationStore, ProjectStore
from . import hfsearch
from .llm import LLMError, LMStudioClient
from .loopback import Loopback
from .session import Session
from .speaker import SpeakerID

log = logging.getLogger("server")
WEB_DIR = os.path.join(ROOT, "web")
# Some Windows registries map .js to text/plain, which browsers refuse for modules.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

PREVIEW_TEXT = {
    "en": "Hi! This is how I sound. Shall we talk for a while?",
    "es": "¡Hola! Así suena mi voz. ¿Charlamos un rato?",
    "fr": "Bonjour ! Voici ma voix. On discute un peu ?",
    "it": "Ciao! Questa è la mia voce. Parliamo un po'?",
    "pt": "Olá! Esta é a minha voz. Vamos conversar um pouco?",
    "hi": "नमस्ते! मेरी आवाज़ ऐसी है। क्या हम थोड़ी बात करें?",
    "ja": "こんにちは！これが私の声です。少しお話ししませんか？",
    "zh": "你好！这就是我的声音。我们聊一会儿吧？",
}


def wav_bytes(audio, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


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
        self.file_pool = ThreadPoolExecutor(1, thread_name_prefix="file-stt")  # attachments; keeps live STT free
        self.stt = SpeechToText(cfg["stt"])
        self.tts = TextToSpeech(cfg["tts"])
        self.store = ConversationStore()
        self.projects = ProjectStore()
        self.loopback = Loopback()
        self.speaker = SpeakerID()
        self.llm = None


async def index(request):
    return web.FileResponse(os.path.join(WEB_DIR, "index.html"),
                            headers={"Cache-Control": "no-store"})


async def favicon(request):
    return web.FileResponse(os.path.join(WEB_DIR, "favicon.ico"))


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


async def llm_defaults(request):
    c = request.app["engines"].cfg["llm"]
    return web.json_response({k: c.get(k) for k in (
        "system_prompt", "temperature", "top_p", "top_k", "repeat_penalty", "max_tokens",
        "history_messages", "disable_thinking")})


async def models_list(request):
    llm = request.app["engines"].llm
    try:
        models = await llm.models()
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e) or type(e).__name__, "models": []})
    models = [m for m in models if m["type"] in ("llm", "vlm", None)]
    return web.json_response({"ok": True, "current": llm.model, "models": models})


async def models_load(request):
    llm = request.app["engines"].llm
    body = await request.json()
    if not body.get("key"):
        raise web.HTTPBadRequest(text="key is required")
    log.info("Loading model %s (context %s, flash attention %s, kv gpu %s)...", body["key"],
             body.get("context_length"), body.get("flash_attention"),
             body.get("offload_kv_cache_to_gpu"))
    t = time.perf_counter()
    try:
        result = await llm.load(
            body["key"], body.get("context_length"), body.get("flash_attention"),
            offload_kv_cache_to_gpu=body.get("offload_kv_cache_to_gpu"),
            k_cache=body.get("k_cache") or None,
            v_cache=body.get("v_cache") or None,
        )
    except (LLMError, aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("Model load failed: %s", e)
        return web.json_response({"ok": False, "error": str(e) or type(e).__name__})
    log.info("Model %s loaded in %.1fs", llm.model, time.perf_counter() - t)
    (result.get("load_config") or {}).pop("prompt_template", None)
    return web.json_response({"ok": True, "result": result, "status": await llm.status()})


async def models_hf(request):
    q = (request.query.get("q") or "").strip()
    ignore = request.query.get("ignore") in ("1", "true", "yes")
    vram = request.query.get("vram")
    try:
        vram = float(vram) if vram else None
    except ValueError:
        vram = None
    if vram is None and not ignore:
        vram = hfsearch.vram_gb()
    try:
        data = await hfsearch.search(request.app["http"], q, vram=vram, ignore=ignore)
    except Exception as e:
        log.warning("Hugging Face search failed: %s", e)
        return web.json_response({"ok": False, "error": str(e) or type(e).__name__,
                                  "results": [], "vram_gb": vram})
    return web.json_response(data)


async def models_download(request):
    body = await request.json()
    repo = (body.get("repo") or body.get("model") or "").strip()
    quant = (body.get("quant") or body.get("quantization") or "").strip() or None
    if not repo:
        raise web.HTTPBadRequest(text="repo is required")
    if not repo.startswith("http"):
        repo = "https://huggingface.co/" + repo.lstrip("/")
    log.info("LM Studio download %s %s", repo, quant or "")
    try:
        result = await request.app["engines"].llm.download(repo, quant)
    except (LLMError, aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("Model download failed to start: %s", e)
        return web.json_response({"ok": False, "error": str(e) or type(e).__name__})
    return web.json_response({"ok": True, **result})


async def models_download_status(request):
    job_id = request.match_info["job_id"]
    if not job_id or any(c in job_id for c in "/\\ ?#"):
        raise web.HTTPBadRequest(text="bad job id")
    try:
        result = await request.app["engines"].llm.download_status(job_id)
    except (LLMError, aiohttp.ClientError, asyncio.TimeoutError) as e:
        return web.json_response({"ok": False, "error": str(e) or type(e).__name__})
    return web.json_response({"ok": True, **result})


async def hardware(request):
    vram = hfsearch.vram_gb()
    return web.json_response({"vram_gb": vram, "gpu": bool(vram)})


async def conv_list(request):
    return web.json_response(request.app["engines"].store.list())


async def conv_get(request):
    try:
        return web.json_response(request.app["engines"].store.get(request.match_info["cid"]))
    except (KeyError, OSError, ValueError):
        raise web.HTTPNotFound()


async def conv_patch(request):
    body = await request.json()
    fields = {}
    if "title" in body:
        fields["title"] = body.get("title")
    if "labels" in body:
        fields["labels"] = body.get("labels")
    if "project_id" in body:
        fields["project_id"] = body.get("project_id")
    try:
        conv = request.app["engines"].store.update(request.match_info["cid"], **fields)
    except (KeyError, OSError, ValueError):
        raise web.HTTPNotFound()
    return web.json_response({"id": conv["id"], "title": conv.get("title"),
                              "labels": conv.get("labels") or [],
                              "project_id": conv.get("project_id")})


async def project_list(request):
    return web.json_response(request.app["engines"].projects.list())


async def project_create(request):
    body = await request.json()
    try:
        return web.json_response(request.app["engines"].projects.create(body.get("name")), status=201)
    except ValueError as e:
        raise web.HTTPBadRequest(text=str(e))


async def project_patch(request):
    body = await request.json()
    try:
        return web.json_response(request.app["engines"].projects.rename(
            request.match_info["pid"], body.get("name")))
    except KeyError:
        raise web.HTTPNotFound()
    except ValueError as e:
        raise web.HTTPBadRequest(text=str(e))


async def project_delete(request):
    pid = request.match_info["pid"]
    try:
        request.app["engines"].projects.delete(pid)
    except KeyError:
        raise web.HTTPNotFound()
    request.app["engines"].store.clear_project(pid)
    return web.json_response({"ok": True})


async def conv_delete(request):
    try:
        request.app["engines"].store.delete(request.match_info["cid"])
    except (KeyError, OSError):
        raise web.HTTPNotFound()
    return web.json_response({"ok": True})


async def extract(request):
    """Document -> text. Body: the raw file; ?name= gives its file name."""
    from .attachments import UnsupportedFile, extract_text

    name = request.query.get("name", "file.txt")
    data = await request.read()
    loop = asyncio.get_running_loop()
    try:
        text, truncated = await loop.run_in_executor(None, extract_text, name, data)
    except UnsupportedFile as e:
        return web.json_response({"ok": False, "error": str(e)})
    except Exception as e:
        log.warning("Could not read %s: %s", name, e)
        return web.json_response({"ok": False, "error": f"could not read the file ({type(e).__name__})"})
    return web.json_response({"ok": True, "text": text, "truncated": truncated, "chars": len(text)})


async def transcribe(request):
    """Audio file (already decoded by the browser) -> text. Body: int16 mono 16 kHz PCM."""
    from .attachments import split_long_audio

    eng = request.app["engines"]
    data = await request.read()
    audio = np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    duration = len(audio) / 16000
    if duration > eng.cfg["stt"].get("max_file_minutes", 30) * 60:
        return web.json_response({"ok": False, "error": "the recording is too long"})
    loop = asyncio.get_running_loop()
    t = time.perf_counter()
    parts = []
    for piece in split_long_audio(audio):
        if np.sqrt(np.mean(piece ** 2)) < 1e-4:
            continue
        parts.append(await loop.run_in_executor(eng.file_pool, eng.stt.transcribe, piece))
    text = " ".join(p.strip() for p in parts if p and p.strip())
    log.info("Transcribed %.0f s of attached audio in %.1f s", duration, time.perf_counter() - t)
    return web.json_response({"ok": True, "text": text, "duration": round(duration, 2)})


async def check_text(request):
    from . import checker

    body = await request.json()
    text = (body.get("text") or "")[:20000]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, checker.check, text, body.get("lang") or "auto",
                                        tuple(body.get("ignore") or ()))
    return web.json_response(result)


async def check_ai(request):
    """Asks the loaded model for a corrected version and returns the differences."""
    from . import checker

    body = await request.json()
    text = (body.get("text") or "")[:6000]
    if not text.strip():
        return web.json_response({"ok": True, "issues": []})
    lang = checker.detect(text, body.get("lang") or "auto")
    name = "Spanish" if lang == "es" else "English"
    prompt = [
        {"role": "system", "content": "You are a meticulous proofreader. You fix spelling, grammar, punctuation "
         "and accents, and remove redundant words, while keeping the author's wording, tone and language."},
        {"role": "user", "content": f"Correct this {name} text. Change as little as possible. Reply with the "
         f"corrected text only - no quotes, no explanations.\n\n{text}"},
    ]
    llm = request.app["engines"].llm
    try:
        fixed = await llm.complete(prompt, temperature=0.1, max_tokens=max(64, len(text) // 2 + 64))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e) or type(e).__name__})
    fixed = fixed.strip().strip('"“”')
    if not fixed or len(fixed) > len(text) * 2 + 40:
        return web.json_response({"ok": False, "error": "the model did not return a usable correction"})
    return web.json_response({"ok": True, "lang": lang, "corrected": fixed,
                              "issues": checker.ai_issues(text, fixed, lang)})


async def voices(request):
    tts = request.app["engines"].tts
    return web.json_response({"voices": tts.catalog, "voice_map": tts.voice_map,
                              "male_map": tts.male_map, "speed": tts.speed})


async def preview(request):
    """Short sample of a voice (with the current speed and effects) as WAV."""
    from .tts import SAMPLE_RATE, voice_language

    eng = request.app["engines"]
    body = await request.json()
    voice = body.get("voice") or None
    lang = body.get("lang") or (voice_language(voice) if voice else "en")
    text = (body.get("text") or PREVIEW_TEXT.get(lang) or PREVIEW_TEXT["en"])[:300]
    speed = float(body["speed"]) if body.get("speed") else None
    loop = asyncio.get_running_loop()
    audio = await loop.run_in_executor(eng.tts_pool, eng.tts.speak, text, lang, voice, speed,
                                       body.get("gender"), body.get("fx"))
    return web.Response(body=wav_bytes(audio, SAMPLE_RATE), content_type="audio/wav")


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024, heartbeat=20)  # attached images
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
    from . import checker
    asyncio.get_running_loop().run_in_executor(None, checker.warm)


async def on_cleanup(app):
    await app["http"].close()
    app["engines"].loopback.close()


async def loopback_devices(request):
    eng = request.app["engines"]
    loop = asyncio.get_running_loop()
    return web.json_response({"devices": await loop.run_in_executor(None, loopback.speakers),
                              "speaker_model": eng.speaker.available(),
                              "error": eng.loopback.error})


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

    app = web.Application(client_max_size=256 * 1024 * 1024)  # attachments
    app["engines"] = engines
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.on_response_prepare.append(revalidate_static)
    app.router.add_get("/", index)
    app.router.add_get("/favicon.ico", favicon)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/voices", voices)
    app.router.add_post("/api/preview", preview)
    app.router.add_get("/api/loopback", loopback_devices)
    app.router.add_post("/api/check", check_text)
    app.router.add_post("/api/check_ai", check_ai)
    app.router.add_post("/api/extract", extract)
    app.router.add_post("/api/transcribe", transcribe)
    app.router.add_get("/api/llm_defaults", llm_defaults)
    app.router.add_get("/api/models", models_list)
    app.router.add_post("/api/models/load", models_load)
    app.router.add_get("/api/models/hf", models_hf)
    app.router.add_post("/api/models/download", models_download)
    app.router.add_get("/api/models/download/{job_id}", models_download_status)
    app.router.add_get("/api/hardware", hardware)
    app.router.add_get("/api/conversations", conv_list)
    app.router.add_get("/api/conversations/{cid}", conv_get)
    app.router.add_patch("/api/conversations/{cid}", conv_patch)
    app.router.add_delete("/api/conversations/{cid}", conv_delete)
    app.router.add_get("/api/projects", project_list)
    app.router.add_post("/api/projects", project_create)
    app.router.add_patch("/api/projects/{pid}", project_patch)
    app.router.add_delete("/api/projects/{pid}", project_delete)
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
