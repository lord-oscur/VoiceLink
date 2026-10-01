"""Streaming client for LM Studio's OpenAI-compatible API."""
import json
import logging
import re

import aiohttp

from .textproc import ThinkFilter

log = logging.getLogger("llm")

_PREFILL = {"role": "assistant", "content": "<think>\n\n</think>\n\n"}


class LLMError(Exception):
    pass


def _parse_caps(caps):
    """Icons only when LM Studio actually flags the capability (not an empty {})."""
    if not isinstance(caps, dict):
        caps = {}
    raw = caps.get("reasoning")
    reasoning = None
    if isinstance(raw, dict) and (raw.get("allowed_options") or raw.get("default") or raw.get("status")):
        reasoning = raw
    elif raw is True:
        reasoning = {"allowed_options": ["on"]}
    return {
        "vision": bool(caps.get("vision")),
        "tools": caps.get("trained_for_tool_use") is True,
        "reasoning": reasoning,
    }


def _attention_from_config(cfg):
    """Standard vs flash, plus any other attention keys LM Studio reports."""
    kinds = ["flash" if cfg.get("flash_attention") else "standard"]
    for k, v in (cfg or {}).items():
        kl = k.lower()
        if "attention" in kl and kl != "flash_attention" and v not in (None, "", False):
            kinds.append(f"{k}={v}" if not isinstance(v, bool) else k)
    return kinds


# Raw llama.cpp / LM Studio architecture → family shown in the picker.
_ARCH_FAMILY = (
    ("qwen", "Qwen"), ("llama", "Llama"), ("gemma", "Gemma"), ("phi", "Phi"),
    ("lfm", "LFM"), ("granite", "Granite"), ("nemotron", "Nemotron"),
    ("gpt", "GPT"), ("mistral", "Mistral"), ("mixtral", "Mistral"),
)


def _families(blob, arch):
    """Model families for the picker (Qwen, Llama, Ternary, Gemma, Opus, …)."""
    out = []
    if re.search(r"ternary|bitnet|\bbonsai\b", blob):
        out.append("Ternary")
    if re.search(r"\bopus\b", blob):
        out.append("Opus")
    arch_l = (arch or "").lower()
    hit = False
    for prefix, fam in _ARCH_FAMILY:
        if arch_l.startswith(prefix) or re.search(r"\b" + re.escape(prefix), blob):
            if fam not in out:
                out.append(fam)
            hit = True
            break
    if not hit and arch:
        pretty = arch.replace("_", " ").replace("-", " ").strip().title()
        if pretty and pretty not in out:
            out.append(pretty)
    return out


def _infer_flags(m):
    key = m.get("key") or m.get("id") or ""
    name = m.get("display_name") or key
    arch = (m.get("architecture") or "")
    quant = m.get("quantization")
    quant_s = quant.get("name") if isinstance(quant, dict) else (quant or "")
    blob = f"{key} {name} {arch} {quant_s}".lower()
    experts = m.get("num_experts") or m.get("n_experts") or m.get("expert_count")
    if experts is None:
        mx = re.search(r"(\d+)\s*x\s*[\d.]+", blob)
        if mx:
            experts = int(mx.group(1))
    try:
        experts = int(experts) if experts is not None else None
    except (TypeError, ValueError):
        experts = None
    moe = bool(experts and experts > 1) or "moe" in (arch or "").lower() or bool(
        re.search(r"\bmoe\b|mixtral|a3b|a22b|\be[24]b\b", blob))
    mtp = bool(re.search(r"\bmtp\b", blob))
    families = _families(blob, arch)
    return {
        "architecture": arch or None,
        "family": families[0] if families else None,
        "families": families,
        "experts": experts,
        "moe": moe,
        "mtp": mtp,
        "kind": "mtp" if mtp else ("moe" if moe else "dense"),
        "nsfw": bool(re.search(r"uncensored|nsfw|\+?18\+|erotica", blob, re.I)),
        "abliterated": "abliterated" in blob,
    }


def _model_v1(m):
    instances = []
    for i in m.get("loaded_instances") or []:
        cfg = i.get("config") or {}
        instances.append({
            "id": i["id"], "context_length": cfg.get("context_length"),
            "flash_attention": cfg.get("flash_attention"),
            "offload_kv_cache_to_gpu": cfg.get("offload_kv_cache_to_gpu"),
            "attention": _attention_from_config(cfg),
        })
    quant = m.get("quantization")
    row = {
        "key": m["key"], "name": m.get("display_name") or m["key"], "type": m.get("type"),
        "params": m.get("params_string"), "quant": quant.get("name") if isinstance(quant, dict) else quant,
        "size": m.get("size_bytes"), "max_context": m.get("max_context_length"),
        "loaded": bool(instances), "instances": instances,
        "caps": _parse_caps(m.get("capabilities")),
    }
    row.update(_infer_flags(m))
    return row


def _model_v0(m):
    caps = m.get("capabilities") or []
    loaded = m.get("state") == "loaded"
    row = {
        "key": m["id"], "name": m["id"], "type": m.get("type"), "params": None, "quant": m.get("quantization"),
        "size": None, "max_context": m.get("max_context_length"), "loaded": loaded,
        "instances": [{"id": m["id"], "context_length": m.get("loaded_context_length"),
                       "flash_attention": None, "offload_kv_cache_to_gpu": None,
                       "attention": []}] if loaded else [],
        "caps": {"vision": m.get("type") == "vlm", "tools": "tool_use" in caps, "reasoning": None},
    }
    row.update(_infer_flags(m))
    return row


class LMStudioClient:
    def __init__(self, cfg, http: aiohttp.ClientSession):
        self.cfg = cfg
        self.http = http
        self.base = cfg["base_url"].rstrip("/")
        self.model = None
        self.info = None   # capabilities of the model in use (from status())
        # "effort" -> send reasoning_effort=none; "prefill" -> empty think block;
        # "off" -> let the model think (reasoning is never spoken).
        self.no_think = "effort" if cfg["disable_thinking"] else "off"

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.cfg['api_key']}", "Content-Type": "application/json"}

    async def _get_json(self, path):
        async with self.http.get(self.base + path, headers=self.headers,
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
            r.raise_for_status()
            return await r.json()

    async def models(self):
        """Downloaded models with capabilities and loaded instances (LM Studio REST v1)."""
        try:
            data = (await self._get_json("/api/v1/models"))["models"]
            return [_model_v1(m) for m in data]
        except Exception as e:
            if isinstance(e, aiohttp.ClientConnectorError):
                raise
            data = (await self._get_json("/api/v0/models"))["data"]  # older LM Studio
            return [_model_v0(m) for m in data]

    async def resolve_model(self):
        """Returns the model id to use (configured, or the one loaded in LM Studio)."""
        wanted = self.cfg["model"]
        if wanted and wanted != "auto":
            self.model = wanted
            return wanted
        try:
            loaded = [m for m in await self.models() if m["loaded"] and m["type"] in ("llm", "vlm")]
            ids = [i["id"] for m in loaded for i in m["instances"]]
            if ids:
                # Keep the current model while it is loaded; asking for an unloaded
                # one would make LM Studio load it again (JIT) with default settings.
                if self.model not in ids:
                    self.model = ids[0]
                return self.model
        except aiohttp.ClientConnectorError:
            raise
        except Exception:
            pass
        data = (await self._get_json("/v1/models"))["data"]
        llms = [m for m in data if "embed" not in m["id"].lower()]
        if not llms:
            raise LLMError("LM Studio reports no models; load one first")
        self.model = llms[0]["id"]
        return self.model

    async def status(self):
        try:
            model = await self.resolve_model()
            info = None
            try:
                info = next((m for m in await self.models()
                             if any(i["id"] == model for i in m["instances"]) or m["key"] == model), None)
            except Exception:
                pass
            if info:
                self.info = info
            return {"ok": True, "model": model, "base_url": self.base, "info": info}
        except Exception as e:
            return {"ok": False, "error": str(e) or type(e).__name__, "base_url": self.base}

    async def load(self, key, context_length=None, flash_attention=None,
                   offload_kv_cache_to_gpu=None, k_cache=None, v_cache=None,
                   unload_others=True):
        """Loads `key` with the given settings, replacing loaded LLM instances."""
        if unload_others:
            for m in await self.models():
                for inst in m["instances"]:
                    if m["type"] in ("llm", "vlm"):
                        await self._post_json("/api/v1/models/unload", {"instance_id": inst["id"]})
        body = {"model": key, "echo_load_config": True}
        if context_length:
            body["context_length"] = int(context_length)
        if flash_attention is not None:
            body["flash_attention"] = bool(flash_attention)
        if offload_kv_cache_to_gpu is not None:
            body["offload_kv_cache_to_gpu"] = bool(offload_kv_cache_to_gpu)
        extra = {}
        if k_cache:
            extra["llama_k_cache_quantization_type"] = k_cache
        if v_cache and flash_attention:
            extra["llama_v_cache_quantization_type"] = v_cache
        result = await self._load_with_fallback(body, extra)
        self.model = result.get("instance_id") or key
        self.info = None
        return result

    async def _load_with_fallback(self, body, extra):
        """REST often rejects K/V quant keys; retry without them if so."""
        try:
            return await self._post_json("/api/v1/models/load", {**body, **extra}, timeout=600)
        except LLMError as e:
            text = str(e).lower()
            if extra and ("unrecognized" in text or "unknown" in text or "unexpected" in text):
                log.info("LM Studio rejected K/V cache types; loading without them")
                return await self._post_json("/api/v1/models/load", body, timeout=600)
            raise

    async def download(self, model, quantization=None):
        """Ask LM Studio to pull a catalog id or Hugging Face GGUF repo."""
        body = {"model": model}
        if quantization:
            body["quantization"] = quantization
        return await self._post_json("/api/v1/models/download", body, timeout=60)

    async def download_status(self, job_id):
        return await self._get_json("/api/v1/models/download/status/" + job_id)

    async def _post_json(self, path, body, timeout=30):
        async with self.http.post(self.base + path, json=body, headers=self.headers,
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            text = await r.text()
            if r.status != 200:
                raise LLMError(f"LM Studio HTTP {r.status}: {text[:300]}")
            return json.loads(text) if text else {}

    def can_tools(self):
        caps = ((self.info or {}).get("caps") or {})
        return bool(caps.get("tools"))

    def _body(self, messages, gen, tools=None):
        think = gen.get("thinking", not self.cfg["disable_thinking"])
        prefill = not think and self.no_think == "prefill" and not tools
        body = {
            "model": self.model,
            "messages": messages + ([_PREFILL] if prefill else []),
            "stream": True,
        }
        for key in ("temperature", "top_p", "top_k", "repeat_penalty", "max_tokens"):
            value = gen.get(key, self.cfg.get(key))
            if value is not None:
                body[key] = value
        if not think and self.no_think == "effort":
            body["reasoning_effort"] = "none"
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    async def complete(self, messages, **gen):
        """Non-streaming helper for short tasks (titles, grammar check)."""
        text = ""
        async for delta in self.stream(messages, dict(gen, thinking=False)):
            text += delta
        return text.strip()

    async def stream(self, messages, gen=None):
        """Yields spoken-content text deltas. Reasoning is never yielded."""
        gen = gen or {}
        if not self.model:
            await self.resolve_model()
        for _ in range(2):
            mode = self.no_think
            got_content = False
            thinking = gen.get("thinking", not self.cfg["disable_thinking"])
            agen = self._stream_once(messages, gen, None)
            try:
                async for kind, text in agen:
                    if kind == "reasoning" and mode == "effort" and not got_content and not thinking:
                        log.info("Model ignores reasoning_effort; switching to empty-think prefill")
                        self.no_think = "prefill"
                        break
                    if kind == "content":
                        got_content = True
                        yield text
            finally:
                await agen.aclose()
            if self.no_think == mode:
                return

    async def generate(self, messages, gen=None, tools=None, execute=None):
        """Yield spoken content. When `tools` and `execute` are set, run tool calls (never spoken)."""
        from .mcp import MAX_ROUNDS

        gen = gen or {}
        msgs = list(messages)
        for rnd in range(MAX_ROUNDS):
            calls = {}
            async for kind, payload in self._stream_once(msgs, gen, tools if tools else None):
                if kind == "content":
                    yield payload
                elif kind == "tool_call":
                    _merge_tool_delta(calls, payload)
            pending = [calls[i] for i in sorted(calls) if calls[i].get("name")]
            if not pending or not execute:
                return
            assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": c.get("id") or "call_%d_%d" % (rnd, i),
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c.get("arguments") or "{}"},
                } for i, c in enumerate(pending)],
            }
            msgs.append(assistant)
            for tc in assistant["tool_calls"]:
                name = tc["function"]["name"]
                raw = tc["function"]["arguments"]
                try:
                    args = json.loads(raw) if raw else {}
                except ValueError:
                    args = {"_raw": raw}
                result = await execute(name, args)
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result or ""})
        log.warning("Stopped after %d tool rounds", MAX_ROUNDS)

    async def _stream_once(self, messages, gen, tools=None):
        think = ThinkFilter()
        url = self.base + "/v1/chat/completions"
        body = self._body(messages, gen, tools)
        try:
            async with self.http.post(url, json=body, headers=self.headers,
                                      timeout=aiohttp.ClientTimeout(total=None, sock_connect=5,
                                                                    sock_read=120)) as r:
                if r.status != 200:
                    detail = (await r.text())[:300]
                    if r.status == 400 and "reasoning_effort" in body and "reasoning" in detail.lower():
                        log.info("Server rejected reasoning_effort; switching to prefill")
                        self.no_think = "prefill"
                        return
                    raise LLMError(f"LM Studio HTTP {r.status}: {detail}")
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get("error"):
                        raise LLMError(str(chunk["error"]))
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("reasoning_content") or delta.get("reasoning"):
                        yield "reasoning", delta.get("reasoning_content") or delta.get("reasoning")
                    content = delta.get("content")
                    if content:
                        visible = think.feed(content)
                        if visible:
                            yield "content", visible
                    for piece in delta.get("tool_calls") or []:
                        yield "tool_call", piece
                tail = think.flush()
                if tail:
                    yield "content", tail
        except aiohttp.ClientConnectorError as e:
            raise LLMError(f"Cannot reach LM Studio at {self.base} ({e.os_error or e})") from e


def _merge_tool_delta(bag, piece):
    """Assemble streamed OpenAI tool_call fragments by index."""
    if not isinstance(piece, dict):
        return
    idx = piece.get("index", 0)
    slot = bag.setdefault(idx, {"id": "", "name": "", "arguments": ""})
    if piece.get("id"):
        slot["id"] = piece["id"]
    fn = piece.get("function") or {}
    if fn.get("name"):
        slot["name"] += fn["name"]
    if fn.get("arguments"):
        slot["arguments"] += fn["arguments"]
