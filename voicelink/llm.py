"""Streaming client for LM Studio's OpenAI-compatible API."""
import json
import logging

import aiohttp

from .textproc import ThinkFilter

log = logging.getLogger("llm")

_PREFILL = {"role": "assistant", "content": "<think>\n\n</think>\n\n"}


class LLMError(Exception):
    pass


def _model_v1(m):
    caps = m.get("capabilities") or {}
    reasoning = caps.get("reasoning") or None
    instances = [{"id": i["id"], "context_length": (i.get("config") or {}).get("context_length"),
                  "flash_attention": (i.get("config") or {}).get("flash_attention")}
                 for i in m.get("loaded_instances") or []]
    quant = m.get("quantization")
    return {
        "key": m["key"], "name": m.get("display_name") or m["key"], "type": m.get("type"),
        "params": m.get("params_string"), "quant": quant.get("name") if isinstance(quant, dict) else quant,
        "size": m.get("size_bytes"), "max_context": m.get("max_context_length"),
        "loaded": bool(instances), "instances": instances,
        "caps": {"vision": bool(caps.get("vision")), "tools": bool(caps.get("trained_for_tool_use")),
                 "reasoning": reasoning},
    }


def _model_v0(m):
    caps = m.get("capabilities") or []
    loaded = m.get("state") == "loaded"
    return {
        "key": m["id"], "name": m["id"], "type": m.get("type"), "params": None, "quant": m.get("quantization"),
        "size": None, "max_context": m.get("max_context_length"), "loaded": loaded,
        "instances": [{"id": m["id"], "context_length": m.get("loaded_context_length"),
                       "flash_attention": None}] if loaded else [],
        "caps": {"vision": m.get("type") == "vlm", "tools": "tool_use" in caps, "reasoning": None},
    }


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

    async def load(self, key, context_length=None, flash_attention=None, unload_others=True):
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
        result = await self._post_json("/api/v1/models/load", body, timeout=600)
        self.model = result.get("instance_id") or key
        self.info = None
        return result

    async def _post_json(self, path, body, timeout=30):
        async with self.http.post(self.base + path, json=body, headers=self.headers,
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            text = await r.text()
            if r.status != 200:
                raise LLMError(f"LM Studio HTTP {r.status}: {text[:300]}")
            return json.loads(text) if text else {}

    def _body(self, messages, gen):
        think = gen.get("thinking", not self.cfg["disable_thinking"])
        prefill = not think and self.no_think == "prefill"
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
            agen = self._stream_once(messages, gen)
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

    async def _stream_once(self, messages, gen):
        think = ThinkFilter()
        url = self.base + "/v1/chat/completions"
        body = self._body(messages, gen)
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
                tail = think.flush()
                if tail:
                    yield "content", tail
        except aiohttp.ClientConnectorError as e:
            raise LLMError(f"Cannot reach LM Studio at {self.base} ({e.os_error or e})") from e
