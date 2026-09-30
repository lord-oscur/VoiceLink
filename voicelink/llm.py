"""Streaming client for LM Studio's OpenAI-compatible API."""
import json
import logging

import aiohttp

from .textproc import ThinkFilter

log = logging.getLogger("llm")

_PREFILL = {"role": "assistant", "content": "<think>\n\n</think>\n\n"}


class LLMError(Exception):
    pass


class LMStudioClient:
    def __init__(self, cfg, http: aiohttp.ClientSession):
        self.cfg = cfg
        self.http = http
        self.base = cfg["base_url"].rstrip("/")
        self.model = None
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

    async def resolve_model(self):
        """Returns the model id to use (configured, or the one loaded in LM Studio)."""
        wanted = self.cfg["model"]
        if wanted and wanted != "auto":
            self.model = wanted
            return wanted
        try:
            data = (await self._get_json("/api/v0/models"))["data"]
            loaded = [m for m in data if m.get("state") == "loaded" and m.get("type") in ("llm", "vlm")]
            if loaded:
                self.model = loaded[0]["id"]
                return self.model
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
            return {"ok": True, "model": model, "base_url": self.base}
        except Exception as e:
            return {"ok": False, "error": str(e) or type(e).__name__, "base_url": self.base}

    def _body(self, messages):
        body = {
            "model": self.model,
            "messages": messages + ([_PREFILL] if self.no_think == "prefill" else []),
            "stream": True,
            "temperature": self.cfg["temperature"],
            "top_p": self.cfg["top_p"],
            "max_tokens": self.cfg["max_tokens"],
        }
        if self.no_think == "effort":
            body["reasoning_effort"] = "none"
        return body

    async def stream(self, messages):
        """Yields spoken-content text deltas. Reasoning is never yielded."""
        if not self.model:
            await self.resolve_model()
        for _ in range(2):
            mode = self.no_think
            got_content = False
            agen = self._stream_once(messages)
            try:
                async for kind, text in agen:
                    if kind == "reasoning" and mode == "effort" and not got_content:
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

    async def _stream_once(self, messages):
        think = ThinkFilter()
        url = self.base + "/v1/chat/completions"
        try:
            async with self.http.post(url, json=self._body(messages), headers=self.headers,
                                      timeout=aiohttp.ClientTimeout(total=None, sock_connect=5,
                                                                    sock_read=120)) as r:
                if r.status != 200:
                    detail = (await r.text())[:300]
                    if r.status == 400 and self.no_think == "effort":
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
