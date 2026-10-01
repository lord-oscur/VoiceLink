"""Stateless MCP (Streamable HTTP): local tools + remote HTTP servers.

Only the new protocol is accepted: one POST per JSON-RPC call, no stdio,
no HTTP+SSE sessions, no Mcp-Session-Id. Remote servers that mint a session
are skipped. See SEP-2575 / MCP Streamable HTTP in stateless mode.

Built-in tools (no API keys): web_search, fetch_page, weather, stock_quote.
"""
import html as htmlmod
import json
import logging
import re
import time
from datetime import datetime
from urllib.parse import quote, urlparse

import aiohttp

from . import __version__

log = logging.getLogger("mcp")

PROTOCOL = "2025-11-25"
SUPPORTED = ("2025-11-25", "2025-06-18", "2025-03-26")
CLIENT_INFO = {"name": "voicelink", "version": __version__}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
MAX_RESULT = 6000
MAX_ROUNDS = 4
_UNSAFE = re.compile(r"^(file|javascript|data):", re.I)
_TAG = re.compile(r"<[^>]+>", re.S)
_WS = re.compile(r"[ \t]+\n|\n{3,}|[ \t]{2,}")


def clock_line(now=None):
    now = now or datetime.now().astimezone()
    return (
        f"The current local date and time is {now.strftime('%A, %B %d, %Y at %I:%M %p')} "
        f"({now.strftime('%Z')}, ISO {now.isoformat(timespec='minutes')}). "
        "Your training data ends earlier than this. Treat that gap as time that has "
        "already passed; do not guess what happened after you were trained."
    )


TOOLS_HINT = (
    "You can look things up on the public web. Use that whenever you are "
    "unsure, your training may be stale or missing, you would otherwise say "
    "you do not know, or you would be guessing or pretending to know. "
    "That includes people, places, products, how-to steps, quotes, "
    "definitions, history, science, software, culture, and anything that "
    "changes. Weather, stock prices, news, scores, hours and rates are "
    "examples, not the only cases. "
    "Do not invent facts, names, numbers, dates, quotes or URLs. "
    "Call web_search for a general lookup. "
    "Call weather when the user wants conditions or a forecast for a place. "
    "Call stock_quote for a ticker price (for example AAPL or MSFT). "
    "Call fetch_page when you need the text of a specific URL from search results. "
    "After a tool returns, answer in spoken sentences. If a tool fails, say "
    "so and do not fill the gap with a guess."
)


def openai_tools(catalog):
    """MCP tool list → OpenAI chat-completions tools."""
    out = []
    for t in catalog:
        params = t.get("inputSchema") or {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") or t["name"],
                "parameters": params,
            },
        })
    return out


def reject_legacy(spec):
    """Raise ValueError if the server entry is stdio, SSE, or command-based."""
    if not isinstance(spec, dict):
        raise ValueError("MCP server entry must be an object")
    if spec.get("command") or spec.get("args"):
        raise ValueError("stdio / command MCP servers are not supported (legacy)")
    url = (spec.get("url") or "").strip()
    if spec.get("name") == "local" or url in ("", "local"):
        return
    if not url.startswith(("http://", "https://")):
        raise ValueError("MCP url must be http(s) Streamable HTTP, got %r" % url)
    path = urlparse(url).path.lower()
    if path.endswith("/sse") or "/sse/" in path:
        raise ValueError("HTTP+SSE MCP (%s) is the 2024 transport; use /mcp" % url)


# ------------------------------------------------------------------ tools

def _plain(text, limit=MAX_RESULT):
    text = htmlmod.unescape(_TAG.sub(" ", text or ""))
    text = _WS.sub("\n", text).strip()
    return text[:limit]


def _private_host(host):
    host = (host or "").lower().strip("[]")
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local") or host.endswith(".localhost"):
        return True
    if host.startswith("10.") or host.startswith("192.168.") or host.startswith("169.254."):
        return True
    if host.startswith("172."):
        try:
            n = int(host.split(".")[1])
        except (IndexError, ValueError):
            return False
        return 16 <= n <= 31
    return False


def _safe_url(url):
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")) or _UNSAFE.match(url):
        return None
    host = (urlparse(url).hostname or "").lower()
    if _private_host(host):
        return None
    return url


async def _get(http, url, **kw):
    headers = {"User-Agent": UA, "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"}
    headers.update(kw.pop("headers", {}) or {})
    timeout = aiohttp.ClientTimeout(total=kw.pop("timeout", 12))
    async with http.get(url, headers=headers, timeout=timeout, allow_redirects=True, **kw) as r:
        return r.status, await r.text()


async def tool_web_search(http, query, count=5):
    query = (query or "").strip()
    if not query:
        return "No search query."
    count = max(1, min(int(count or 5), 8))
    rows = await _ddg_html(http, query, count)
    if not rows:
        rows = await _ddg_api(http, query, count)
    if not rows:
        return "Search returned no results (the search site may be blocking this PC)."
    lines = [f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(rows, 1)]
    return "Search results for %r:\n\n%s" % (query, "\n\n".join(lines))


async def _ddg_html(http, query, count):
    try:
        data = aiohttp.FormData()
        data.add_field("q", query)
        data.add_field("kl", "us-en")
        headers = {"User-Agent": UA, "Referer": "https://html.duckduckgo.com/"}
        timeout = aiohttp.ClientTimeout(total=12)
        async with http.post("https://html.duckduckgo.com/html/", data=data,
                             headers=headers, timeout=timeout) as r:
            body = await r.text()
            if r.status >= 400 or "anomaly" in body.lower()[:400]:
                return []
    except Exception as e:
        log.info("DuckDuckGo HTML search failed: %s", e)
        return []
    rows = []
    for m in re.finditer(
            r'uddg=([^&"]+).*?class="result__a"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
            body, re.S | re.I):
        url = htmlmod.unescape(m.group(1))
        try:
            from urllib.parse import unquote
            url = unquote(url)
        except Exception:
            pass
        url = _safe_url(url) or url
        rows.append({
            "title": _plain(m.group(2), 200),
            "url": url,
            "snippet": _plain(m.group(3), 320),
        })
        if len(rows) >= count:
            break
    if rows:
        return rows
    for m in re.finditer(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                         body, re.S | re.I):
        url = _safe_url(htmlmod.unescape(m.group(1)))
        if not url:
            continue
        rows.append({"title": _plain(m.group(2), 200), "url": url, "snippet": ""})
        if len(rows) >= count:
            break
    return rows


async def _ddg_api(http, query, count):
    url = "https://api.duckduckgo.com/?q=%s&format=json&no_html=1&skip_disambig=1" % quote(query)
    try:
        status, text = await _get(http, url, timeout=10, headers={"Accept": "application/json"})
        if status != 200:
            return []
        data = json.loads(text)
    except Exception:
        return []
    rows = []
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        rows.append({
            "title": data.get("Heading") or query,
            "url": data.get("AbstractURL") or "",
            "snippet": abstract[:400],
        })
    for t in data.get("RelatedTopics") or []:
        if not isinstance(t, dict) or not t.get("Text"):
            continue
        rows.append({
            "title": (t.get("Text") or "")[:80],
            "url": t.get("FirstURL") or "",
            "snippet": t.get("Text") or "",
        })
        if len(rows) >= count:
            break
    return rows


async def tool_fetch_page(http, url, max_chars=4000):
    url = _safe_url(url)
    if not url:
        return "That URL is not allowed (only public http and https pages)."
    try:
        status, text = await _get(http, url, timeout=12)
    except Exception as e:
        return "Could not fetch the page: %s" % e
    if status >= 400:
        return "The page returned HTTP %s." % status
    body = _plain(text, max(500, min(int(max_chars or 4000), MAX_RESULT)))
    if not body:
        return "The page had no readable text."
    return "Text from %s:\n\n%s" % (url, body)


_WMO = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 95: "thunderstorm",
}


async def tool_weather(http, place, days=2):
    place = (place or "").strip()
    if not place:
        return "Name a city or place for the weather."
    days = max(1, min(int(days or 2), 5))
    try:
        geo_url = ("https://geocoding-api.open-meteo.com/v1/search?name=%s&count=1&language=en"
                   % quote(place))
        status, text = await _get(http, geo_url, timeout=10, headers={"Accept": "application/json"})
        geo = json.loads(text) if status == 200 else {}
        hit = (geo.get("results") or [None])[0]
        if not hit:
            return "Could not find a location named %r." % place
        lat, lon = hit["latitude"], hit["longitude"]
        label = ", ".join(x for x in (hit.get("name"), hit.get("admin1"), hit.get("country")) if x)
        wx_url = (
            "https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s"
            "&current=temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min"
            "&forecast_days=%s&timezone=auto" % (lat, lon, days)
        )
        status, text = await _get(http, wx_url, timeout=10, headers={"Accept": "application/json"})
        wx = json.loads(text) if status == 200 else {}
    except Exception as e:
        return "Weather lookup failed: %s" % e
    cur = wx.get("current") or {}
    code = _WMO.get(cur.get("weather_code"), "unknown conditions")
    now = ("Now in %s: %s, %s °C, wind %s km/h, humidity %s%%."
           % (label, code, cur.get("temperature_2m"), cur.get("wind_speed_10m"),
              cur.get("relative_humidity_2m")))
    daily = wx.get("daily") or {}
    extras = []
    for i, day in enumerate(daily.get("time") or []):
        extras.append("%s: %s, high %s °C, low %s °C" % (
            day, _WMO.get((daily.get("weather_code") or [None])[i], "unknown"),
            (daily.get("temperature_2m_max") or [None])[i],
            (daily.get("temperature_2m_min") or [None])[i],
        ))
    return now + ((" Forecast: " + "; ".join(extras) + ".") if extras else "")


async def tool_stock_quote(http, symbol):
    symbol = re.sub(r"[^A-Za-z0-9.=^-]", "", (symbol or "").strip()).upper()
    if not symbol:
        return "Give a ticker symbol, for example AAPL."
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d&range=5d"
           % quote(symbol))
    try:
        status, text = await _get(http, url, timeout=10, headers={"Accept": "application/json"})
        data = json.loads(text) if status == 200 else {}
    except Exception as e:
        return "Stock lookup failed: %s" % e
    result = ((data.get("chart") or {}).get("result") or [None])[0]
    if not result:
        err = ((data.get("chart") or {}).get("error") or {}).get("description")
        return err or "No quote for %s." % symbol
    meta = result.get("meta") or {}
    price = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    currency = meta.get("currency") or ""
    name = meta.get("shortName") or meta.get("symbol") or symbol
    if price is None:
        return "No current price for %s." % symbol
    change = ""
    if prev:
        delta = price - prev
        change = " (%s%.2f, %s%.1f%% vs previous close)" % (
            "+" if delta >= 0 else "", delta, "+" if delta >= 0 else "", 100 * delta / prev)
    return "%s (%s) is %.2f %s%s as of the latest market data." % (
        name, symbol, price, currency, change)


LOCAL_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the public web for anything you do not know or were not trained on: facts, people, how-to, documents, news, and current events. Use this instead of saying you do not know or inventing an answer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "How many results (1-8)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": "Download the readable text of a public web page. Use after web_search when a result needs more detail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http or https URL"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "weather",
        "description": "Current weather and a short forecast for a named place. Do not guess the weather.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "place": {"type": "string", "description": "City or place name"},
                "days": {"type": "integer", "description": "Forecast days (1-5)", "default": 2},
            },
            "required": ["place"],
        },
    },
    {
        "name": "stock_quote",
        "description": "Latest market price for a stock or ETF ticker (for example AAPL, MSFT, BTC-USD).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
]

_HANDLERS = {
    "web_search": lambda http, a: tool_web_search(http, a.get("query"), a.get("count")),
    "fetch_page": lambda http, a: tool_fetch_page(http, a.get("url")),
    "weather": lambda http, a: tool_weather(http, a.get("place"), a.get("days")),
    "stock_quote": lambda http, a: tool_stock_quote(http, a.get("symbol")),
}


# ------------------------------------------------------------------ hub

class McpHub:
    """Local stateless MCP server plus optional remote HTTP MCP servers."""

    def __init__(self, cfg, http: aiohttp.ClientSession):
        self.cfg = cfg.get("mcp") or {}
        self.http = http
        self.remote = []   # (prefix, url)
        self._catalog = list(LOCAL_TOOLS)
        self.enabled = self.cfg.get("enabled", True)

    async def start(self):
        if not self.enabled:
            log.info("MCP tools off")
            return
        for spec in self.cfg.get("servers") or []:
            try:
                reject_legacy(spec)
            except ValueError as e:
                log.warning("Skipping MCP server %s: %s", spec.get("name") or spec, e)
                continue
            url = (spec.get("url") or "").strip()
            if spec.get("name") == "local" or url in ("", "local"):
                continue
            name = re.sub(r"[^a-z0-9]+", "_", (spec.get("name") or "remote").lower()).strip("_") or "remote"
            try:
                tools = await self._remote_list(url)
            except Exception as e:
                log.warning("MCP %s at %s is not usable: %s", name, url, e)
                continue
            self.remote.append((name, url))
            for t in tools:
                row = dict(t)
                row["name"] = "%s__%s" % (name, t["name"])
                row["_remote"] = (url, t["name"])
                self._catalog.append(row)
            log.info("MCP %s: %d tool(s) from %s", name, len(tools), url)
        log.info("MCP ready: %s", ", ".join(t["name"] for t in self._catalog) or "(none)")

    def catalog(self):
        return list(self._catalog) if self.enabled else []

    def openai(self):
        return openai_tools(self.catalog())

    async def call(self, name, arguments):
        t0 = time.perf_counter()
        args = arguments if isinstance(arguments, dict) else {}
        try:
            text = await self._dispatch(name, args)
        except Exception as e:
            log.warning("Tool %s failed: %s", name, e)
            text = "Tool %s failed: %s" % (name, e)
        text = (text or "").strip() or "(empty)"
        if len(text) > MAX_RESULT:
            text = text[:MAX_RESULT] + "…"
        log.info("Tool %s in %.0f ms", name, (time.perf_counter() - t0) * 1000)
        return text

    async def _dispatch(self, name, args):
        if name in _HANDLERS:
            return await _HANDLERS[name](self.http, args)
        for t in self._catalog:
            if t["name"] == name and t.get("_remote"):
                url, raw = t["_remote"]
                return await self._remote_call(url, raw, args)
        return "Unknown tool %s." % name

    def _meta(self, version=PROTOCOL):
        return {
            "io.modelcontextprotocol/protocolVersion": version,
            "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            "io.modelcontextprotocol/clientCapabilities": {},
        }

    async def _rpc(self, url, method, params, version=PROTOCOL):
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": method,
            "params": dict(params or {}, _meta=self._meta(version)),
        }
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": version,
        }
        timeout = aiohttp.ClientTimeout(total=20)
        async with self.http.post(url, json=payload, headers=headers, timeout=timeout) as r:
            if r.headers.get("Mcp-Session-Id") or r.headers.get("MCP-Session-Id"):
                raise ValueError("server minted a session id (stateful MCP)")
            ctype = (r.content_type or "").lower()
            if "text/event-stream" in ctype:
                body = await self._read_sse_result(r)
            else:
                body = await r.json(content_type=None)
        if not isinstance(body, dict):
            raise ValueError("MCP response is not a JSON object")
        if body.get("error"):
            err = body["error"]
            raise ValueError(err.get("message") or str(err))
        return body.get("result") or {}

    async def _read_sse_result(self, response):
        """One-shot SSE result (allowed). A long-lived session GET is not."""
        result = None
        async for raw in response.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data in ("", "[DONE]"):
                continue
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if isinstance(msg, dict) and (msg.get("result") is not None or msg.get("error")):
                result = msg
        return result or {}

    async def _remote_list(self, url):
        try:
            result = await self._rpc(url, "tools/list", {})
        except ValueError as e:
            if "initialize" in str(e).lower() or "session" in str(e).lower():
                raise
            result = await self._rpc(url, "server/discover", {})
            if result.get("supportedVersions") and PROTOCOL not in result["supportedVersions"]:
                log.info("Remote MCP versions: %s", result.get("supportedVersions"))
            result = await self._rpc(url, "tools/list", {})
        tools = result.get("tools") or []
        if not tools:
            raise ValueError("tools/list returned no tools")
        return tools

    async def _remote_call(self, url, name, arguments):
        result = await self._rpc(url, "tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            parts = []
            for c in result.get("content") or []:
                if c.get("text"):
                    parts.append(c["text"])
            return "Tool error: " + (" ".join(parts) or "unknown")
        texts = [c.get("text") for c in (result.get("content") or []) if c.get("text")]
        return "\n".join(texts) if texts else json.dumps(result)[:MAX_RESULT]


def handle_rpc(body):
    """Stateless local MCP endpoint: one JSON-RPC object in, one out."""
    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
        return _err(None, -32600, "Invalid Request"), 400
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}
    meta = params.get("_meta") if isinstance(params, dict) else {}
    version = ""
    if isinstance(meta, dict):
        version = meta.get("io.modelcontextprotocol/protocolVersion") or ""
    if version and version not in SUPPORTED:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32022, "message": "Unsupported protocol version",
                      "data": {"supported": list(SUPPORTED), "requested": version}},
        }, 400
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": CLIENT_INFO,
            "instructions": TOOLS_HINT,
        }}, 200
    if method == "notifications/initialized":
        return None, 202
    if method == "server/discover":
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "supportedVersions": list(SUPPORTED),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": CLIENT_INFO,
            "instructions": TOOLS_HINT,
        }}, 200
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": LOCAL_TOOLS}}, 200
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}, 200
    if method == "tools/call":
        return None, 200  # async path in the HTTP handler
    return _err(req_id, -32601, "Method not found"), 404


def _err(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


async def call_local(http, params):
    name = (params or {}).get("name")
    args = (params or {}).get("arguments") or {}
    if name not in _HANDLERS:
        return {"content": [{"type": "text", "text": "Unknown tool %s." % name}], "isError": True}
    text = await _HANDLERS[name](http, args)
    return {"content": [{"type": "text", "text": text or ""}]}
