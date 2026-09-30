"""Stored conversations: one JSON file per conversation in data/conversations/."""
import json
import os
import re
import secrets
import threading
import time

from .config import ROOT

DATA_DIR = os.path.join(ROOT, "data", "conversations")
_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class ConversationStore:
    def __init__(self, folder=DATA_DIR):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.lock = threading.Lock()

    def _path(self, cid):
        if not _ID.match(cid or ""):
            raise KeyError(cid)
        return os.path.join(self.folder, cid + ".json")

    def new_id(self):
        return time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)

    def list(self):
        out = []
        for name in os.listdir(self.folder):
            if not name.endswith(".json"):
                continue
            try:
                c = self.get(name[:-5])
            except (KeyError, OSError, ValueError):
                continue
            msgs = c.get("messages", [])
            first = next((m["content"] for m in msgs if m["role"] == "user"), "")
            out.append({"id": c["id"], "title": c.get("title") or "Untitled", "created": c.get("created"),
                        "updated": c.get("updated"), "count": len(msgs), "preview": first[:120]})
        out.sort(key=lambda c: c["updated"] or "", reverse=True)
        return out

    def get(self, cid):
        with open(self._path(cid), encoding="utf-8") as f:
            return json.load(f)

    def save(self, cid, messages, title=None, model=None):
        """Creates or updates a conversation; returns its stored record."""
        with self.lock:
            path = self._path(cid)
            conv = {"id": cid, "title": None, "created": _now(), "messages": []}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    conv = json.load(f)
            conv["messages"] = messages
            conv["updated"] = _now()
            if title:
                conv["title"] = title
            if not conv.get("title"):
                first = next((m["content"] for m in messages if m["role"] == "user"), "")
                conv["title"] = _short_title(first)
            if model:
                conv["model"] = model
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(conv, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
            return conv

    def rename(self, cid, title):
        conv = self.get(cid)
        return self.save(cid, conv["messages"], title=(title or "").strip()[:120] or conv.get("title"))

    def delete(self, cid):
        with self.lock:
            os.remove(self._path(cid))


def _short_title(text, n=48):
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text or "New conversation"
    cut = text[:n].rsplit(" ", 1)[0]
    return (cut or text[:n]) + "…"
