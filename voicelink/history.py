"""Stored conversations and project folders under data/."""
import json
import os
import re
import secrets
import threading
import time

from .config import ROOT

DATA_DIR = os.path.join(ROOT, "data", "conversations")
PROJECTS_PATH = os.path.join(ROOT, "data", "projects.json")
_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
_UNSET = object()


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _clean_labels(labels):
    out, seen = [], set()
    for x in labels or []:
        name = " ".join(str(x).split())[:40]
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= 12:
            break
    return out


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
            out.append({
                "id": c["id"], "title": c.get("title") or "Untitled",
                "created": c.get("created"), "updated": c.get("updated"),
                "count": len(msgs), "preview": first[:120],
                "labels": _clean_labels(c.get("labels")),
                "project_id": c.get("project_id") or None,
            })
        out.sort(key=lambda c: c["updated"] or "", reverse=True)
        return out

    def get(self, cid):
        with open(self._path(cid), encoding="utf-8") as f:
            return json.load(f)

    def save(self, cid, messages, title=None, model=None, labels=_UNSET, project_id=_UNSET):
        """Creates or updates a conversation; returns its stored record."""
        with self.lock:
            path = self._path(cid)
            conv = {"id": cid, "title": None, "created": _now(), "messages": [],
                    "labels": [], "project_id": None}
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
            if labels is not _UNSET:
                conv["labels"] = _clean_labels(labels)
            else:
                conv["labels"] = _clean_labels(conv.get("labels"))
            if project_id is not _UNSET:
                conv["project_id"] = project_id or None
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(conv, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
            return conv

    def update(self, cid, title=_UNSET, labels=_UNSET, project_id=_UNSET):
        conv = self.get(cid)
        if title is not _UNSET:
            title = (title or "").strip()[:120] or conv.get("title")
        else:
            title = None
        return self.save(cid, conv.get("messages") or [], title=title,
                         labels=labels, project_id=project_id)

    def rename(self, cid, title):
        return self.update(cid, title=title)

    def delete(self, cid):
        with self.lock:
            os.remove(self._path(cid))

    def clear_project(self, pid):
        for row in self.list():
            if row.get("project_id") == pid:
                try:
                    self.update(row["id"], project_id=None)
                except (KeyError, OSError, ValueError):
                    pass


class ProjectStore:
    def __init__(self, path=PROJECTS_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.Lock()

    def _read(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _write(self, items):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def list(self):
        with self.lock:
            items = self._read()
        items.sort(key=lambda p: (p.get("name") or "").lower())
        return items

    def create(self, name):
        name = " ".join((name or "").split())[:60]
        if not name:
            raise ValueError("name is required")
        with self.lock:
            items = self._read()
            if any((p.get("name") or "").lower() == name.lower() for p in items):
                raise ValueError("a project with that name already exists")
            pid = "p-" + secrets.token_hex(4)
            items.append({"id": pid, "name": name, "created": _now()})
            self._write(items)
            return items[-1]

    def rename(self, pid, name):
        name = " ".join((name or "").split())[:60]
        if not name:
            raise ValueError("name is required")
        with self.lock:
            items = self._read()
            row = next((p for p in items if p.get("id") == pid), None)
            if not row:
                raise KeyError(pid)
            if any(p is not row and (p.get("name") or "").lower() == name.lower() for p in items):
                raise ValueError("a project with that name already exists")
            row["name"] = name
            self._write(items)
            return row

    def delete(self, pid):
        with self.lock:
            items = self._read()
            nxt = [p for p in items if p.get("id") != pid]
            if len(nxt) == len(items):
                raise KeyError(pid)
            self._write(nxt)


def _short_title(text, n=48):
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text or "New conversation"
    cut = text[:n].rsplit(" ", 1)[0]
    return (cut or text[:n]) + "…"
