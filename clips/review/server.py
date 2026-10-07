"""Local review UI for candidate clips: accept, reject, or accept under another class.

    .venv/bin/python -m clips.review.server   # http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import threading
import time

from flask import Flask, abort, jsonify, request, send_from_directory

from clips.mine import CLASSES, OUT

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(OUT, "labels.json")
_lock = threading.Lock()
app = Flask(__name__)


def load_labels() -> dict:
    if not os.path.exists(LABELS):
        return {}
    with open(LABELS) as f:
        return json.load(f)


@app.get("/")
def page():
    return send_from_directory(HERE, "index.html")


@app.get("/api/index")
def index():
    with open(os.path.join(OUT, "index.json")) as f:
        idx = json.load(f)
    idx["labels"] = load_labels()
    return jsonify(idx)


@app.get("/api/geometry")
def geometry():
    return send_from_directory(OUT, "geometry.json")


@app.get("/api/clip/<cid>")
def clip(cid: str):
    if "/" in cid or ".." in cid:
        abort(400)
    return send_from_directory(os.path.join(OUT, "clips"), cid + ".json")


@app.post("/api/label")
def label():
    body = request.get_json(force=True)
    cid, decision, cls = body.get("id"), body.get("decision"), body.get("cls")
    if decision not in ("accept", "reject", "clear") or (decision == "accept" and cls not in CLASSES):
        abort(400)
    with _lock:
        labels = load_labels()
        if decision == "clear":
            labels.pop(cid, None)
        else:
            labels[cid] = {"decision": decision, "cls": cls, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = LABELS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(labels, f, indent=1, sort_keys=True)
        os.replace(tmp, LABELS)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8765)), debug=False)
