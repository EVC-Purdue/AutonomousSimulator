"""Copies the accepted, reviewed clips to data/clips with the reviewed class and writes the index."""
import json
import os
import subprocess

from clips.mine import AK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "clip_labels", "2026-10-04")
OUT = os.path.join(ROOT, "data", "clips")
LINES_OUT = os.path.join(ROOT, "tracks", "clip_lines")


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for row in json.load(open(os.path.join(SRC, "summary.json"))):
        if row["decision"] != "accept":
            continue
        clip = json.load(open(os.path.join(SRC, "clips", row["id"] + ".json")))
        clip["cls"] = row["cls"]
        with open(os.path.join(OUT, row["id"] + ".json"), "w") as f:
            json.dump(clip, f)
        index.append({"id": row["id"], "cls": row["cls"], "line": clip["line"],
                      "handoff_t": clip["handoff_t"], "end_t": clip["end_t"]})
    index.sort(key=lambda r: r["id"])
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(len(index), "clips")
    # Exact recorded line blobs, verbatim from AutonomousKart history.
    os.makedirs(LINES_OUT, exist_ok=True)
    for name in sorted({r["line"] for r in index}):
        blob = subprocess.run(["git", "-C", AK, "cat-file", "-p", name.split("@")[1]], capture_output=True, check=True).stdout
        with open(os.path.join(LINES_OUT, name + ".csv"), "wb") as f:
            f.write(blob)
    print(len(os.listdir(LINES_OUT)), "clip lines")


if __name__ == "__main__":
    main()
