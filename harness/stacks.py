"""Stack profiles: how the sim builds, launches and talks to each stack repo, from configs/stacks/<repo>.json."""
import json
import os
import re

from harness import config

DEFAULT_REPO = "AutonomousKart"


def load(repo=DEFAULT_REPO):
    p = os.path.join(config.CONFIGS, "stacks", repo + ".json")
    if not os.path.exists(p):
        raise ValueError(f"Unknown stack repo {repo}")
    with open(p) as f:
        return json.load(f)


def repo_of(recipe):
    return ((recipe or {}).get("source") or {}).get("repo", DEFAULT_REPO)


def of(recipe):
    return load(repo_of(recipe))


def repos():
    return sorted(f[:-5] for f in os.listdir(os.path.join(config.CONFIGS, "stacks")) if f.endswith(".json"))


def label(recipe):
    """Stack tag of a job: the label its recipe name starts with, else its repo's label."""
    labels = [load(r)["label"] for r in repos()]
    first = re.split(r"[^A-Za-z0-9]", recipe["name"].strip())[0].upper()
    return first if first in labels else of(recipe)["label"]
