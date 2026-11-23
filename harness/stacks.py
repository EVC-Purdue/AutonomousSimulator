"""Stack profiles: how the sim builds, launches and talks to each stack repo, from configs/stacks/<repo>.json."""
import json
import os

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
