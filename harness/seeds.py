"""Episode seeds: a hash of the run seed and the episode identity, never the recipe."""
import hashlib
import json


def episode_seed(run_seed, kart, condition, item):
    h = hashlib.sha256(json.dumps([run_seed, kart, condition, item]).encode()).hexdigest()
    return int(h[:8], 16)
