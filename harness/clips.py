"""Accepted clips: ids from the local index, loading and time interpolation of the series."""
import bisect
import json
import math
import os

import numpy as np

from harness import failures
from harness.config import ROOT, load as load_config

CLIPS = os.path.join(ROOT, "data", "clips")
HF_REPO = "ShayManor/Sim_data"
CTX_SMOOTH_S = 0.5  # clip path averaged over this window before its curvature is taken (GPS jitter)
CTX_RATE_MARGIN = 0.5  # share of the kart's wheel rate the clip context may ask for (offline sweep)
CTX_RAMP_MPS2 = 1.5  # path acceleration of the ramp from the slowed context up to the recorded speed (offline sweep)
CTX_SETTLE_S = 1.0  # recorded seconds at the recorded speed before the handoff, about 2.5 drive time constants
CLIP_END_EXTRA_S = 0.2  # seconds a clip episode runs past its evaluation window, which ended a little early


def _fetch(revision):
    from huggingface_hub import snapshot_download   # host only; the image bakes data/clips
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", revision=revision, allow_patterns=["clips/*"],
                      local_dir=os.path.dirname(CLIPS))


def accepted_ids(revision=None):
    if not os.path.exists(os.path.join(CLIPS, "index.json")):
        _fetch(revision)
    with open(os.path.join(CLIPS, "index.json")) as f:
        return sorted(r["id"] for r in json.load(f))


def load(clip_id):
    if not os.path.exists(os.path.join(CLIPS, clip_id + ".json")):
        _fetch(load_config("tiers", "dev").get("clip_revision"))
    with open(os.path.join(CLIPS, clip_id + ".json")) as f:
        return json.load(f)


def series_at(clip, t):
    s = clip["series"]
    ts = s["t"]
    i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(ts) - 2)
    w = (t - ts[i]) / (ts[i + 1] - ts[i])
    out = {}
    for k, v in s.items():
        if k == "t":
            continue
        a, b = v[i], v[i + 1]
        if k == "yaw":
            b = a + math.remainder(b - a, 2 * math.pi)
        out[k] = a + w * (b - a)
    return out


def clip_speed_scale(clip, wheelbase, wheel_rate_dps):
    """Context speed factor, at most 1: driven at it, the recorded path before the handoff asks at most CTX_RATE_MARGIN of the kart's wheel rate."""
    s = clip["series"]
    t = np.asarray(s["t"], float)
    n = max(int(round(CTX_SMOOTH_S / float(np.median(np.diff(t))))), 1)
    x, y = (np.convolve(np.pad(np.asarray(s[k], float), n, mode="edge"), np.ones(n) / n, "same")[n:-n] for k in ("x", "y"))
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    kap = (dx * np.gradient(dy, t) - dy * np.gradient(dx, t)) / np.maximum(np.hypot(dx, dy), 1.0) ** 3  # GPS jitter below 1 m/s
    rate = np.abs(np.gradient(np.degrees(np.arctan(wheelbase * kap)), t))
    still = np.convolve(np.asarray(s["v"], float) < failures.CTX_SIG["moving_mps"], np.ones(2 * n + 1), "same") > 0
    m = (t <= 0.0) & (np.arange(len(t)) >= n) & ~still  # context, past the smoothing start, a smoothing window clear of standstill
    peak = float(np.max(rate[m])) if m.any() else 0.0
    return min(1.0, CTX_RATE_MARGIN * wheel_rate_dps / max(peak, 1e-9))


def clip_clock(clip, scale, accel=None):
    """Context clock on the series grid: (recorded t, sim s from the context start, speed factor); scale, ramped at accel to 1 CTX_SETTLE_S before the handoff."""
    tau = np.asarray(clip["series"]["t"], float)
    if accel is None:
        return tau, (tau - tau[0]) / scale, np.full(len(tau), float(scale))
    k = accel / max(float(np.interp(0.0, tau, clip["series"]["v"])), 0.1)  # factor per sim second
    te = -CTX_SETTLE_S
    g0 = max(scale, math.sqrt(max(0.0, 1.0 - 2.0 * k * (te - tau[0])))) if te > tau[0] else 1.0  # a short context starts faster
    tr = te - (1.0 - g0 * g0) / (2.0 * k)
    g = np.where(tau < tr, g0, np.where(tau > te, 1.0, np.sqrt(g0 * g0 + 2.0 * k * np.clip(tau - tr, 0.0, None))))
    t = np.where(tau < tr, (tau - tau[0]) / g0, (tr - tau[0]) / g0 + (np.minimum(g, 1.0) - g0) / k + np.maximum(tau - te, 0.0))
    return tau, t, g
