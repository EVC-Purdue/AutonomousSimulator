"""Episode cost before and after the Modal optimisations -> figures/episode_cost.png.

.venv/bin/python tools/plot_episode_cost.py <before_run> <after_run> <before_smoke_app> <after_smoke_app> <smoke_sim_s>
Runs are tools/opt_modal.py output dirs (lap first); smoke apps are the Modal app ids of two smokes of the same 15
episodes, billed per `modal billing report`. A dev job is projected from each smoke's billed dollars per sim second.
"""
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import clips  # noqa: E402

N_LAP, N_CLIP = 171, 387
WARMUP_S = 3.1  # sim seconds parked before the context phase (IMU calibration gate)
PROCS = ("sim", "pathfinder_node", "imu_node", "gps_node", "e_comms_node", "localization_node", "master_api", "ros2 bag")
LABELS = ("sim (Chrono + devices)", "pathfinder", "imu", "gps", "e_comms", "localization", "master_api", "bag")
COLORS = ("#000000", "#CFB991", "#8E6F3E", "#B42318", "#6F727B", "#1E7B4F", "#4B6A9B", "#E6E5E1")


def dev_sim_s(lap_s):
    """Upper bound: every lap lap_s long, every accepted clip running its full window."""
    clip = []
    for cid in clips.accepted_ids():
        c = clips.load(cid)
        clip.append(WARMUP_S + c["handoff_t"] + max(c["end_t"] - c["handoff_t"], 15.0))
    return N_LAP * lap_s + N_CLIP * sum(clip) / len(clip)


def billed(app_id):
    rows = json.loads(subprocess.run([os.path.join(ROOT, ".venv", "bin", "modal"), "billing", "report", "--for", "today",
                                      "--json"], capture_output=True, text=True, check=True).stdout)
    return sum(float(r["cost"]) for r in rows if r["object_id"] == app_id)


def main():
    before, after, sb, sa, smoke_sim = *sys.argv[1:5], float(sys.argv[5])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    for x, d in enumerate((before, after)):
        lap = json.load(open(os.path.join(d, "stats.json")))["episodes"][0]
        bottom = 0.0
        for p, lab, c in zip(PROCS, LABELS, COLORS):
            v = lap["by_proc"].get(p, 0.0) / lap["sim_s"]  # zombie re-counts land in "other" and are left out
            ax1.bar(x, v, bottom=bottom, color=c, width=0.6, label=lab if x == 0 else None)
            bottom += v
        ax1.text(x, bottom, f"{bottom:.1f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks([0, 1], ["before", "after"])
    ax1.set_ylabel("vCPU s per sim s, kart 1 lap")
    ax1.legend(fontsize=7, frameon=False, loc="upper right")
    dev, sim = [], dev_sim_s(lap["sim_s"])
    for app in (sb, sa):
        usd = billed(app)
        dev.append(usd / smoke_sim * sim)
        print(app, f"smoke ${usd:.4f} for {smoke_sim:.0f} sim s, dev ({sim:.0f} sim s) ${dev[-1]:.2f}")
    ax2.bar([0, 1], dev, color=["#6F727B", "#CFB991"], width=0.6)
    for i, v in enumerate(dev):
        ax2.text(i, v, f"${v:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks([0, 1], ["before", "after"])
    ax2.set_ylabel("dev job, $ (from billed smoke)")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "episode_cost.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
