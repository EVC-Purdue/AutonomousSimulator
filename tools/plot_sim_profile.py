"""figures/sim_profile.png from tools/profile_episode.py outputs: where one sim second of wall time and CPU goes.

.venv/bin/python tools/plot_sim_profile.py <modal 1 ms run> <modal deadline run> <local 1 ms py-spy run> <local deadline py-spy run>
Left: Modal wall per sim second on the sim's critical path. Right: Modal CPU per sim second by process, each stack
node split into its own callbacks and rclpy executor overhead with the shares of the local py-spy run (py-spy cannot
attach under Modal's gVisor).
"""
import collections
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDLE = ("rclpy/executors.py:682)", "concurrent/futures/thread.py:81)", "serial/serialposix.py:565)",
        "subprocess.py:1917)")  # leaf lines that block (rcl_wait, pool queue, serial select, child wait), not CPU
NODES = ("imu_node", "gps_node", "pathfinder_node", "e_comms_node", "master_api", "localization_node")
WALL = [("Chrono physics", "#1A1A1A"), ("/clock publish", "#B42318"), ("sensors", "#6F727B"),
        ("waiting for the stack", "#CFB991"), ("other sim work (by difference)", "#E6E5E1")]


def own_code(frame):
    return "autonomous_kart/" in frame and "main (" not in frame and "start (" not in frame


def executor_share(path):
    """Fraction of a node's on-CPU py-spy samples outside its own callbacks (rclpy executor and DDS)."""
    c = collections.Counter()
    for line in open(path):
        line = line.rstrip()
        if not line:
            continue
        st, n = line.rsplit(" ", 1)
        frames = st.split(";")
        if frames[-1].endswith(IDLE) or "threading.py:" in frames[-1]:
            continue
        c["own" if any(own_code(f) for f in frames) else "exec"] += int(n)
    return c["exec"] / max(sum(c.values()), 1)


def wall_parts(p):
    ph, o = p["phase_s"], p["outcome"]
    sim = o["sim_s"]
    parts = {"Chrono physics": ph["plant_step"] + ph["plant_truth"] + ph["motor_vesc"], "/clock publish": ph["clock_pub"],
             "sensors": ph["sense"],
             "waiting for the stack": sum(ph[k] for k in ("wait_spin", "wait_pty", "wait_nmea", "wait_sleep", "wait_other"))}
    loop = o["wall_s"] - p["startup_wall"]  # ticking wall, startup excluded
    parts["other sim work (by difference)"] = loop - sum(parts.values())
    return {k: v / sim for k, v in parts.items()}


def main():
    m_all, m_dl, l_all, l_dl = (json.load(open(os.path.join(d, "profile.json"))) for d in sys.argv[1:5])
    shares = {"1 ms": {n: executor_share(os.path.join(sys.argv[3], f"pyspy_{n}.txt")) for n in NODES},
              "deadline": {n: executor_share(os.path.join(sys.argv[4], f"pyspy_{n}.txt")) for n in NODES}}
    print(json.dumps({"wall": {"1 ms": wall_parts(m_all), "deadline": wall_parts(m_dl)}, "executor_share": shares,
                      "cpu": {m: {k: v / p["auto"]["sim"] for k, v in p["auto"]["cpu_s"].items()}
                              for m, p in (("1 ms", m_all), ("deadline", m_dl))}}, indent=1))
    for sp in ("top", "right"):
        plt.rcParams[f"axes.spines.{sp}"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [0.7, 1.5], "wspace": 0.9})
    ax = axes[0]
    for x, p in enumerate((m_all, m_dl)):
        parts, bottom = wall_parts(p), 0.0
        for name, color in WALL:
            ax.bar(x, parts[name], 0.55, bottom=bottom, color=color, label=name if x == 0 else None)
            bottom += parts[name]
        ax.text(x, bottom + 0.05, f"{bottom:.2f}x", ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["/clock 1 ms", "/clock 5 ms"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("wall s per sim s")
    ax.set_title("Modal, sim critical path", fontsize=10, loc="left")
    ax.tick_params(axis="x", length=0)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax = axes[1]
    rows = ["sim"] + list(NODES) + ["bag"]
    y = list(range(len(rows)))[::-1]
    for (key, p, off) in (("1 ms", m_all, 0.2), ("deadline", m_dl, -0.2)):
        cpu = {k: v / p["auto"]["sim"] for k, v in p["auto"]["cpu_s"].items()}
        for name, yy in zip(rows, y):
            v = cpu.get(name, 0.0)
            ex = v * shares[key].get(name, 0.0)
            ax.barh(yy + off, v - ex, 0.36, color="#1A1A1A" if name == "sim" else "#CFB991",
                    label="callbacks (sim: physics, devices, wait loop)" if (name == "pathfinder_node" and key == "1 ms") else None)
            ax.barh(yy + off, ex, 0.36, left=v - ex, color="#B42318" if key == "1 ms" else "#E8A39B",
                    label="rclpy executor overhead, /clock " + ("1 ms" if key == "1 ms" else "5 ms") if name == "imu_node" else None)
            ax.text(v + 0.04, yy + off, "1 ms" if key == "1 ms" else "5 ms", va="center", fontsize=6.5, color="#6F727B")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("CPU s per sim s (vCPU)")
    t = sum(m_all["auto"]["cpu_s"].values()) / m_all["auto"]["sim"], sum(m_dl["auto"]["cpu_s"].values()) / m_dl["auto"]["sim"]
    ax.set_title(f"Modal CPU by process: {t[0]:.1f} vs {t[1]:.1f} CPU s per sim s", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.92, wspace=1.0)
    fig.savefig(os.path.join(ROOT, "figures", "sim_profile.png"), dpi=140)


if __name__ == "__main__":
    main()
