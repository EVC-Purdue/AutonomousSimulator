"""Scores a perception run: matches the stack's /opponents to the true karts per frame, then summary and figures."""
import json
import math
import os

import numpy as np

from perception.run import RENDERS

MIN_VISIBLE_PX = 20    # a kart with fewer visible pixels (render resolution) is not expected to be detected
MAX_RANGE_M = 40.0     # kart_detector_node max_range_m
SETTLE_K = 3           # tracker confirms after 3 hits: frames before this are excluded from accuracy and detection rate
BINS = [0, 5, 10, 15, 20, 30, 40, 1e9]


def _gate_rad(r):
    """Association is by bearing (range error is what is measured): within 1 m sideways at the true range, at least 2 deg."""
    return max(math.radians(2.0), math.atan2(1.0, r))


RANGE_RATIO = (0.5, 2.0)   # an estimate this far off in range is a different object, not a ranging error


def _load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_frame(fr, rec, truth_px):
    """Per-truth rows and the false-positive count for one frame."""
    ex, ey, eyaw = fr["ego"][:3]
    est = [] if rec is None or "error" in rec else rec["opponents"]
    used, rows = set(), []
    order = sorted(range(len(fr["truth"])), key=lambda j: fr["truth"][j]["role"] != "target")
    for j in order:
        t = fr["truth"][j]
        r = math.hypot(t["x"] - ex, t["y"] - ey)
        brg = math.atan2(t["y"] - ey, t["x"] - ex) - eyaw
        vis = truth_px[j] if j < len(truth_px) else 0
        best, bd = None, _gate_rad(r)
        for i, e in enumerate(est):
            q = math.hypot(e["x"] - ex, e["y"] - ey) / max(r, 1e-6)
            d = abs((math.atan2(e["y"] - ey, e["x"] - ex) - eyaw - brg + math.pi) % (2 * math.pi) - math.pi)
            if i not in used and RANGE_RATIO[0] <= q <= RANGE_RATIO[1] and d < bd:
                best, bd = i, d
        row = {"role": t["role"], "variant": t["variant"], "range": r, "bearing_deg": math.degrees((brg + math.pi) % (2 * math.pi) - math.pi),
               "visible_px": vis, "detectable": vis >= MIN_VISIBLE_PX and r <= MAX_RANGE_M, "matched": best is not None}
        if best is not None:
            used.add(best)
            e = est[best]
            re = math.hypot(e["x"] - ex, e["y"] - ey)
            be = math.atan2(e["y"] - ey, e["x"] - ex) - eyaw
            row.update(range_est=re, range_err=re - r, pos_err=math.hypot(e["x"] - t["x"], e["y"] - t["y"]),
                       bearing_err_deg=math.degrees((be - brg + math.pi) % (2 * math.pi) - math.pi), sigma_s=e["sigma_s"], sigma_d=e["sigma_d"])
            if "vx" in t:
                row["speed_err"] = e["s_dot"] - math.hypot(t["vx"], t["vy"])
        orc = (rec or {}).get("oracle") or []
        if j < len(orc) and orc[j]:
            row["oracle_range_err"] = math.hypot(orc[j]["x"] - ex, orc[j]["y"] - ey) - r
            row["oracle_ground"] = orc[j]["ground"]
        rows.append(row)
    return rows, len(est) - len(used)


def _stats(rows, fps, n_frames):
    det = [r for r in rows if r["role"] == "target" and r["detectable"]]
    hit = [r for r in det if r["matched"]]
    err = np.array([r["range_err"] for r in hit])
    rel = np.array([r["range_err"] / r["range"] for r in hit])
    out = {"n": len(det), "detection_rate": len(hit) / len(det) if det else None, "fp_per_frame": fps / n_frames if n_frames else None}
    orc = np.array([r["oracle_range_err"] for r in det if "oracle_range_err" in r])
    if len(orc):   # the stack's ranging on true boxes: the error left with a perfect detector
        out.update(oracle_range_err_median=float(np.median(orc)), oracle_range_abs_p50=float(np.percentile(np.abs(orc), 50)),
                   oracle_range_abs_p90=float(np.percentile(np.abs(orc), 90)))
    if len(err):
        out.update(range_err_median=float(np.median(err)), range_abs_p50=float(np.percentile(np.abs(err), 50)),
                   range_abs_p90=float(np.percentile(np.abs(err), 90)), range_rel_abs_p50=float(np.percentile(np.abs(rel), 50)),
                   bearing_abs_p50_deg=float(np.percentile(np.abs([r["bearing_err_deg"] for r in hit]), 50)))
    return out


def evaluate(run_dir):
    frames = _load(os.path.join(run_dir, "frames.jsonl"))
    outs = {(r["case"], r["seq"], r["k"]): r for r in _load(os.path.join(run_dir, "out.jsonl"))}
    truth_px = {}
    rows, per_frame = [], []
    for fr in frames:
        if fr["scene"] not in truth_px:
            with open(os.path.join(RENDERS, fr["scene"] + ".json")) as f:
                tk = json.load(f)["karts"]
            truth_px[fr["scene"]] = [next((k["visible_px"] for k in tk if k["idx"] == j + 1), 0) for j in range(len(fr["truth"]))]
        rec = outs.get((fr["case"], fr["seq"], fr["k"]))
        rs, fp = score_frame(fr, rec, truth_px[fr["scene"]])
        settled = fr["k"] >= SETTLE_K
        err = None if rec is None else rec.get("error")
        for r in rs:
            r.update(case=fr["case"], seq=fr["seq"], k=fr["k"], t=fr["t"], env=fr["env"], settled=settled, **{"p_" + k: v for k, v in fr["params"].items()})
        rows += rs
        per_frame.append({"case": fr["case"], "seq": fr["seq"], "k": fr["k"], "fp": fp, "settled": settled, "error": err})
    with open(os.path.join(run_dir, "eval.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"overall": None, "cases": {}, "errors": sum(p["error"] is not None for p in per_frame)}
    cases = sorted({r["case"] for r in per_frame})
    for c in cases:
        cr = [r for r in rows if r["case"] == c and r["settled"]]
        cf = [p for p in per_frame if p["case"] == c and p["settled"]]
        s = _stats(cr, sum(p["fp"] for p in cf), len(cf))
        s["by_range"] = {f"{lo}-{hi if hi < 1e9 else 'inf'}": _stats([r for r in cr if lo <= r["range"] < hi], 0, 0)
                         for lo, hi in zip(BINS[:-1], BINS[1:]) if any(lo <= r["range"] < hi for r in cr)}
        keys = sorted({k for r in cr for k in r if k.startswith("p_") and k not in ("p_range_m", "p_gap_m")}) + ["env"]
        s["by_param"] = {k: {str(v): _stats([r for r in cr if r.get(k) == v], 0, 0) for v in sorted({r.get(k) for r in cr}, key=str)} for k in keys}
        first = {}
        for r in rows:
            if r["case"] == c and r["role"] == "target" and r["matched"] and r["seq"] not in first:
                first[r["seq"]] = r["k"]
        s["frames_to_first_track"] = {str(q): first.get(q) for q in sorted({r["seq"] for r in rows if r["case"] == c})}
        summary["cases"][c] = s
    allr = [r for r in rows if r["settled"] and not r["case"].startswith(("static_calibration", "static_empty"))]
    allf = [p for p in per_frame if p["settled"]]
    summary["overall"] = _stats(allr, sum(p["fp"] for p in allf), len(allf))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    figures(run_dir, rows, summary)
    return summary


def figures(run_dir, rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    acc, gold, grey = "#2f6db5", "#cfb991", "#9a9a9a"

    st = [r for r in rows if r["settled"] and r["role"] == "target" and r["case"].startswith("static") and r["case"] != "static_calibration"]
    if st:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        hit = [r for r in st if r["matched"]]
        a1.scatter([r["range"] for r in hit], [r["range_est"] for r in hit], s=6, alpha=0.25, color=acc, lw=0)
        mx = max(r["range"] for r in st) * 1.05
        a1.plot([0, mx], [0, mx], color=grey, ls="--", lw=1)
        centres, med, rate = [], [], []
        for lo, hi in zip(range(0, 45, 3), range(3, 48, 3)):
            b = [r for r in st if lo <= r["range"] < hi and r["detectable"]]
            if b:
                h = [r["range_est"] for r in b if r["matched"]]
                centres.append((lo + hi) / 2)
                med.append(np.median(h) if h else np.nan)
                rate.append(len(h) / len(b))
        a1.plot(centres, med, color=gold, lw=2.5)
        a1.set(xlabel="true range (m)", ylabel="stack range (m)", xlim=(0, mx), ylim=(0, mx))
        a2.plot(centres, rate, color=acc, lw=2.5, marker="o", ms=4)
        a2.set(xlabel="true range (m)", ylabel="detection rate", ylim=(0, 1.02), xlim=(0, mx))
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_range.png"), dpi=150)
        plt.close(fig)

    cs = [c for c, s in summary["cases"].items() if s.get("n")]
    if cs:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.45 * len(cs) + 1.2), sharey=True)
        y = np.arange(len(cs))
        a1.barh(y, [summary["cases"][c]["detection_rate"] or 0 for c in cs], color=acc)
        a1.set(yticks=y, yticklabels=cs, xlabel="detection rate", xlim=(0, 1))
        a2.barh(y, [summary["cases"][c].get("range_abs_p50", np.nan) for c in cs], color=gold)
        a2.set(xlabel="median |range error| (m)")
        a1.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_cases.png"), dpi=150)
        plt.close(fig)

    tr = [r for r in rows if r["role"] == "target" and r["case"].startswith("track")]
    seqs = sorted({(r["case"], r["seq"]) for r in tr})
    if seqs:
        fig, axes = plt.subplots(len(seqs), 1, figsize=(10, 2.3 * len(seqs)), squeeze=False)
        for ax, (c, q) in zip(axes[:, 0], seqs):
            rr = sorted((r for r in tr if r["case"] == c and r["seq"] == q), key=lambda r: r["k"])
            t0 = rr[0]["t"]
            ax.plot([r["t"] - t0 for r in rr], [r["range"] for r in rr], color=grey, lw=2)
            ax.scatter([r["t"] - t0 for r in rr if r["matched"]], [r["range_est"] for r in rr if r["matched"]], s=8, color=acc, zorder=3)
            ax.set_ylabel("range (m)")
            ax.set_title(f"{c} {rr[0].get('p_seq', q)}", loc="left", fontsize=10)
        axes[-1, 0].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_track.png"), dpi=150)
        plt.close(fig)

    cal = [r for r in rows if r["case"] == "static_calibration" and r["settled"] and r["matched"]]
    if cal:
        fig, ax = plt.subplots(figsize=(6, 4))
        for h, col in zip(sorted({r["p_cam_height_err_m"] for r in cal}), (acc, grey, gold)):
            xs = sorted({r["p_cam_pitch_err_deg"] for r in cal})
            ys = [np.median([r["range_err"] / r["range"] for r in cal if r["p_cam_height_err_m"] == h and r["p_cam_pitch_err_deg"] == x]) * 100 for x in xs]
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4, label=f"height {h:+.1f} m")
        ax.axhline(0, color=grey, lw=0.8)
        ax.set(xlabel="camera pitch error (deg, + = more down than configured)", ylabel="median range error (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_calibration.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    import sys
    print(json.dumps(evaluate(sys.argv[1])["overall"], indent=1))
