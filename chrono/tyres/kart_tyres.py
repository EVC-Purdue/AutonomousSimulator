"""Kart TMeasy parameters from the TTC Hoosier R25B fits in ttc_fits.json; run directly to plot them against the data."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = json.load(open(os.path.join(HERE, "ttc_fits.json")))
FS_RATIO, SS = 0.85, {"lat": 0.6, "lon": 0.5}   # post-peak: outside the tested slip range, not fitted
LON = "lon/R25B"                                  # longitudinal data exists only for the 18x6.0-10


def _at(fits, fz, key):
    z = [e["Fz"] for e in fits]; return float(np.interp(fz, z, [e[key] for e in fits]))


def curve(pos, ax, fz_nom, mu, stiffness=1.0):
    """Two-load TMeasy set (Fz_nom, 2 Fz_nom), whole curve scaled so the peak is mu at Fz_nom, small-slip slope times stiffness."""
    lat = FITS[f"lat/{pos}/R25B"]; fits = lat if ax == "lat" else FITS[LON]
    loads = (fz_nom, 2 * fz_nom)
    lam = mu * fz_nom / _at(fits, fz_nom, "FM")
    fm = [lam * _at(fits, z, "FM") for z in loads]
    if ax == "lon":
        # Low-load longitudinal sweeps are noisy and give mu rising with load; take the load sensitivity from lateral
        fm[1] = fm[0] * _at(lat, loads[1], "FM") / _at(lat, loads[0], "FM")
    return {"dF0": [stiffness * lam * _at(fits, z, "dF0") for z in loads], "FM": fm,
            "sM": [_at(fits, z, "sM") for z in loads], "FS": [FS_RATIO * f for f in fm], "sS": [SS[ax]] * 2}


def tmeasy_blocks(pos, fz_nom, mu, stiffness=1.0):
    """The Longitudinal and Lateral blocks of a Chrono TMeasy tyre JSON."""
    out = {}
    for ax, name, c in (("lon", "Longitudinal", "x"), ("lat", "Lateral", "y")):
        p = curve(pos, ax, fz_nom, mu, stiffness)
        out[name] = {f"Initial Slopes dF{c}/ds{c} [N]": p["dF0"], f"Maximum F{c} Load [N]": p["FM"],
                     f"Sliding F{c} Load [N]": p["FS"], f"Slip s{c} at Maximum F{c}": p["sM"],
                     f"Slip s{c} where sliding begins": p["sS"]}
    return out


if __name__ == "__main__":
    import sys
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, HERE); import fit_ttc as F

    def at_load(p, fzn, fz):
        """Chrono/Rill interpolation of a two-load set to load fz."""
        r = fz / fzn; q = lambda a: r * (2 * a[0] - 0.5 * a[1] - (a[0] - 0.5 * a[1]) * r)
        return q(p["dF0"]), q(p["FM"]), p["sM"][0] + (p["sM"][1] - p["sM"][0]) * (r - 1), q(p["FS"]), p["sS"][0]

    kart = {p: json.load(open(os.path.join(HERE, "..", "kart", f"Kart_Tire{p.title()}.json"))) for p in ("front", "rear")}
    for sp in ("top", "right"): plt.rcParams[f"axes.spines.{sp}"] = False
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, (pos, ax, title) in zip(axs, (("front", "lat", "front, lateral"), ("rear", "lat", "rear, lateral"), ("rear", "lon", "rear, longitudinal"))):
        fzn = kart[pos]["Parameters"]["Vertical"]["Nominal Vertical Force [N]"]; mu = kart[pos]["Coefficient of Friction"]
        p = curve(pos, ax, fzn, mu); fits = FITS[f"lat/{pos}/R25B"] if ax == "lat" else FITS[LON]
        d = F.load(F.LAT[pos]["R25B"] if ax == "lat" else F.LON["R25B"]); fz = -d["FZ"]
        m = (np.abs(d["IA"]) < 0.3) & (np.abs(d["P"] - 12) < 0.6)
        if ax == "lat": m &= d["V"] > 20; s = np.tan(np.radians(d["SA"])); Fd = -d["FY"]
        else: m &= np.abs(d["SA"]) < 0.3; s = d["SR"]; Fd = d["FX"]
        lam = mu * fzn / _at(fits, fzn, "FM")
        for b, col in ((50, "#4C72B0"), (100, "#DD8452")):
            mb = m & (np.abs(fz - b) < 12)
            if mb.sum() < 300: continue
            e = min(fits, key=lambda e: abs(e["Fz"] - b * F.LBF))
            a.scatter(s[mb] - e["s0"], lam * (Fd[mb] * F.LBF - e["off"]) / e["Fz"], s=1, c=col, alpha=0.25, lw=0)
            xs = np.linspace(s[mb].min(), s[mb].max(), 400)
            for k, ls, lab in ((1.0, "-", "kart 1"), (1.3, ":", "kart 2 (x1.3)"), (0.7, "--", "kart 3 (x0.7)")):
                pk = curve(pos, ax, fzn, mu, k)
                a.plot(xs, F.tm(xs, *at_load(pk, fzn, e["Fz"])) / e["Fz"], c=col, ls=ls, lw=2 if k == 1.0 else 1.2,
                       label=f"{lab}, {e['Fz']:.0f} N")
        a.set_title(title, fontsize=10); a.set_xlabel("slip ratio" if ax == "lon" else "lateral slip tan(α)"); a.set_ylim(-1.2, 1.2)
        a.axhline(0, c="0.85", lw=0.5); a.axvline(0, c="0.85", lw=0.5)
    axs[0].set_ylabel("F / Fz"); axs[0].legend(frameon=False, fontsize=7, loc="lower right")
    out = os.path.join(HERE, "..", "..", "figures", "kart_tyres.png")
    fig.tight_layout(); fig.savefig(out, dpi=120); print("figure", os.path.abspath(out))
