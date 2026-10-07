"""Fits TMeasy pure-slip curves to TTC Hoosier runs at each load bin (TTC raw units: lbf, deg, psi, mph); writes ttc_fits.json."""
import json, os, numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares

LBF = 4.44822
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'data', 'ttc') + '/'      # TTC raw runs from github.com/ccleb152/PacejkaModelv2 RawDataFiles
LAT = {'front': {'R25B': 'Cornering/A1965run12.mat', 'LCO': 'Cornering/A1965run24.mat'},
       'rear':  {'R25B': 'Cornering/A1965run2.mat',  'LCO': 'Cornering/A1965run18.mat'}}
LON = {'R25B': 'BrakeDrive/A1654run43.mat', 'LCO': 'BrakeDrive/A1654run47.mat'}
BINS = np.array([50, 100, 150, 200, 250])


def tm(s, dF0, FM, sM, FS, sS):
    """TMeasy pure-slip characteristic (Rill), odd in s."""
    a = np.abs(s); f = np.empty_like(a)
    i = a <= sM; x = a[i] / sM
    f[i] = dF0 * a[i] / (1 + x * (x + dF0 * sM / FM - 2))
    j = (a > sM) & (a < sS); sg = (a[j] - sM) / (sS - sM)
    f[j] = FM - (FM - FS) * sg ** 2 * (3 - 2 * sg)
    f[a >= sS] = FS
    return np.sign(s) * f


def load(f):
    r = loadmat(D + f); g = lambda k: r[k].ravel().astype(float)
    return {k: g(k) for k in ('FZ', 'FY', 'FX', 'SA', 'SR', 'IA', 'P', 'V')}


def fit_bins(s, F, fz, mask, fs_ratio, sS):
    out = []
    for b in BINS:
        m = mask & (np.abs(fz - b) < 12)
        if m.sum() < 300: continue
        Fz = fz[m].mean() * LBF; x, y = s[m], F[m] * LBF
        # p[3] force offset (conicity, ply steer), p[4] slip offset (TTC zero-Fx slip is not at SR = 0)
        res = least_squares(lambda p: tm(x - p[4], p[0], p[1], p[2], fs_ratio * p[1], sS) + p[3] - y,
                            [20 * Fz, 2.5 * Fz, 0.1, 0.0, 0.0], bounds=([1, 1, 0.01, -1e4, -0.08], [1e6, 1e5, 0.5, 1e4, 0.08]))
        dF0, FM, sM, off, s0 = res.x
        out.append(dict(Fz=Fz, dF0=dF0, FM=FM, sM=sM, off=off, s0=s0, rms=float(np.sqrt(np.mean(res.fun ** 2))), n=int(m.sum()), smax=float(np.abs(x).max())))
    return out


FS_RATIO, SS_Y, SS_X = 0.85, 0.6, 0.5   # not observable inside the tested slip range; kept from make_kart.py
if __name__ == '__main__':
    res = {}
    for pos, files in LAT.items():
        for comp, f in files.items():
            d = load(f); m = (np.abs(d['IA']) < 0.3) & (np.abs(d['P'] - 12) < 0.6) & (d['V'] > 20)
            # SAE sign: positive SA gives negative FY, so fit -FY against tan(SA)
            res[f'lat/{pos}/{comp}'] = fit_bins(np.tan(np.radians(d['SA'])), -d['FY'], -d['FZ'], m, FS_RATIO, SS_Y)
    for comp, f in LON.items():
        d = load(f); m = (np.abs(d['IA']) < 0.3) & (np.abs(d['P'] - 12) < 0.6) & (np.abs(d['SA']) < 0.3)
        res[f'lon/{comp}'] = fit_bins(d['SR'], d['FX'], -d['FZ'], m, FS_RATIO, SS_X)
    json.dump(res, open(os.path.join(HERE, 'ttc_fits.json'), 'w'), indent=1)
    for k, v in res.items():
        for e in v: print(f"{k:16s} s0 {e['s0']:+.3f} Fz {e['Fz']:6.0f} N  dF0/Fz {e['dF0']/e['Fz']:6.1f}  mu {e['FM']/e['Fz']:5.2f}  sM {e['sM']:.3f}  rms {e['rms']:5.1f} N  n {e['n']}  |s|max {e['smax']:.3f}")
