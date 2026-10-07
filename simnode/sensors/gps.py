"""UM982 GPS model: fix-state chain, position error and the NMEA lines the receiver emits."""
import math

import numpy as np

from harness.geo import to_latlon

QUALITY = {"fixed": 4, "float": 5, "dgps": 2, "sps": 1, "none": 0}
GST_SIGMA_M = {"fixed": 0.01, "float": 0.15, "dgps": 0.5, "sps": 1.5}
GM_SIGMA_M = {"float": 0.10, "dgps": 0.4, "sps": 1.2}
GM_TAU_S = 20.0
SATS = {"fixed": 12, "float": 12, "dgps": 10, "sps": 8}
MODE = {"fixed": "D", "float": "D", "dgps": "D", "sps": "A"}
HDOP, ALT_M, GEOID_M, DATE = 0.7, 187.5, -33.8, "130926"
VEL_SIGMA_MPS = 0.03  # UM982 datasheet velocity accuracy
HDT_SIGMA_DEG = 0.1
MIN_COURSE_MPS = 0.1
OUTAGES_S = (1.0,) * 5 + (2.0,) * 2 + (3.0,) * 2 + (4.0,)
OUTAGE_WINDOW_S = 180.0
WRONG_FIX_M, WRONG_FIX_S, WRONG_FIX_EVERY_S = 0.15, 20.0, 600.0


def nmea_checksum(body):
    c = 0
    for b in body.encode("ascii"):
        c ^= b
    return f"{c:02X}"


def _crc32(body):
    crc = 0  # Unicore/NovAtel CRC32: reflected 0xEDB88320, zero init, no final xor
    for b in body.encode("ascii"):
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return f"{crc:08x}"


def _dm(v, width):
    n = int(round(abs(v) * 60e9))  # nano-minutes, so rounding never yields 60 minutes
    d, rem = divmod(n, 60 * 10**9)
    return f"{d:0{width}d}{rem // 10**9:02d}.{rem % 10**9:09d}"


def _utc(te):
    cs = int(round(te * 100)) % 8640000
    return f"{cs // 360000:02d}{cs // 6000 % 60:02d}{cs // 100 % 60:02d}.{cs % 100:02d}"


def _line(body):
    return f"${body}*{nmea_checksum(body)}\r\n"


class Gps:
    def __init__(self, cfg, rng, chain, mount):
        self.cfg, self.chain, self.mount = cfg, chain, mount
        self._chain_rng, self._noise_rng, self._outage_rng, self._wrong_rng, self._vel_rng, self._hdg_rng = rng.spawn(6)
        self._dt = 1.0 / cfg["rate_hz"]
        self._k, self._queue, self._prev = 0, [], None
        self._state, self._err, self._err_state, self._err_t = "fixed", None, None, 0.0
        self._switch_t = math.inf if cfg["fix_chain"] == "always_fixed" else self._dwell("fixed")
        self._window, self._outages = -1, []
        self._wrong_end, self._wrong_dir = -math.inf, 0.0
        self._wrong_next = self._wrong_rng.exponential(WRONG_FIX_EVERY_S) if cfg["wrong_fix"] else math.inf

    def _dwell(self, s):
        return self._chain_rng.exponential(self.chain["mean_dwell_s"][s] * (self.cfg["float_dwell_scale"] if s == "float" else 1.0))

    def _advance_state(self, te):
        while te >= self._switch_t:
            row = self.chain["transitions"][self._state]
            if row:
                keys = sorted(row)
                self._state = keys[self._chain_rng.choice(len(keys), p=[row[k] for k in keys])]
            self._switch_t += self._dwell(self._state)

    def _in_outage(self, te):
        if not self.cfg["outages"]:
            return False
        w = int(te // OUTAGE_WINDOW_S)
        if w != self._window:
            self._window, self._outages = w, []
            for d in OUTAGES_S[::-1]:
                while True:
                    s = w * OUTAGE_WINDOW_S + self._outage_rng.uniform(0.0, OUTAGE_WINDOW_S - d)
                    if all(s + d <= a or s >= a + b for a, b in self._outages):
                        break
                self._outages.append((s, d))
        return any(a <= te < a + b for a, b in self._outages)

    def _error(self, te):
        s, cfg = self._state, self.cfg
        if s == "fixed":
            sigma, tau, white = cfg["fixed_gm_sigma_m"] * cfg["error_scale"], cfg["fixed_gm_tau_s"], cfg["fixed_white_m"] * cfg["error_scale"]
        else:
            sigma, tau, white = GM_SIGMA_M[s], GM_TAU_S, 0.0
        if self._err_state != s:
            self._err, self._err_state = self._noise_rng.normal(0.0, sigma, 2), s
        else:
            phi = math.exp(-(te - self._err_t) / tau)
            self._err = phi * self._err + math.sqrt(1.0 - phi * phi) * sigma * self._noise_rng.normal(0.0, 1.0, 2)
        self._err_t = te
        return self._err + self._noise_rng.normal(0.0, white, 2)

    def _epoch(self, te, x, y, yaw, vx, vy, r):
        self._advance_state(te)
        while te >= self._wrong_next:
            self._wrong_end = self._wrong_next + WRONG_FIX_S
            self._wrong_dir = self._wrong_rng.uniform(0.0, 2 * math.pi)
            self._wrong_next = self._wrong_end + self._wrong_rng.exponential(WRONG_FIX_EVERY_S)
        if self._in_outage(te) or self._state == "none":
            return []
        s, c, sn = self._state, math.cos(yaw), math.sin(yaw)
        ax, ay = self.mount["antenna"][0], self.mount["antenna"][1]
        e = self._error(te)
        if s == "fixed" and te < self._wrong_end:
            e = e + WRONG_FIX_M * np.array([math.cos(self._wrong_dir), math.sin(self._wrong_dir)])
        lat, lon = to_latlon(x + c * ax - sn * ay + e[0], y + sn * ax + c * ay + e[1])
        dv = self._vel_rng.normal(0.0, VEL_SIGMA_MPS, 2)
        ve, vn = vx - r * (sn * ax + c * ay) + dv[0], vy + r * (c * ax - sn * ay) + dv[1]
        sog = math.hypot(ve, vn)
        cog = f"{math.degrees(math.atan2(ve, vn)) % 360.0:.3f}" if sog >= MIN_COURSE_MPS else ""
        utc, kn, mode, sig = _utc(te), sog * 3600.0 / 1852.0, MODE[s], GST_SIGMA_M[s]
        pos = f"{_dm(lat, 2)},{'N' if lat >= 0 else 'S'},{_dm(lon, 3)},{'E' if lon >= 0 else 'W'}"
        age, station = ("1.0", "0000") if s != "sps" else ("", "")
        out = [_line(f"GNGGA,{utc},{pos},{QUALITY[s]},{SATS[s]:02d},{HDOP:.2f},{ALT_M:.3f},M,{GEOID_M:.3f},M,{age},{station}"),
               _line(f"GNGST,{utc},{sig:.3f},{sig:.3f},{sig:.3f},0.0,{sig:.3f},{sig:.3f},{sig:.3f}"),
               _line(f"GNRMC,{utc},A,{pos},{kn:.3f},{cog},{DATE},,,{mode}"),
               _line(f"GNVTG,{cog},T,,M,{kn:.3f},N,{sog * 3.6:.3f},K,{mode}")]
        if self.cfg["heading_source"] == "dual":
            hdg = (90.0 + self.mount["baseline_az_deg"] - math.degrees(yaw)  # baseline azimuth, clockwise of the nose
                   + self._hdg_rng.normal(0.0, HDT_SIGMA_DEG)) % 360.0
            if s == "fixed":
                out.append(_line(f"GNHDT,{hdg:.4f},T"))
            status = "SOL_COMPUTED,NARROW_INT" if s == "fixed" else "INSUFFICIENT_OBS,NONE"
            body = (f"HEADINGA,COM1,0,0.0,FINESTEERING,0,{te % 604800.0:.3f},0,0,0;{status},"
                    f"{self.mount['baseline_m']:.4f},{hdg:.4f},0.0000,0.0000,{HDT_SIGMA_DEG:.4f},0.0000,\"0\",12,12,12,0,0,00,0,00")
            out.append(f"#{body}*{_crc32(body)}\r\n")
        return out

    def due(self, t):
        """True if sample(t) would release a line: a queued line, or an epoch whose time plus latency has passed."""
        if not self.cfg["enabled"]:
            return False
        te = self._k * self._dt
        return bool(self._queue) and self._queue[0][0] <= t + 1e-9 or te + self.cfg["latency_ms"] / 1000.0 <= t + 1e-9

    def sample(self, t, x, y, yaw, vx_world, vy_world):
        """Lines whose epoch plus latency has passed at t; position is the reference point, velocity world frame."""
        if not self.cfg["enabled"]:
            return []
        r = 0.0
        if self._prev is not None and t > self._prev[0]:
            r = math.remainder(yaw - self._prev[1], 2 * math.pi) / (t - self._prev[0])
        self._prev = (t, yaw)
        while self._k * self._dt <= t + 1e-9:
            te = self._k * self._dt
            self._queue += [(te + self.cfg["latency_ms"] / 1000.0, l) for l in self._epoch(te, x, y, yaw, vx_world, vy_world, r)]
            self._k += 1
        n = 0
        while n < len(self._queue) and self._queue[n][0] <= t + 1e-9:
            n += 1
        out, self._queue = [l for _, l in self._queue[:n]], self._queue[n:]
        return out
