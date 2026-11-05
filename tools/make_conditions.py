"""Writes the condition files from configs/levels: nominal, 3x3 steering/sensor noise x tire configs, localization configs, and legacy single-axis conditions."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = os.path.join(ROOT, "configs", "levels")
OUT = os.path.join(ROOT, "configs", "conditions")
SUBSYSTEMS = ("steering", "gps", "imu", "wheel", "actuator", "grip", "frame")

TIRE_OVERRIDES = {
    "low": [{"path": "tyres.grip", "scale": 0.8}, {"path": "tyres.stiffness", "scale": 0.85}],
    "medium": [{"path": "tyres.grip", "scale": 1.0}, {"path": "tyres.stiffness", "scale": 1.0}],
    "high": [{"path": "tyres.grip", "scale": 1.12}, {"path": "tyres.stiffness", "scale": 1.15}],
}

LOC_SCALE = {"low": 0.7, "medium": 1.0, "high": 1.6}


def main():
    lv = {s: json.load(open(os.path.join(LEVELS, s + ".json"))) for s in SUBSYSTEMS}
    pick = lambda chosen: [o for s in SUBSYSTEMS for o in lv[s][chosen.get(s, "medium")]]
    conds = {"nominal": pick({})}
    for s in SUBSYSTEMS:
        for level in ("low", "high"):
            conds[f"{s}_{level}"] = pick({s: level})
    conds["all_low"] = pick({s: "low" for s in SUBSYSTEMS})
    conds["all_high"] = pick({s: "high" for s in SUBSYSTEMS})
    conds["heading_dual"] = pick({}) + [{"path": "sensors.gps.heading_source", "set": "dual"}]
    conds["freq_45"] = pick({}) + [{"path": "system_frequency", "set": 45}]
    conds["gps_dead"] = pick({}) + [{"path": "sensors.gps.enabled", "set": False}]
    # 3 steering/sensor noise configs x 3 tire configs (core A27 3x3 grid)
    for noise in ("low", "medium", "high"):
        base_noise = [o for s in ("steering", "gps", "imu", "wheel", "actuator", "frame")
                      for o in lv[s][noise if s != "frame" else "medium"]]
        for tire in ("low", "medium", "high"):
            conds[f"noise_{noise}_tire_{tire}"] = base_noise + TIRE_OVERRIDES[tire]
        # Localization study: GPS+IMU vs SLAM vs 3DGS vs SLAM+3DGS across noise levels
        sc = LOC_SCALE[noise]
        for mode, slam_on, gs_on in (
            ("gps_imu", False, False),
            ("slam", True, False),
            ("3dgs", False, True),
            ("slam_3dgs", True, True),
        ):
            conds[f"loc_{mode}_{noise}"] = base_noise + TIRE_OVERRIDES["medium"] + [
                {"path": "sensors.slam.enabled", "set": slam_on},
                {"path": "sensors.slam.pos_sigma_m", "scale": sc},
                {"path": "sensors.gs3d.enabled", "set": gs_on},
                {"path": "sensors.gs3d.inlier_sigma_m", "scale": sc},
            ]
    for a in ("low", "medium", "high"):
        for b in ("low", "medium", "high"):
            conds[f"sensors_{a}_steer_{b}"] = pick({"gps": a, "imu": a, "wheel": a, "steering": b})
    for a in ("low", "medium", "high"):
        for b in ("low", "medium", "high"):
            for c in ("low", "medium", "high"):
                conds[f"steer_{a}_kart_{b}_sensors_{c}"] = pick({"steering": a, "actuator": a, "frame": b, "grip": b,
                                                                 "gps": c, "imu": c, "wheel": c})
    for p in glob.glob(os.path.join(OUT, "*.json")):
        os.remove(p)
    for name, ov in conds.items():
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump({"name": name, "overrides": ov}, f, indent=1)
    print(len(conds), "conditions")


if __name__ == "__main__":
    main()
