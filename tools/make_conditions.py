"""Writes the condition files from configs/levels: nominal, low and high per subsystem, all-low, all-high and the extras."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = os.path.join(ROOT, "configs", "levels")
OUT = os.path.join(ROOT, "configs", "conditions")
SUBSYSTEMS = ("steering", "gps", "imu", "wheel", "actuator", "grip", "frame")


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
    conds["gps_dead"] = pick({}) + [{"path": "sensors.gps.enabled", "set": False}]   # custom clip jobs only, in no tier
    for a in ("low", "medium", "high"):   # large tier: sensor noise (gps, imu, wheel) x steering noise
        for b in ("low", "medium", "high"):
            conds[f"sensors_{a}_steer_{b}"] = pick({"gps": a, "imu": a, "wheel": a, "steering": b})
    for a in ("low", "medium", "high"):   # large tier v3: steering+actuator x kart (frame, grip) x sensors (gps, imu, wheel)
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
