"""Writes the Top Kart Twister (heavy, 275 lb) as Chrono::Vehicle JSON into chrono/kart/, and karts 2 and 3 into chrono/kart2/ and chrono/kart3/."""
import json, math, os
import numpy as np
from tyres.kart_tyres import tmeasy_blocks

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kart")

# Geometry: Twister 103/CH/17 form and sim/mj/twister.py; vehicle frame x fwd, y left, z up, origin on the ground under the rear axle
WHEELBASE = 1.050
FRONT_TRACK, REAR_TRACK = 1.200 - 0.130, 1.400 - 0.210
R_FRONT = 0.127                       # 10x4.50-5
R_REAR = 0.1397                       # 11x7.10-5 nominal
KINGPIN_X, KINGPIN_Y, KINGPIN_Z = 1.0795, 0.3398, 0.127
KPI_DEG, CASTER_DEG = 10.5, 12.0
ARM_LEN, ARM_DEG = 0.1200, 87.7       # "standard" spindle arm hole
COLUMN_LOW = np.array([1.021, 0.0, 0.130])
COLUMN_DEG = 38.0
SPADE_R, SPADE_SPAN = 0.050, 0.014    # bottom spade holes, 52 mm off the axis, measured from the Top Kart shaft in UPenn's LSBW CAD
COLUMN_MAX_DEG = 70.0
TOE_OUT_TOTAL_M = 0.0025

# Mass: 124.74 kg (275 lb) total; CG x and y from the measured corner weights, 0.147 m up (sim/mj/kart.py computed)
M_TOTAL = 124.74
CORNER_LB = dict(FL=56.18, FR=64.56, RL=63.62, RR=89.78)   # scale, 274.14 lb read, deliberately unbalanced (battery, motor)
_W = sum(CORNER_LB.values())
CG = np.array([(CORNER_LB["FL"] + CORNER_LB["FR"]) / _W * WHEELBASE,
               ((CORNER_LB["FL"] - CORNER_LB["FR"]) * FRONT_TRACK + (CORNER_LB["RL"] - CORNER_LB["RR"]) * REAR_TRACK) / 2 / _W, 0.147])
TORSION_K = 1.0 / (1.0 / 17000.0 + 1.0 / 26000.0)   # sim/mj/kart.py front and rear torsion joints in series, N m/rad
CONNECTOR = np.array([0.55, 0.0, 0.055])

# Powertrain: mj_kart_september_spec.json values (94.8 N m, 2632 W, 39.3 N m) rescaled so this model's full-throttle and braking curves hit the September fit, a = min(5.62, 21.1/v) and 2.33 m/s^2
WHEEL_TORQUE, WHEEL_POWER, BRAKE_TORQUE = 94.8 * 5.62 / 5.18, 2632.0 * 1.135, 39.3 * 2.33 / 2.20
GEAR = 65.0 / 15.0
V_MAX = 12.0
R_ROLL = 10.411 * 0.0254 / 2.0        # measured loaded rolling radius, powertrain.py
MU = 0.91                             # 2026-09-19 spinouts; TTC slip at peak agrees with the measured 11.5% knee

FRONT_PARTS = dict(upright=0.72, spindle=0.70, wheel=1.0, tire=2.1)
REAR_PARTS = dict(spindle=1.6, wheel=1.4, tire=3.0)
STEER_MASS = 1.5
FRONT_CHASSIS = dict(mass=12.0, com=np.array([0.85, 0.0, 0.10]), I=[0.50, 0.17, 0.65])
REAR_I = [5.0, 4.6, 8.4]
TYRE_STIFFNESS = 1.0
NAME = "Top Kart Twister 103/CH/17, Purdue EVC heavy"

# Karts 2 and 3 override the kart 1 values above; every kart is over 250 lb and brakes by VESC regen only
KARTS = {
    "kart": {},
    "kart2": dict(  # heavy, long, slow-steering, understeering EV, chosen to differ clearly from kart 1; every value an estimate
        NAME="Heavy long-wheelbase EV (estimates)", WHEELBASE=1.150, KINGPIN_X=1.1795, COLUMN_LOW=np.array([1.121, 0.0, 0.130]),
        CASTER_DEG=8.0, ARM_LEN=0.1200, ARM_DEG=89.5, SPADE_R=0.038,  # outer spindle-arm hole (almost parallel), 38 mm spade
        M_TOTAL=150.0, CORNER_LB=None, CG=np.array([0.38 * 1.150, 0.0, 0.200]), REAR_I=[6.5, 6.5, 12.0], TYRE_STIFFNESS=1.3,
        TORSION_K=1.5 / (1.0 / 17000.0 + 1.0 / 26000.0)),
    "kart3": dict(  # UPenn AV4EV (github.com/mlab-upenn/gokart-mechatronics, MIT): motor turns a horizontal cut-down Top Kart shaft
        NAME="UPenn AV4EV Top Kart, steer-by-wire on the spade", COLUMN_LOW=np.array([0.985, 0.0, 0.170]), COLUMN_DEG=0.0,
        M_TOTAL=115.0, CORNER_LB=None, CG=np.array([0.400, 0.0, 0.170]), TYRE_STIFFNESS=0.7),
}


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def quat_z_to(u):
    """(w, x, y, z) rotating +z onto u."""
    z = np.array([0.0, 0.0, 1.0]); u = unit(u); c = float(z @ u); ax = np.cross(z, u); s = np.linalg.norm(ax)
    if s < 1e-12: return [1.0, 0.0, 0.0, 0.0]
    ax /= s; h = math.atan2(s, c) / 2.0
    return [math.cos(h), *(ax * math.sin(h))]


def kingpin_axis(side):
    """Unit vector up the kingpin: top leans back by caster and inboard by KPI."""
    return unit([-math.tan(math.radians(CASTER_DEG)), -side * math.tan(math.radians(KPI_DEG)), 1.0])


def column_axis():
    a = math.radians(COLUMN_DEG); return np.array([-math.cos(a), 0.0, math.sin(a)])


def spade_holes():
    ax = column_axis(); u = unit(np.cross(ax, [0, 0, 1])) * SPADE_SPAN
    back = unit(np.cross(u, ax)); back = -back if back[2] > 0 else back
    return {s: COLUMN_LOW + back * SPADE_R + s * u for s in (1, -1)}


def arm_hole(side):
    th = math.radians(ARM_DEG)
    kp = np.array([KINGPIN_X, side * KINGPIN_Y, KINGPIN_Z])
    return kp + np.array([-math.sin(th), side * math.cos(th), 0.0]) * ARM_LEN + np.array([0, 0, -0.010])


def rear_radius():
    """Unloaded rear radius that rolls at the measured loaded radius; TMeasy rolls at about r0 minus a third of the static deflection."""
    fz = M_TOTAL * 9.81 * (1 - CG[0] / WHEELBASE) / 2.0
    return R_ROLL + fz / 100000.0 / 3.0


def tmeasy(name, pos, r0, width, mass, fz_nom):
    I_spin = 0.5 * mass * (r0 ** 2 + 0.0635 ** 2); I_tr = 0.25 * mass * (r0 ** 2 + 0.0635 ** 2) + mass * width ** 2 / 12.0
    return {
        "Name": name, "Type": "Tire", "Template": "TMeasyTire",
        "Design": {"Unloaded Radius [m]": r0, "Mass [kg]": mass, "Inertia [kg.m2]": [I_tr, I_spin, I_tr],
                   "Width [m]": width, "Rim Radius [m]": 0.0635},
        "Coefficient of Friction": MU,
        "Rolling Resistance Coefficient": 0.0,
        "Parameters": {
            "Vertical": {"Nominal Vertical Force [N]": fz_nom,
                         "Vertical Tire Stiffness [N/m]": 100000.0,
                         "Tire Vertical Damping [Ns/m]": 1400.0,
                         "Tire Bottoming Radius [m]": 0.0635 + 0.01, "Tire Bottoming Stiffness [N/m]": 1.0e6},
            **tmeasy_blocks(pos, fz_nom, MU, TYRE_STIFFNESS),
            "Aligning": {"Normalized Trail at Zero Slip sy": [0.17, 0.17], "Slip sy where Trail Changes Sign": [0.15, 0.15],
                         "Slip sy where Trail Tends to Zero": [0.6, 0.6]}}}


def main():
    base = {k: globals()[k] for k in set().union(*KARTS.values())}
    for name, over in KARTS.items():
        globals().update(base); globals().update(over); build(name)


def build(name):
    out = os.path.join(os.path.dirname(OUT), name)
    os.makedirs(out, exist_ok=True)
    w = lambda n, d: json.dump(d, open(os.path.join(out, n), "w"), indent=2)
    r_rear = rear_radius()
    front_loc = np.array([WHEELBASE, 0.0, R_FRONT]); rear_loc = np.array([0.0, 0.0, r_rear])

    # Rear chassis carries what the front chassis, wheels, corners and steering do not, placed so the whole kart hits CG
    m_front_corner = sum(FRONT_PARTS.values()); m_rear_corner = sum(REAR_PARTS.values())
    parts = [(2 * m_front_corner, front_loc), (2 * m_rear_corner, rear_loc), (STEER_MASS, COLUMN_LOW),
             (FRONT_CHASSIS["mass"], FRONT_CHASSIS["com"])]
    m_rear = M_TOTAL - sum(m for m, _ in parts)
    rear_com = (M_TOTAL * CG - sum(m * p for m, p in parts)) / m_rear

    w("Kart_FrontChassis.json", {
        "Name": "Twister front frame", "Type": "Chassis", "Template": "RigidChassis",
        "Components": [{"Centroidal Frame": {"Location": list(FRONT_CHASSIS["com"]), "Orientation": [1, 0, 0, 0]},
                        "Mass": FRONT_CHASSIS["mass"], "Moments of Inertia": FRONT_CHASSIS["I"],
                        "Products of Inertia": [0, 0, 0], "Void": False}],
        "Driver Position": {"Location": [0.40, 0.0, 0.40], "Orientation": [1, 0, 0, 0]},
        "Rear Connector Location": list(CONNECTOR)})
    w("Kart_RearChassis.json", {
        "Name": "Twister rear frame, motor, packs, seat", "Type": "ChassisRear", "Template": "RigidChassisRear",
        "Components": [{"Centroidal Frame": {"Location": list(rear_com), "Orientation": [1, 0, 0, 0]},
                        "Mass": m_rear, "Moments of Inertia": REAR_I,
                        "Products of Inertia": [0, 0, 0], "Void": False}],
        "Front Connector Location": list(CONNECTOR)})
    w("Kart_TorsionConnector.json", {"Name": "Twister frame twist", "Type": "ChassisConnector",
                                     "Template": "ChassisConnectorTorsion", "Torsional Stiffness": TORSION_K})

    # Front corner: spindle on a castered, inclined kingpin, steered by a tie rod from the spade
    kp = np.array([KINGPIN_X, KINGPIN_Y, KINGPIN_Z]) - front_loc
    hub = np.array([WHEELBASE, FRONT_TRACK / 2, R_FRONT]) - front_loc
    toe_deg = math.degrees(math.atan2(TOE_OUT_TOTAL_M / 2.0, 2 * R_FRONT))
    w("Kart_FrontCorner.json", {
        "Name": "Twister front corner", "Type": "Suspension", "Template": "GenericWheeledSuspension",
        "Vehicle-Frame Inertia": False, "Independent": True, "Steerable": True,
        "Camber Angle (deg)": 0, "Toe Angle (deg)": -toe_deg,
        "Spindle": {"Attachment Body": "Upright", "Mass": FRONT_PARTS["spindle"], "COM": list(hub),
                    "Inertia": [0.0006, 0.0010, 0.0006], "Radius": 0.0125, "Width": 0.10},
        "Bodies": [{"Name": "Upright", "Mirrored": True, "Position": list((kp + hub) / 2), "Rotation": [1, 0, 0, 0],
                    "Mass": FRONT_PARTS["upright"], "Moments of Inertia": [0.002, 0.0005, 0.002],
                    "Products of Inertia": [0, 0, 0]}],
        "Joints": [{"Name": "Kingpin", "Type": "Revolute", "Mirrored": True, "Body1": "Chassis", "Body2": "Upright",
                    "Position": list(kp), "Rotation": quat_z_to(kingpin_axis(+1))}],
        "Distance Constraints": [{"Name": "Tierod", "Mirrored": True, "Body1": "Steering", "Body2": "Upright",
                                  "Point1": list(spade_holes()[1] - front_loc), "Point2": list(arm_hole(+1) - front_loc)}],
        "Axle": {"Inertia": 0.001}})
    w("Kart_RearAxle.json", {
        "Name": "Twister solid rear axle", "Type": "Suspension", "Template": "RigidSuspension",
        "Spindle": {"Mass": REAR_PARTS["spindle"], "COM": [0.0, REAR_TRACK / 2, 0.0],
                    "Inertia": [0.003, 0.004, 0.003], "Radius": 0.025, "Width": 0.05},
        "Axle": {"Inertia": 0.01}})
    w("Kart_Column.json", {
        "Name": "Twister column and double spade, worm-drive stepper 1:1", "Type": "Steering", "Template": "RotaryArm",
        "Pitman Arm": {"Mass": STEER_MASS, "Inertia": [0.03, 0.03, 0.002], "Inertia Products": [0, 0, 0], "Radius": 0.009,
                       "Axis of Rotation": list(column_axis()), "Point of Rotation": list(COLUMN_LOW - front_loc),
                       "Point to Draglink": list(spade_holes()[1] - front_loc), "Maximum Angle (deg)": COLUMN_MAX_DEG}})
    for n, d in (("Kart_WheelFront.json", FRONT_PARTS["wheel"]), ("Kart_WheelRear.json", REAR_PARTS["wheel"])):
        w(n, {"Name": n[:-5], "Type": "Wheel", "Template": "Wheel", "Mass": d, "Inertia": [0.004, 0.006, 0.004]})
    w("Kart_RearBrake.json", {"Name": "VESC regen, no mechanical brake", "Type": "Brake", "Template": "BrakeSimple",
                              "Maximum Torque": BRAKE_TORQUE / 2.0})
    w("Kart_NoBrake.json", {"Name": "no front brakes", "Type": "Brake", "Template": "BrakeSimple", "Maximum Torque": 0.0})
    w("Kart_Driveline.json", {
        "Name": "Solid rear axle, chain", "Type": "Driveline", "Template": "ShaftsDriveline2WD",
        "Shaft Direction": {"Motor Block": [0, 1, 0], "Axle": [0, 1, 0]},
        "Shaft Inertia": {"Driveshaft": 0.056, "Differential Box": 0.01},
        "Gear Ratio": {"Conical Gear": 1.0}, "Axle Differential Locking Limit": 1.0e5})
    w("Kart_Motor.json", {"Name": "Motenergy ME0708 on VESC", "Type": "Engine", "Template": "EngineSimple",
                          "Maximum Engine Torque": WHEEL_TORQUE / GEAR, "Maximum Engine Power": WHEEL_POWER,
                          "Maximum Engine Speed": V_MAX / R_ROLL * GEAR})
    w("Kart_Chain.json", {"Name": "15T/65T chain", "Type": "Transmission", "Template": "AutomaticTransmissionSimpleMap",
                          "Gear Box": {"Reverse Gear Ratio": -1.0 / GEAR, "Forward Gear Ratios": [1.0 / GEAR],
                                       "Shift Points Map RPM": [[0, 1.0e5]]}})
    fz_f = M_TOTAL * 9.81 * CG[0] / WHEELBASE / 2; fz_r = M_TOTAL * 9.81 * (1 - CG[0] / WHEELBASE) / 2
    w("Kart_TireFront.json", tmeasy("10x4.50-5 slick", "front", R_FRONT, 0.114, FRONT_PARTS["tire"], fz_f))
    w("Kart_TireRear.json", tmeasy("11x7.10-5 slick", "rear", r_rear, 0.180, REAR_PARTS["tire"], fz_r))

    p = name + "/"
    w("Kart_Vehicle.json", {
        "Name": NAME, "Type": "Vehicle", "Template": "WheeledVehicle",
        "Chassis": {"Input File": p + "Kart_FrontChassis.json"},
        "Rear Chassis": [{"Input File": p + "Kart_RearChassis.json", "Connector Input File": p + "Kart_TorsionConnector.json",
                          "Chassis Index": -1}],
        "Axles": [
            {"Chassis Index": -1, "Suspension Input File": p + "Kart_FrontCorner.json", "Suspension Location": list(front_loc),
             "Steering Index": 0, "Left Wheel Input File": p + "Kart_WheelFront.json", "Right Wheel Input File": p + "Kart_WheelFront.json",
             "Left Brake Input File": p + "Kart_NoBrake.json", "Right Brake Input File": p + "Kart_NoBrake.json"},
            {"Chassis Index": 0, "Suspension Input File": p + "Kart_RearAxle.json", "Suspension Location": list(rear_loc),
             "Left Wheel Input File": p + "Kart_WheelRear.json", "Right Wheel Input File": p + "Kart_WheelRear.json",
             "Left Brake Input File": p + "Kart_RearBrake.json", "Right Brake Input File": p + "Kart_RearBrake.json"}],
        "Steering Subsystems": [{"Input File": p + "Kart_Column.json", "Location": list(front_loc), "Orientation": [1, 0, 0, 0]}],
        **({"Corner Weights [lb]": CORNER_LB} if CORNER_LB else {}),   # not read by Chrono; the CG above is derived from it
        "Wheelbase": WHEELBASE, "Minimum Turning Radius": 3.1, "Maximum Steering Angle (deg)": 30,
        "Driveline": {"Input File": p + "Kart_Driveline.json", "Suspension Indexes": [1]}})
    print(name, "rear chassis mass %.2f com %s, rear r0 %.4f, toe %.3f deg, torsion %.0f N m/rad" % (m_rear, rear_com.round(4), r_rear, toe_deg, TORSION_K))


if __name__ == "__main__":
    main()
