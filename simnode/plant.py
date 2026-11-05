"""Chrono kart plant: grip, stiffness, power and frame scaling of the kart JSONs, spawn, 0.5 ms steps and ground-truth state."""
import json
import math
import os
import shutil
import tempfile
from collections import deque

import numpy as np

STEP = 5e-4
CDA = 0.1185 / (0.5 * 1.225)
RHO = 1.225
G = 9.81
R_ROLL = 10.411 * 0.0254 / 2.0  # loaded rolling radius the stack's powertrain.py assumes
TYRES = ("Kart_TireFront.json", "Kart_TireRear.json")
SCALED = ("Initial Slopes", "Maximum F", "Sliding F")
SPAWN_Z = -0.003  # reference-frame height at which the tyres start just loaded, so the kart does not drop
ACCEL_WINDOW = 20  # steps; mean acceleration over 10 ms filters the solver's step-to-step chatter (std 4.7 m/s^2 at rest)
NOMINAL_POWER_KW = 14.0
IMU_FRONT_POS = (0.9476, 0.0033, 0.0782)  # imu_1 on chassis_front (waist/floorpan)
IMU_BACK_POS = (0.1852, -0.0117, 0.0720)   # imu_2 on chassis_rear (crossbar B3)
GPS_ANTENNA_POS = (0.6397, -0.1085, 0.6736)


def imu_accel_at(accel, gyro, pos_m, ref_pos_m=(0.85, 0.0, 0.10)):
    """Specific force at an IMU mount offset from the chassis COM reference (includes planar centripetal omega_z^2 * r)."""
    wz = float(gyro[2])
    dx = float(pos_m[0]) - float(ref_pos_m[0])
    dy = float(pos_m[1]) - float(ref_pos_m[1])
    return [float(accel[0]) - wz * wz * dx, float(accel[1]) - wz * wz * dy, float(accel[2])]


def aligning_torque(vx, vy, yaw_rate, wheel_l_deg, wheel_r_deg, wheelbase=1.05, mu=0.91, fz_f=539.0):
    """Front-axle self-aligning torque (N m at steering column) with TMeasy pneumatic trail collapse before peak Fy."""
    if abs(vx) < 0.4:
        return 0.0
    wheel_rad = math.radians((wheel_l_deg + wheel_r_deg) * 0.5)
    sy = math.tan(wheel_rad) - (vy + yaw_rate * wheelbase) / max(abs(vx), 0.4)
    asy = abs(sy)
    # TMeasy pneumatic trail parameters: t_n0=0.17, sign change at sy0=0.15, zero at sye=0.60
    if asy <= 0.15:
        tn = 0.17 * (1.0 - asy / 0.15)
    elif asy < 0.60:
        u = (asy - 0.15) / 0.45
        tn = -0.025 * 4.0 * u * (1.0 - u)
    else:
        tn = 0.0
    patch_len = 0.090
    caster_trail = 0.012
    sy_peak = 0.18
    fy_norm = math.tanh(asy / sy_peak * 1.4)
    fy = math.copysign(mu * fz_f * fy_norm, sy)
    return float(fy * (tn * patch_len + caster_trail) * 0.305)


def scaled_kart_dir(kart, root):
    """Temp copy of kart["chrono_dir"] with tyre grip/stiffness, power limit, brake and frame torsion scaled."""
    src = os.path.join(root, kart["chrono_dir"])
    dst = os.path.join(tempfile.mkdtemp(prefix="kart_"), os.path.basename(os.path.normpath(src)))
    shutil.copytree(src, dst)
    grip = kart["tyres"]["grip"]
    stiff = kart["tyres"].get("stiffness", 1.0)
    for name in TYRES:
        p = os.path.join(dst, name)
        with open(p) as f:
            d = json.load(f)
        d["Coefficient of Friction"] *= grip
        for block in ("Longitudinal", "Lateral"):
            for k, v in d["Parameters"][block].items():
                if k.startswith("Initial Slopes"):
                    d["Parameters"][block][k] = [x * grip * stiff for x in v]
                elif k.startswith(SCALED):
                    d["Parameters"][block][k] = [x * grip for x in v]
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
    p = os.path.join(dst, "Kart_TorsionConnector.json")
    with open(p) as f:
        d = json.load(f)
    d["Torsional Stiffness"] *= kart["frame"]["torsion_scale"]
    with open(p, "w") as f:
        json.dump(d, f, indent=2)
    power_kw = kart.get("power_limit_kw", NOMINAL_POWER_KW)
    if abs(power_kw - NOMINAL_POWER_KW) > 1e-6:
        pm = os.path.join(dst, "Kart_Motor.json")
        with open(pm) as f:
            dm = json.load(f)
        scale = power_kw / NOMINAL_POWER_KW
        dm["Maximum Engine Power"] = power_kw * 1000.0
        dm["Maximum Engine Torque"] *= scale
        with open(pm, "w") as f:
            json.dump(dm, f, indent=2)
    if not kart.get("mechanical_brake", True):
        pb = os.path.join(dst, "Kart_RearBrake.json")
        with open(pb) as f:
            db = json.load(f)
        db["Maximum Torque"] = 20.81113636363636
        with open(pb, "w") as f:
            json.dump(db, f, indent=2)
    return dst


class Plant:
    def __init__(self, kart, root="/opt/sim"):
        from harness import lines
        self.kart, self.root = kart, root
        self.dir = scaled_kart_dir(kart, root)
        self.name = os.path.basename(self.dir)
        with open(os.path.join(self.dir, "Kart_TireFront.json")) as f:
            tf = json.load(f)
            self.mu = tf["Coefficient of Friction"]
            self.kz = tf["Parameters"]["Vertical"]["Vertical Tire Stiffness [N/m]"]
        with open(os.path.join(self.dir, "Kart_Vehicle.json")) as f:
            self.wheelbase = json.load(f)["Wheelbase"]
        track = lines.load_track()
        pts = np.vstack([track.outer, track.inner])
        self.box = (pts.min(axis=0) - 50.0, pts.max(axis=0) + 50.0)
        with open(os.path.join(self.dir, "Kart_Column.json")) as f:
            self.col_max = json.load(f)["Pitman Arm"]["Maximum Angle (deg)"]  # RotaryArm turns input x this angle
        mounts = kart.get("mounts") or {}
        self.imu_front_pos = tuple((mounts.get("imu_front") or {}).get("pos_m", IMU_FRONT_POS))
        self.imu_back_pos = tuple((mounts.get("imu_back") or {}).get("pos_m", IMU_BACK_POS))
        self.v = self.terrain = None
        self.t0, self.t = 0.0, 0.0

    def spawn(self, x, y, yaw, v):
        """Builds a fresh vehicle with its rear-axle centre at (x, y), heading yaw, forward speed v; sim time carries on."""
        import pychrono as chrono
        import pychrono.vehicle as veh
        self.t0 = self.t
        veh.SetVehicleDataPath(os.path.dirname(self.dir) + "/")
        f = lambda n: veh.GetVehicleDataFile(self.name + "/" + n)
        car = veh.WheeledVehicle(f("Kart_Vehicle.json"), chrono.ChContactMethod_SMC)
        car.Initialize(chrono.ChCoordsysd(chrono.ChVector3d(x, y, SPAWN_Z), chrono.QuatFromAngleZ(yaw)), v)
        car.GetChassis().SetAerodynamicDrag(1.0, CDA, RHO)
        eng = veh.ReadEngineJSON(f("Kart_Motor.json"))
        tr = veh.ReadTransmissionJSON(f("Kart_Chain.json"))
        car.InitializePowertrain(veh.ChPowertrainAssembly(eng, tr))
        for axle, name in zip(car.GetAxles(), TYRES):
            for wheel in axle.GetWheels():
                car.InitializeTire(veh.ReadTireJSON(f(name)), wheel, chrono.VisualizationType_NONE)
        sys_ = car.GetSystem()
        sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        sys_.SetSolverType(chrono.ChSolver.Type_SPARSE_LU)
        sys_.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        terrain = veh.RigidTerrain(sys_)
        mat = chrono.ChContactMaterialSMC()
        mat.SetFriction(self.mu)
        (x0, y0), (x1, y1) = self.box
        terrain.AddPatch(mat, chrono.ChCoordsysd(chrono.ChVector3d((x0 + x1) / 2, (y0 + y1) / 2, 0), chrono.QUNIT),
                         float(x1 - x0), float(y1 - y0))
        terrain.Initialize()
        fwd = chrono.ChVector3d(v * math.cos(yaw), v * math.sin(yaw), 0.0)
        for b in sys_.GetBodies():  # Initialize gives only the front chassis the forward speed
            if not b.IsFixed():
                b.SetPosDt(fwd)
        for axle in car.GetAxles():
            for side, wh in enumerate(axle.GetWheels()):
                omega = v / wh.GetTire().GetRadius()
                wh.GetSpindle().SetAngVelLocal(chrono.ChVector3d(0.0, omega, 0.0))
                axle.m_suspension.GetAxle(side).SetPosDt(omega)
        self._chrono, self._veh = chrono, veh
        self.v, self.terrain = car, terrain
        self._sys = sys_
        self._body = car.GetChassis().GetBody()
        self._inp = veh.DriverInputs()
        front_w = list(car.GetAxles()[0].GetWheels())
        rear_w = list(car.GetAxles()[1].GetWheels())
        self._front_spindles = [w.GetSpindle() for w in front_w]
        self._rear_spindles = [w.GetSpindle() for w in rear_w]
        self._tires = [w.GetTire() for w in front_w + rear_w]
        self._grav = sys_.GetGravitationalAcceleration()
        self._vel = deque([(self.t, fwd.x, fwd.y, fwd.z)], maxlen=ACCEL_WINDOW + 1)

    def step(self, throttle, brake, column_deg):
        inp = getattr(self, "_inp", None) or self._veh.DriverInputs()
        inp.m_steering = float(column_deg) / self.col_max
        inp.m_throttle = float(throttle)
        inp.m_braking = float(brake)
        sys_ = getattr(self, "_sys", None) or self.v.GetSystem()
        tc = sys_.GetChTime()
        self.terrain.Synchronize(tc)
        self.v.Synchronize(tc, inp, self.terrain)
        self.terrain.Advance(STEP)
        self.v.Advance(STEP)
        self.t = self.t0 + sys_.GetChTime()
        body = getattr(self, "_body", None) or self.v.GetChassis().GetBody()
        u = body.GetPosDt()
        self._vel.append((self.t, u.x, u.y, u.z))

    def truth(self):
        """Ground truth: (x, y) is the rear-axle centre; body frame x forward, y left, z up; wheel angles positive left."""
        body = getattr(self, "_body", None) or self.v.GetChassis().GetBody()
        rot = body.GetRot()
        yaw = rot.GetCardanAnglesZYX().z
        rear_sp = getattr(self, "_rear_spindles", None) or [w.GetSpindle() for w in self.v.GetAxles()[1].GetWheels()]
        p0, p1 = rear_sp[0].GetPos(), rear_sp[1].GetPos()
        pv0, pv1 = rear_sp[0].GetPosDt(), rear_sp[1].GetPosDt()
        x, y, z = (p0.x + p1.x) / 2, (p0.y + p1.y) / 2, (p0.z + p1.z) / 2
        vwx, vwy = (pv0.x + pv1.x) / 2, (pv0.y + pv1.y) / 2
        c, s = math.cos(yaw), math.sin(yaw)
        w = body.GetAngVelLocal()
        (ta, *va), (tb, *vb) = self._vel[0], self._vel[-1]
        inv_dt = 1.0 / max(tb - ta, STEP)
        a_w = self._chrono.ChVector3d((vb[0] - va[0]) * inv_dt, (vb[1] - va[1]) * inv_dt, (vb[2] - va[2]) * inv_dt)
        a_b = rot.RotateBack(a_w)
        grav = getattr(self, "_grav", None) or self.v.GetSystem().GetGravitationalAcceleration()
        f_b = rot.RotateBack(a_w - grav)
        out = {"t": self.t, "x": x, "y": y, "z": z, "yaw": yaw, "vx": c * vwx + s * vwy, "vy": -s * vwx + c * vwy,
               "vx_world": vwx, "vy_world": vwy, "yaw_rate": w.z, "ax": a_b.x, "ay": a_b.y, "az": a_b.z,
               "gyro": [w.x, w.y, w.z], "accel": [f_b.x, f_b.y, f_b.z]}
        out["accel_front"] = imu_accel_at(out["accel"], out["gyro"], self.imu_front_pos)
        out["accel_back"] = imu_accel_at(out["accel"], out["gyro"], self.imu_back_pos)
        front_sp = getattr(self, "_front_spindles", None) or [w.GetSpindle() for w in self.v.GetAxles()[0].GetWheels()]
        for key, sp in zip(("wheel_l_deg", "wheel_r_deg"), front_sp):
            a = rot.RotateBack(sp.GetRot().GetAxisY())
            out[key] = math.degrees(math.atan2(-a.x, a.y))
        out["rear_axle_speed"] = (rear_sp[0].GetAngVelLocal().y + rear_sp[1].GetAngVelLocal().y) / 2 * R_ROLL
        tires = getattr(self, "_tires", None)
        if tires is not None:
            corners = [max(t.GetDeflection(), 0.0) * self.kz for t in tires]
        else:
            front = self.v.GetAxles()[0].GetWheels()
            rear = self.v.GetAxles()[1].GetWheels()
            corners = [max(wh.GetTire().GetDeflection(), 0.0) * self.kz for wh in front + rear]
        out["corners"] = corners
        fz_f = max(corners[0] + corners[1], 250.0)
        out["aligning_torque_nm"] = aligning_torque(
            out["vx"], out["vy"], out["yaw_rate"], out["wheel_l_deg"], out["wheel_r_deg"],
            self.wheelbase, self.mu, fz_f,
        )
        return out

    def steer_gain(self, col_deg=10.0, steps=200):
        """Mean front-wheel degrees per column degree near centre, probed on a parked copy of the kart; plant time is restored."""
        t = self.t
        self.spawn(0.0, 0.0, 0.0, 0.0)
        w = []
        for c in (0.0, col_deg, -col_deg):  # the first block lets the spawned kart settle onto its tyres
            for _ in range(steps):
                self.step(0.0, 1.0, c)
            tr = self.truth()
            w.append((tr["wheel_l_deg"] + tr["wheel_r_deg"]) / 2)
        self.t = t
        return (w[2] - w[1]) / (2 * col_deg)  # column + is right, wheel angles + are left

    def rear_axle_speed(self):
        """truth()["rear_axle_speed"] alone, for the steps where nothing else is read."""
        rear_sp = getattr(self, "_rear_spindles", None)
        if rear_sp is not None:
            return (rear_sp[0].GetAngVelLocal().y + rear_sp[1].GetAngVelLocal().y) / 2 * R_ROLL
        return sum(wh.GetSpindle().GetAngVelLocal().y for wh in self.v.GetAxles()[1].GetWheels()) / 2 * R_ROLL

    def close(self):
        shutil.rmtree(os.path.dirname(self.dir), ignore_errors=True)
