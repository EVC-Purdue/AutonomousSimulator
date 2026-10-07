"""Chrono kart plant: grip and frame scaling of the kart JSONs, spawn, 0.5 ms steps and ground-truth state."""
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


def scaled_kart_dir(kart, root):
    """Temp copy of kart["chrono_dir"] with tyre grip and frame torsion scaled; returns the copied kart dir."""
    src = os.path.join(root, kart["chrono_dir"])
    dst = os.path.join(tempfile.mkdtemp(prefix="kart_"), os.path.basename(os.path.normpath(src)))
    shutil.copytree(src, dst)
    grip = kart["tyres"]["grip"]
    for name in TYRES:
        p = os.path.join(dst, name)
        with open(p) as f:
            d = json.load(f)
        d["Coefficient of Friction"] *= grip
        for block in ("Longitudinal", "Lateral"):
            for k, v in d["Parameters"][block].items():
                if k.startswith(SCALED):
                    d["Parameters"][block][k] = [x * grip for x in v]
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
    p = os.path.join(dst, "Kart_TorsionConnector.json")
    with open(p) as f:
        d = json.load(f)
    d["Torsional Stiffness"] *= kart["frame"]["torsion_scale"]
    with open(p, "w") as f:
        json.dump(d, f, indent=2)
    return dst


class Plant:
    def __init__(self, kart, root="/opt/sim"):
        from harness import lines
        self.kart, self.root = kart, root
        self.dir = scaled_kart_dir(kart, root)
        self.name = os.path.basename(self.dir)
        with open(os.path.join(self.dir, "Kart_TireFront.json")) as f:
            self.mu = json.load(f)["Coefficient of Friction"]
        with open(os.path.join(self.dir, "Kart_Vehicle.json")) as f:
            self.wheelbase = json.load(f)["Wheelbase"]
        with open(os.path.join(self.dir, "Kart_TireFront.json")) as f:
            self.kz = json.load(f)["Parameters"]["Vertical"]["Vertical Tire Stiffness [N/m]"]
        track = lines.load_track()
        pts = np.vstack([track.outer, track.inner])
        self.box = (pts.min(axis=0) - 50.0, pts.max(axis=0) + 50.0)
        with open(os.path.join(self.dir, "Kart_Column.json")) as f:
            self.col_max = json.load(f)["Pitman Arm"]["Maximum Angle (deg)"]  # RotaryArm turns input x this angle
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
        self._vel = deque([(self.t, fwd.x, fwd.y, fwd.z)], maxlen=ACCEL_WINDOW + 1)

    def step(self, throttle, brake, column_deg):
        inp = self._veh.DriverInputs()
        inp.m_steering = float(column_deg) / self.col_max
        inp.m_throttle = float(throttle)
        inp.m_braking = float(brake)
        tc = self.v.GetSystem().GetChTime()
        self.terrain.Synchronize(tc)
        self.v.Synchronize(tc, inp, self.terrain)
        self.terrain.Advance(STEP)
        self.v.Advance(STEP)
        self.t = self.t0 + self.v.GetSystem().GetChTime()
        u = self.v.GetChassis().GetBody().GetPosDt()
        self._vel.append((self.t, u.x, u.y, u.z))

    def truth(self):
        """Ground truth: (x, y) is the rear-axle centre; body frame x forward, y left, z up; wheel angles positive left."""
        car = self.v
        body = car.GetChassis().GetBody()
        rot = body.GetRot()
        yaw = rot.GetCardanAnglesZYX().z
        rear = car.GetAxles()[1].GetWheels()
        p = [w.GetSpindle().GetPos() for w in rear]
        pv = [w.GetSpindle().GetPosDt() for w in rear]
        x, y, z = (p[0].x + p[1].x) / 2, (p[0].y + p[1].y) / 2, (p[0].z + p[1].z) / 2
        vwx, vwy = (pv[0].x + pv[1].x) / 2, (pv[0].y + pv[1].y) / 2
        c, s = math.cos(yaw), math.sin(yaw)
        w = body.GetAngVelLocal()
        (ta, *va), (tb, *vb) = self._vel[0], self._vel[-1]
        a_w = self._chrono.ChVector3d(*[(q - p) / max(tb - ta, STEP) for p, q in zip(va, vb)])
        a_b = rot.RotateBack(a_w)
        f_b = rot.RotateBack(a_w - car.GetSystem().GetGravitationalAcceleration())
        out = {"t": self.t, "x": x, "y": y, "z": z, "yaw": yaw, "vx": c * vwx + s * vwy, "vy": -s * vwx + c * vwy,
               "vx_world": vwx, "vy_world": vwy, "yaw_rate": w.z, "ax": a_b.x, "ay": a_b.y, "az": a_b.z,
               "gyro": [w.x, w.y, w.z], "accel": [f_b.x, f_b.y, f_b.z]}
        front = car.GetAxles()[0].GetWheels()
        for key, wh in zip(("wheel_l_deg", "wheel_r_deg"), front):
            a = rot.RotateBack(wh.GetSpindle().GetRot().GetAxisY())
            out[key] = math.degrees(math.atan2(-a.x, a.y))
        out["rear_axle_speed"] = self.rear_axle_speed()
        out["corners"] = [max(wh.GetTire().GetDeflection(), 0.0) * self.kz for wh in front + rear]
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
        return sum(wh.GetSpindle().GetAngVelLocal().y for wh in self.v.GetAxles()[1].GetWheels()) / 2 * R_ROLL

    def close(self):
        shutil.rmtree(os.path.dirname(self.dir), ignore_errors=True)
