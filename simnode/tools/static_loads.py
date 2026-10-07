"""Settles the kart on flat ground and compares Chrono wheel loads to the measured corner weights."""
import json, os, sys
import pychrono as chrono, pychrono.vehicle as veh
DATA = sys.argv[1]   # directory holding the kart dirs
K = sys.argv[2] if len(sys.argv) > 2 else "kart"
veh.SetVehicleDataPath(DATA + "/")
v = veh.WheeledVehicle(veh.GetVehicleDataFile(K + "/Kart_Vehicle.json"), chrono.ChContactMethod_SMC)
v.Initialize(chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0.01), chrono.QUNIT), 0.0)
eng = veh.ReadEngineJSON(veh.GetVehicleDataFile(K + "/Kart_Motor.json")); tr = veh.ReadTransmissionJSON(veh.GetVehicleDataFile(K + "/Kart_Chain.json"))
v.InitializePowertrain(veh.ChPowertrainAssembly(eng, tr))
for axle, f in zip(v.GetAxles(), (K + "/Kart_TireFront.json", K + "/Kart_TireRear.json")):
    for w in axle.GetWheels(): v.InitializeTire(veh.ReadTireJSON(veh.GetVehicleDataFile(f)), w, chrono.VisualizationType_NONE)
v.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
v.GetSystem().SetSolverType(chrono.ChSolver.Type_SPARSE_LU)
v.GetSystem().SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
terr = veh.RigidTerrain(v.GetSystem()); mat = chrono.ChContactMaterialSMC(); mat.SetFriction(0.91)
terr.AddPatch(mat, chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0), chrono.QUNIT), 200, 200); terr.Initialize()
inp = veh.DriverInputs(); inp.m_braking = 1.0
for k in range(int(3.0 / 5e-4)):
    t = v.GetSystem().GetChTime(); terr.Synchronize(t); v.Synchronize(t, inp, terr); terr.Advance(5e-4); v.Advance(5e-4)
names = ["FL", "FR", "RL", "RR"]   # Chrono wheel order per axle: left, right
ws = v.GetAxles()[0].GetWheels() + v.GetAxles()[1].GetWheels()
loads = {n: max(w.GetTire().GetDeflection(), 0) * 100000.0 for n, w in zip(names, ws)}
tot = sum(loads.values())
meas = dict(FL=56.18, FR=64.56, RL=63.62, RR=89.78); mt = sum(meas.values())
print("mass %.2f kg, weight %.1f N, sum of wheel loads %.1f N" % (v.GetMass(), v.GetMass() * 9.81, tot))
for n in names: print(f"{n}: chrono {loads[n]/tot*100:5.2f}%   measured {meas[n]/mt*100:5.2f}%")
s = lambda d, a, b: (d[a] + d[b]) / sum(d.values()) * 100
print(f"front {s(loads,'FL','FR'):.2f}% vs {s(meas,'FL','FR'):.2f}%; right {s(loads,'FR','RR'):.2f}% vs {s(meas,'FR','RR'):.2f}%; cross FR+RL {s(loads,'FR','RL'):.2f}% vs {s(meas,'FR','RL'):.2f}%")
