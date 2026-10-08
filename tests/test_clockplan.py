import math
from fractions import Fraction

from simnode.clockplan import ClockPlan, clock_plan

MS = 1_000_000


def instants(f, t_end_ms):
    """Every instant the sim publishes /clock: each 1 ms (2 physics steps) and each control tick end."""
    out, step, k = [], 0, 0
    while step * 500_000 < t_end_ms * MS:
        k += 1
        end = math.ceil(Fraction(k * 1_000_000_000) / (Fraction(f) * 500_000))
        while step < end:
            step += 1
            if step % 2 == 0 or step == end:
                out.append(step * 500_000)
    return out


def first_at_or_after(ins, deadlines):
    return {next(i for i in ins if i >= d) for d in deadlines}


def test_periodic_node_is_clocked_at_the_first_instant_after_each_deadline_only():
    for f in (60, 45):
        ins = instants(f, 1200)
        period = int(float(1.0 / f) * 1e9)
        plan = ClockPlan({"pathfinder_node": [period, 500 * MS], "imu_node": [10 * MS]})
        got = {n: [] for n in ("pathfinder_node", "imu_node")}
        for ns in ins:
            for n in plan.due(ns):
                got[n].append(ns)
        want_pf = first_at_or_after(ins, [k * period for k in range(1, ins[-1] // period + 1)] + [500 * MS, 1000 * MS])
        assert set(got["pathfinder_node"]) == want_pf and len(got["pathfinder_node"]) == len(want_pf)
        assert got["imu_node"] == [k * 10 * MS for k in range(1, 121)]


def test_poll_node_is_clocked_only_where_new_data_can_be_read():
    plan = ClockPlan({}, (5 * MS, ["gps_node", "localization_node"]))
    got = []
    for ns in range(MS, 80 * MS, MS):
        if plan.due(ns, writing=ns in (50 * MS, 62 * MS)):
            got.append(ns)
        if ns in (50 * MS, 62 * MS):
            plan.wrote(ns)
    # a write on a poll deadline races that poll, so the next one is clocked as well
    assert got == [50 * MS, 55 * MS, 65 * MS]
    assert plan.due(85 * MS) == []


def test_poll_followers_share_the_poll_instants():
    plan = ClockPlan({"localization_node": [10 * MS]}, (5 * MS, ["gps_node", "localization_node"]))
    plan.due(4 * MS)
    plan.wrote(4 * MS)
    assert plan.due(5 * MS) == ["gps_node", "localization_node"]
    assert plan.due(10 * MS) == ["localization_node"]


def test_clock_plan_reads_stack_params_with_recipe_overrides(tmp_path):
    p = tmp_path / "params"
    p.mkdir()
    (p / "gps.yaml").write_text("gps_node:\n  ros__parameters:\n    gps_frequency: 20\n    status_frequency: 10\n    serial_poll_hz: 200\n")
    (p / "imu.yaml").write_text("imu_node:\n  ros__parameters:\n    imu_frequency: 100\n    status_frequency: 1.0\n")
    (p / "e_comms.yaml").write_text("e_comms_node:\n  ros__parameters:\n    heartbeat_period_ms: 100.0\n")
    periods, poll = clock_plan(60, {"params": {"imu_node": {"imu_frequency": 50}}}, str(tmp_path))
    assert periods["pathfinder_node"] == [16_666_666, 500 * MS, 5000 * MS]
    assert periods["imu_node"] == [20 * MS, 1000 * MS]
    assert periods["localization_node"] == [20 * MS, 1000 * MS]
    assert periods["gps_node"] == [100 * MS]
    assert periods["e_comms_node"] == [100 * MS, 100 * MS, 5000 * MS]
    assert periods["master_api"] == [1000 * MS]
    assert poll == (5 * MS, ["gps_node", "localization_node"])


def test_clock_plan_at_tick_mult_4_clocks_pathfinder_at_240_hz(tmp_path):
    p = tmp_path / "params"
    p.mkdir()
    (p / "gps.yaml").write_text("gps_node:\n  ros__parameters:\n    gps_frequency: 20\n")
    one, four = clock_plan(60.0 * 1, {}, str(tmp_path)), clock_plan(60.0 * 4, {}, str(tmp_path))
    assert one == clock_plan(60, {}, str(tmp_path)) and one[0]["pathfinder_node"][0] == 16_666_666
    assert four[0]["pathfinder_node"] == [4_166_666, 500 * MS, 5000 * MS]
    assert {k: v for k, v in four[0].items() if k != "pathfinder_node"} == {k: v for k, v in one[0].items() if k != "pathfinder_node"}
