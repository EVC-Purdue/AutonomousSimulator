import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui"))
import fixture_data  # noqa: E402

from runner import jobs, store  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        root = tmp_path_factory.mktemp("results")
        ids = fixture_data.build(root, mp, full=False)
        yield root, ids


@pytest.fixture
def client(built, monkeypatch):
    root, ids = built
    monkeypatch.setattr(store, "RESULTS", str(root))
    fixture_data.install(monkeypatch, conditions=["nominal", "grip_low", "all_high"])
    from web import app
    monkeypatch.setattr(app, "_cache", {})   # each test starts without another's cached values
    return app.create_app().test_client(), ids


def test_pages(client):
    c, ids = client
    for url in ("/", "/leaderboard", f"/?job={ids['mpc-h60']}"):
        r = c.get(url)
        assert r.status_code == 200 and b"EVC Sim" in r.data


def test_jobs_list(client):
    c, ids = client
    rows = c.get("/api/jobs").get_json()
    assert rows[0]["status"] == "running" and len(rows) >= len(ids)
    assert {"id", "recipe", "tier", "user", "notes", "status", "submitted_at", "finished_at", "done", "total",
            "eta_s"} <= set(rows[0])
    assert rows[0]["eta_s"] > 0


def test_job_detail(client):
    c, ids = client
    d = c.get(f"/api/jobs/{ids['mpc-h60']}").get_json()
    assert {"job", "progress", "headline", "breakdown"} <= set(d)
    assert d["headline"]["cte_p90"]["worst_kart"] is None and d["headline"]["cte_p90"]["value"] is not None
    w = c.get(f"/api/jobs/{ids['mpc-h60']}?kart=worst").get_json()
    assert w["headline"]["cte_p90"]["worst_kart"] in ("kart1", "kart2", "kart3")
    assert w["headline"]["cte_p90"]["value"] >= d["headline"]["cte_p90"]["value"]
    cell = d["breakdown"]["laps"]["nominal"]["kart1"]["center"]
    assert {"cte_p90", "status", "cause", "key", "t"} <= set(cell)
    assert set(d["breakdown"]["clips"]) == {"oscillation", "offset_aligned", "misaligned_on_line",
                                            "recoverable_failure"}


def test_job_slice(client):
    c, ids = client
    jid = ids["mpc-h60"]
    eps = store.episodes(jid)
    d = c.get(f"/api/jobs/{jid}?kart=kart2").get_json()
    assert d["headline"]["n"]["value"] == sum(1 for e in eps if e["kart_name"] == "kart2")
    assert d["headline"]["cte_p90"]["per_kart"].keys() == {"kart2"}
    assert {k for ks in d["breakdown"]["laps"].values() for k in ks} == {"kart2"}
    assert {k for ks in d["breakdown"]["clips"].values() for k in ks} == {"kart2"}
    d = c.get(f"/api/jobs/{jid}?kart=kart1&line=line1").get_json()
    assert d["headline"]["n"]["value"] == sum(1 for e in eps if e["kart_name"] == "kart1"
                                              and e["scenario"].get("line") == "line1")
    assert d["breakdown"]["clips"] == {} and set(d["breakdown"]["laps"]["nominal"]["kart1"]) == {"line1"}
    assert d["options"]["cls"] == {c: False for c in d["options"]["cls"]} and d["options"]["kart"]["kart3"]
    d = c.get(f"/api/jobs/{jid}?kart=kart1&condition=grip_low").get_json()
    assert set(d["breakdown"]["laps"]) == {"grip_low"} and d["breakdown"]["clips"] == {}
    d = c.get(f"/api/jobs/{jid}?cls=oscillation").get_json()
    assert set(d["breakdown"]["clips"]) == {"oscillation"} and d["breakdown"]["laps"] == {}
    assert d["headline"]["laps_pct"]["value"] is None and d["headline"]["recovery_rate"]["value"] is not None
    assert d["options"]["condition"] == {"all_high": False, "grip_low": False, "nominal": True}


def test_job_empty_slice(client):
    c, ids = client
    for q in ("condition=grip_low&cls=oscillation", "kart=kart9"):
        r = c.get(f"/api/jobs/{ids['mpc-h60']}?{q}")
        d = r.get_json()
        assert r.status_code == 200 and d["headline"]["n"]["value"] == 0 and d["headline"]["cte_p90"]["value"] is None
        assert d["breakdown"] == {"laps": {}, "clips": {}}


def test_build_log(client):
    c, ids = client
    r = c.get(f"/api/jobs/{ids['mpc-broken']}/build_log")
    assert r.status_code == 200 and "SyntaxError" in r.get_data(as_text=True)


def test_episode(client):
    c, ids = client
    b = c.get(f"/api/jobs/{ids['pure-pursuit']}").get_json()["breakdown"]
    for key in (b["laps"]["nominal"]["kart2"]["line1"]["key"],
                b["clips"]["oscillation"]["kart1"]["episodes"][0]["key"]):
        e = c.get(f"/api/episodes/{ids['pure-pursuit']}/{key}").get_json()
        assert {"log", "events", "line", "track", "outcome", "context_valid"} <= set(e)
        log = e["log"]
        assert set(log) == {"t", "x", "y", "yaw", "wheel_l", "wheel_r", "cte", "wheel", "column", "speed", "cmd_speed",
                            "phase", "ay", "yaw_rate", "loc_err", "ay_mps2", "beta_deg", "rear_slip_deg", "front_slip_deg",
                            "rear_slip_ratio", "lat_front_pct", "lat_rear_pct", "slide_pct", "slide_front", "slide_rear"}
        assert all(len(log[k]) == len(log["t"]) for k in ("rear_slip_ratio", "slide_pct", "slide_front", "slide_rear"))
        assert "cost" not in e["plan"] and "cost" not in log and "cost_terms" not in log   # pure pursuit has no cost
        assert all(len(log[k]) == len(log["t"]) for k in ("ay_mps2", "beta_deg", "rear_slip_deg", "front_slip_deg"))
        d = sorted(abs(f - a) for f, a in zip(log["ay_mps2"], log["ay"]))
        assert d[len(d) // 2] < 0.2 and max(map(abs, log["ay_mps2"])) <= max(map(abs, log["ay"]))   # median filter: tracks, never overshoots
        lim = e["lat_limit"]
        assert lim["grip"] == pytest.approx(e["outcome"]["peak_mu"] * 9.81, abs=0.01)
        assert lim["envelope"] == pytest.approx(0.95 * lim["grip"], abs=0.01)
        assert set(e["unsafe"]) == {"lat", "rate", "travel", "spin", "off"}
        assert all(0 <= a < b <= log["t"][-1] + 0.1 for iv in e["unsafe"].values() for a, b in iv)
        pk = e["geom"]["peak_slip_deg"]
        assert set(pk) == {"front", "rear"} and all(8 < v < 12 for v in pk.values())
        assert len(log["yaw"]) == len(log["wheel_l"]) == len(log["wheel_r"]) == len(log["ay"]) == len(log["yaw_rate"]) == len(log["t"])
        assert all(abs((a + b) / 2 - w) < 0.02 for a, b, w in zip(log["wheel_l"], log["wheel_r"], log["wheel"]))
        g = e["geom"]
        assert g["wheelbase"] == (1.15 if e["kart"] == "kart2" else 1.05) and 1.0 < g["track_f"] < 1.2
        assert 1.0 < g["track_r"] < 1.3 and all(d > 0.2 and 0.1 < w < 0.2 for d, w in (g["tyre_f"], g["tyre_r"]))
        dt = [b - a for a, b in zip(log["t"], log["t"][1:])]
        assert min(dt) >= 0.049 and 0.0 in [round(t, 3) for t in log["t"]]
        assert {"outer", "inner"} == set(e["track"]) and len(e["line"]["x"]) == len(e["line"]["y"]) > 10
        assert len(e["track"]["outer"]) > 4000 and len(e["line"]["x"]) > 4000   # the densified splines the scoring uses
        assert e["events"][0]["kind"] == "handoff"
        p = e["plan"]
        assert len(p["t"]) == len(p["xy"]) > 10 and all(len(xy) % 2 == 0 and len(xy) >= 2 for xy in p["xy"])
        assert min(p["t"]) >= -0.01 and min(b - a for a, b in zip(p["t"], p["t"][1:])) >= 0.049   # after handoff, 20 Hz
        turns = [(w, col) for w, col in zip(log["wheel"], log["column"]) if abs(w) > 1]
        assert turns and all((w > 0) == (col > 0) for w, col in turns)   # left turn: both positive


def test_mpc_episode_carries_step_costs_cost_series_and_field(client):
    c, ids = client
    b = c.get(f"/api/jobs/{ids['mpc-h60']}").get_json()["breakdown"]
    key = b["laps"]["nominal"]["kart1"]["center"]["key"]
    e = c.get(f"/api/episodes/{ids['mpc-h60']}/{key}").get_json()
    p, log = e["plan"], e["log"]
    assert len(p["cost"]) == len(p["best"]) == len(p["t"]) and all(len(c_) == len(xy) // 2 - 1 for c_, xy in zip(p["cost"], p["xy"]))
    assert p["terms"][0] == "d" and len(p["terms"]) == 10
    assert p["field"]["d0"] == -7.0 and p["field"]["dd"] == 0.1 and len(p["field"]["c"]) == 141
    assert len(log["cost"]) == len(log["t"])
    terms = log["cost_terms"]   # per tick: the chosen trajectory's cost per term, summed over the horizon
    assert list(terms) == p["terms"] and all(len(v) == len(log["t"]) for v in terms.values())
    i = next(k for k, t in enumerate(log["t"]) if t > 1.0)
    j = min(range(len(p["t"])), key=lambda k: abs(p["t"][k] - log["t"][i]))
    assert sum(terms[k][i] for k in terms) == pytest.approx(sum(p["cost"][j]), abs=0.01)
    assert log["cost"][i] == p["best"][j] and all(v is None for v, t in zip(log["cost"], log["t"]) if t < -0.06)   # before handoff: no plan
    d = c.get(f"/api/episodes/{ids['mpc-h60']}/{key}/plan/{j}").get_json()
    assert d["terms"] == p["terms"] and len(d["steps"]) == len(p["cost"][j]) and len(d["steps"][0]) == 10
    assert [round(sum(r), 3) for r in d["steps"]] == [round(v, 3) for v in p["cost"][j]]


def test_episode_log_drops_the_parked_warm_up(client, monkeypatch):
    c, ids = client
    b = c.get(f"/api/jobs/{ids['mpc-h60']}").get_json()["breakdown"]
    key = b["laps"]["nominal"]["kart1"]["center"]["key"]
    real = store.read_log
    def with_warm_up(k):
        rows = real(k)
        return [dict(rows[0], t=rows[0]["t"] - 3.0 + i / 20, phase=0, auto=0) for i in range(60)] + rows
    monkeypatch.setattr(store, "read_log", with_warm_up)
    e = c.get(f"/api/episodes/{ids['mpc-h60']}/{key}").get_json()
    assert 0 not in e["log"]["phase"] and e["log"]["t"][0] == pytest.approx(real(key)[0]["t"] - (e["outcome"]["handoff_t"] or 0.0), abs=1e-3)


def test_slide_from_spin_and_slip_angles():
    from web.app import _slide
    geom = {"peak_sy": {"front": 0.16, "rear": 0.2}, "peak_sx": {"front": 0.1, "rear": 0.1}}
    slip = {"front_slip_deg": [math.degrees(math.atan(0.08)), None], "rear_slip_deg": [-math.degrees(math.atan(0.3)), None]}
    rows = [{"vx": 5.0, "rear_spin": 5.25}, {"vx": 0.5, "rear_spin": 0.0}]
    s = _slide(rows, slip, geom)
    assert s["rear_slip_ratio"] == [5.0, -50.0]   # (spin - vx) / max(|vx|, 1 m/s), %
    assert s["lat_front_pct"] == [50.0, None] and s["lat_rear_pct"] == [150.0, None]
    assert s["slide_pct"] == [150.0, 500.0] and s["slide_front"] == [0, 0] and s["slide_rear"] == [1, 1]


def test_slip_left_turn():
    from web.app import _slip
    left = {"vx": 5.0, "vy": -0.2, "yaw_rate": 0.5, "ay": 2.5, "wheel_l_deg": 8.0, "wheel_r_deg": 6.0}
    slow = dict(left, vx=0.5)
    s = _slip([left, slow], 1.05)
    rear, front = -math.degrees(math.atan2(-0.2, 5.0)), 7.0 - math.degrees(math.atan2(-0.2 + 0.5 * 1.05, 5.0))
    assert s["rear_slip_deg"][0] == -s["beta_deg"][0] == pytest.approx(rear, abs=0.01) and rear > 0
    assert s["front_slip_deg"][0] == pytest.approx(front, abs=0.01) and front > 0   # left turn: both positive, like steering
    assert s["beta_deg"][1] is None and s["rear_slip_deg"][1] is None and s["front_slip_deg"][1] is None


def test_submit(client):
    c, ids = client
    good = {"recipe_name": "mpc-h50", "ref": fixture_data.WT, "params": {"pathfinder_node": {"mpc.horizon_steps": 50}},
            "tier": "smoke", "where": "local", "user": "u", "notes": "n"}
    r = c.post("/api/jobs", json=good)
    assert r.status_code == 200
    jid = r.get_json()["id"]
    assert jobs.get(jid)["status"] == "queued" and jobs.get(jid)["recipe"]["params"]["pathfinder_node"]["mpc.horizon_steps"] == 50
    r = c.post("/api/jobs", json=dict(good, params="{bad"))
    assert r.status_code == 400 and "params" in r.get_json()["errors"]
    r = c.post("/api/jobs", json=dict(good, tier="dev", where="modal", user=""))
    assert r.status_code == 400 and {"where", "user"} <= set(r.get_json()["errors"])
    r = c.post("/api/jobs", json=dict(good, recipe_name=5, ref=["x"], user={"a": 1}))
    assert r.status_code == 400 and {"recipe_name", "ref", "user"} <= set(r.get_json()["errors"])


def test_compare_mixed(client):
    c, ids = client
    five = [ids[n] for n in ("pure-pursuit", "mpc-h60", "mpc-h40", "pp-lookahead-3", "mpc-h80")]
    d = c.get("/api/compare?ids=" + ",".join(five)).get_json()
    assert [j["id"] for j in d["jobs"]] == five
    names = [m["name"] for m in d["metrics"]]
    assert "cte_p90" in names and "recovery_rate" not in names
    assert all(len(m["values"]) == 5 and None not in m["values"] for m in d["metrics"])
    assert set(d["laps"]) == {"nominal"} and set(d["laps"]["nominal"]["kart1"]) == {"center", "line1"}   # lines shared by dev and smoke
    assert d["clips"] == {} and list(d["conditions"]) == ["nominal"] and len(d["conditions"]["nominal"]) == 5
    two = c.get(f"/api/compare?ids={ids['pure-pursuit']},{ids['mpc-h60']}").get_json()
    assert "recovery_rate" in [m["name"] for m in two["metrics"]] and set(two["clips"]) == set(
        c.get(f"/api/jobs/{ids['mpc-h60']}").get_json()["breakdown"]["clips"])


def test_compare_slice(client):
    c, ids = client
    two = [ids["pure-pursuit"], ids["mpc-h60"]]
    d = c.get("/api/compare?ids=" + ",".join(two) + "&kart=kart1&line=center").get_json()
    p90 = next(m for m in d["metrics"] if m["name"] == "cte_p90")["values"]
    assert p90 == [c.get(f"/api/jobs/{i}?kart=kart1&line=center").get_json()["headline"]["cte_p90"]["value"] for i in two]
    assert d["clips"] == {} and {k for ks in d["laps"].values() for k in ks} == {"kart1"}
    hd = [c.get(f"/api/jobs/{i}?kart=kart1&line=center").get_json()["headline"] for i in two]
    assert set(d["unsafe"]) == {"unsafe_lat", "unsafe_rate", "unsafe_travel", "unsafe_spin", "unsafe_off"}
    assert all(d["unsafe"][k] == [x[k]["value"] for x in hd] and None not in d["unsafe"][k] for k in d["unsafe"])
    assert d["options"]["line"] == {"center": True, "line1": True, "line4": True} and not any(d["options"]["cls"].values())


def test_compare_cells(client):
    c, ids = client
    two = [ids["pure-pursuit"], ids["mpc-h60"]]
    d = c.get("/api/compare?ids=" + ",".join(two) + "&kart=kart2").get_json()
    single = [c.get(f"/api/jobs/{i}?kart=kart2").get_json()["breakdown"] for i in two]
    cells = d["laps"]["grip_low"]["kart2"]["line1"]
    assert [x["key"] for x in cells] == [b["laps"]["grip_low"]["kart2"]["line1"]["key"] for b in single]
    assert {"cte_p90", "status", "cause", "key", "t"} <= set(cells[0])
    osc = d["clips"]["oscillation"]["kart2"]
    assert len(osc) == 2 and [x["recovery_rate"] for x in osc] == [b["clips"]["oscillation"]["kart2"]["recovery_rate"] for b in single]
    assert {e["clip_id"] for e in osc[0]["episodes"]} == {e["clip_id"] for e in single[0]["clips"]["oscillation"]["kart2"]["episodes"]}
    m = next(m for m in d["metrics"] if m["name"] == "cte_p90")
    assert m["values"] == [c.get(f"/api/jobs/{i}?kart=kart2").get_json()["headline"]["cte_p90"]["value"] for i in two]


def test_leaderboard_marks_old_engine(client, built):
    c, ids = client
    root, _ = built
    path = os.path.join(store.job_dir(ids["mpc-h40"]), "job.json")
    job = store.read_json(path)
    store.write_json(path, dict(job, engine_version="old"))
    try:
        rows = {r["job_id"]: r for r in c.get("/api/leaderboard?tier=smoke").get_json()}
        assert rows[ids["mpc-h40"]]["stale"] is True and rows[ids["pp-lookahead-3"]]["stale"] is False
    finally:
        store.write_json(path, job)


def test_compare_limits(client):
    c, ids = client
    six = list(ids.values())[:5] + [ids["mpc-h60"]]
    assert c.get("/api/compare?ids=" + ",".join(six)).status_code == 400
    assert c.get("/api/compare?ids=").status_code == 400


def test_leaderboard_sorted(client):
    c, ids = client
    rows = c.get("/api/leaderboard?sort=cte_p90&dir=asc").get_json()
    vals = [r["cte_p90"] for r in rows]
    assert len(rows) == 2 and vals == sorted(vals)
    rows = c.get("/api/leaderboard?sort=recovery_rate&dir=desc").get_json()
    assert [r["recovery_rate"] for r in rows] == sorted((r["recovery_rate"] for r in rows), reverse=True)
    for r in rows:
        h = c.get(f"/api/jobs/{r['job_id']}").get_json()["headline"]
        w = c.get(f"/api/jobs/{r['job_id']}?kart=worst").get_json()["headline"]
        assert r["cte_p90"] == h["cte_p90"]["value"] and r["worst"]["cte_p90"] == w["cte_p90"]["value"]


def test_meta(client):
    c, ids = client
    m = c.get("/api/meta").get_json()
    assert {r["name"] for r in m["recipes"]} >= {"pure-pursuit", "stack-default"}
    assert m["tiers"]["smoke"]["episodes"] == 3 * 3 + 4 * 3 * 3 and m["tiers"]["dev"]["episodes"] == 3 * 3 * 3 + 8 * 3
    assert m["tiers"]["dev"]["cost_usd"] > m["tiers"]["smoke"]["cost_usd"] > 0
    assert m["default_ref"].startswith("worktree:")
    assert m["tiers"]["dev"]["rows"] == 2 and m["tiers"]["smoke"]["rows"] == 2 and m["default_tier"] == "dev"


def test_leaderboard_per_tier(client):
    c, ids = client
    smoke = c.get("/api/leaderboard?tier=smoke&sort=cte_p90&dir=asc").get_json()
    assert {r["job_id"] for r in smoke} == {ids["mpc-h40"], ids["pp-lookahead-3"]}
    assert {r["job_id"] for r in c.get("/api/leaderboard?tier=dev").get_json()} == {ids["pure-pursuit"], ids["mpc-h60"]}
    assert c.get("/api/leaderboard?tier=nope").status_code == 400


def test_default_tier():
    from web.app import _default_tier
    big = {"large": {"episodes": 10692, "rows": 0}}
    assert _default_tier({"smoke": {"episodes": 15, "rows": 2}, "dev": {"episodes": 558, "rows": 1}, **big}) == "dev"
    assert _default_tier({"smoke": {"episodes": 15, "rows": 2}, "dev": {"episodes": 558, "rows": 0}, **big}) == "smoke"
    assert _default_tier({"smoke": {"episodes": 15, "rows": 0}, "dev": {"episodes": None, "rows": 0},
                          "large": {"episodes": None, "rows": 0}}) == "smoke"
    assert _default_tier({"smoke": {"episodes": 15, "rows": 2}, "dev": {"episodes": 558, "rows": 1},
                          "large": {"episodes": 10692, "rows": 3}}) == "large"


def test_favicon(client):
    c, ids = client
    for url in ("/favicon.ico", "/static/favicon.svg"):
        r = c.get(url)
        assert r.status_code == 200 and r.mimetype == "image/svg+xml" and b"<svg" in r.data
    assert b'rel="icon"' in c.get("/").data


def test_compare_labels_split_same_recipe():
    from web.app import _labels
    js = [{"recipe": "a", "where": "local", "submitted_at": "2026-10-05T09:38:55+00:00"},
          {"recipe": "a", "where": "modal", "submitted_at": "2026-10-05T10:11:29+00:00"},
          {"recipe": "b", "where": "local", "submitted_at": "2026-10-05T10:12:00+00:00"}]
    assert _labels(js) == ["a local", "a modal", "b"]
    js[1]["where"] = "local"
    import datetime
    hm = [datetime.datetime.fromisoformat(j["submitted_at"]).astimezone().strftime("%H:%M") for j in js[:2]]
    assert _labels(js) == ["a " + hm[0], "a " + hm[1], "b"]


def test_wall_estimate_prefers_modal_episodes(tmp_path, monkeypatch):
    from web.app import _wall_estimate
    monkeypatch.setattr(store, "RESULTS", str(tmp_path))
    for jid, where, wall in (("a", "local", 500.0), ("b", "modal", 1500.0)):
        os.makedirs(store.job_dir(jid))
        store.write_json(os.path.join(store.job_dir(jid), "job.json"), {"where": where})
        store._append(os.path.join(store.job_dir(jid), "episodes.jsonl"), {"wall_s": wall})
    assert _wall_estimate() == 1500.0


def test_submit_rejects_unknown_ros_param(client):
    c, ids = client
    bad = {"recipe_name": "x", "ref": fixture_data.WT, "params": {"pathfinder_node": {"horizon": 50}},
           "tier": "smoke", "where": "local", "user": "u", "notes": ""}
    r = c.post("/api/jobs", json=bad)
    assert r.status_code == 400 and "horizon" in r.get_json()["errors"]["params"]


def test_dev_needs_a_done_smoke(client):
    c, ids = client
    smoked = jobs.get(ids["mpc-h40"])["recipe"]
    d = {"recipe_name": "x", "ref": smoked["source"]["ref"], "params": smoked["params"], "tier": "dev",
         "where": "local", "user": "u", "notes": ""}
    assert c.post("/api/jobs", json=d).status_code == 200
    r = c.post("/api/jobs", json=dict(d, params={"pathfinder_node": {"mpc.horizon_steps": 41}}))
    assert r.status_code == 400 and "tier" in r.get_json()["errors"]


def test_meta_degrades_without_clips(client, monkeypatch):
    c, ids = client
    from harness import clips

    def gone(revision=None):
        raise FileNotFoundError("data/clips/index.json")
    monkeypatch.setattr(clips, "accepted_ids", gone)
    m = c.get("/api/meta")
    assert m.status_code == 200 and m.get_json()["tiers"]["dev"]["episodes"] is None


def test_failed_job_shows_its_error(client, tmp_path):
    c, ids = client
    jid = jobs.submit("smoke", {"name": "f", "source": {"repo": "AutonomousKart", "ref": "worktree:/x"}, "params": {},
                                "notes": ""}, "u", "")
    log = os.path.join(store.job_dir(jid), "error.txt")
    with open(log, "w") as f:
        f.write("Traceback: boom")
    jobs.update(jid, status="failed", build_log=log)
    r = c.get(f"/api/jobs/{jid}/build_log")
    assert r.status_code == 200 and "boom" in r.get_data(as_text=True)


def test_signals_stop_the_owned_modal_app(monkeypatch):
    import signal
    import web.app as web_app
    from runner import local
    handlers, stopped = {}, []
    monkeypatch.setattr(local.Runner, "start", lambda self: self)
    monkeypatch.setattr(local.Runner, "shutdown", lambda self: stopped.append(self))
    monkeypatch.setattr(web_app.app, "run", lambda **kw: None)
    monkeypatch.setattr(signal, "signal", lambda s, h: handlers.__setitem__(s, h))
    web_app.main()
    for s in (signal.SIGTERM, signal.SIGINT):
        with pytest.raises(SystemExit):
            handlers[s](s, None)
    assert len(stopped) == 2


def test_port_from_env(monkeypatch):
    import web.app as web_app
    from runner import local
    seen = {}
    monkeypatch.setattr(local.Runner, "start", lambda self: self)
    monkeypatch.setattr(web_app.app, "run", lambda **kw: seen.update(kw))
    monkeypatch.setattr(web_app.signal, "signal", lambda s, h: None)
    monkeypatch.setenv("WEB_PORT", "5051")
    web_app.main()
    assert seen["port"] == 5051


def test_submit_gautschi_smoke_and_dev(client):
    c, ids = client
    good = {"recipe_name": "g1", "ref": fixture_data.WT, "params": {}, "tier": "smoke", "where": "gautschi",
            "user": "u", "notes": "n"}
    r = c.post("/api/jobs", json=good)
    assert r.status_code == 200 and jobs.get(r.get_json()["id"])["where"] == "gautschi"
    r = c.post("/api/jobs", json=dict(good, tier="dev"))
    assert r.status_code == 400 and set(r.get_json()["errors"]) == {"tier"}   # dev only after a done smoke
    r = c.post("/api/jobs", json=dict(good, where="cpu"))
    assert r.status_code == 400 and "where" in r.get_json()["errors"]


def test_submit_bad_ref_names_ref_and_meta_lists_refs(client, monkeypatch):
    from runner import local

    def resolve(ref):
        raise ValueError("Unknown ref " + ref)
    monkeypatch.setattr(local, "resolve_ref", resolve)
    monkeypatch.setattr(local, "recent_refs", lambda: ["origin/newer", "main"])
    c, ids = client
    r = c.post("/api/jobs", json={"recipe_name": "x", "ref": "nope", "params": {}, "tier": "smoke", "where": "local",
                                  "user": "u", "notes": ""})
    assert r.status_code == 400 and r.get_json()["errors"] == {"ref": "Unknown ref nope"}
    assert c.get("/api/meta").get_json()["refs"] == ["origin/newer", "main"]


def test_cuda_recipe_needs_the_gpu_backend(client):
    c, ids = client
    good = {"recipe_name": "mpc-cuda-8192", "ref": fixture_data.WT, "params": {}, "sim": {"cuda": True},
            "tier": "smoke", "where": "gautschi", "user": "u", "notes": ""}
    r = c.post("/api/jobs", json=good)
    assert r.status_code == 400 and set(r.get_json()["errors"]) == {"where"}
    r = c.post("/api/jobs", json=dict(good, where="gautschi-gpu"))
    assert r.status_code == 200 and jobs.get(r.get_json()["id"])["recipe"]["sim"] == {"cuda": True}


def test_large_tier_submits_and_has_a_board(client):
    c, ids = client
    r = c.post("/api/jobs", json={"recipe_name": "A26 MPC stack", "ref": fixture_data.WT, "params": {}, "sim": {"cuda": True},
                                  "tier": "large", "where": "gautschi-gpu", "user": "u", "notes": ""})
    assert r.status_code == 200 and jobs.get(r.get_json()["id"])["tier"] == "large"
    assert "large" in c.get("/api/meta").get_json()["tiers"]
    assert c.get("/api/leaderboard?tier=large").status_code == 200


def test_modal_gpu_takes_cuda_recipes_on_smoke_and_dev(client):
    c, ids = client
    good = {"recipe_name": "mpc-cuda-8192", "ref": fixture_data.WT, "params": {}, "sim": {"cuda": True},
            "tier": "smoke", "where": "modal-gpu", "user": "u", "notes": ""}
    r = c.post("/api/jobs", json=good)
    assert r.status_code == 200 and jobs.get(r.get_json()["id"])["where"] == "modal-gpu"
    r = c.post("/api/jobs", json=dict(good, tier="large"))
    assert r.status_code == 400 and set(r.get_json()["errors"]) == {"where"}
    r = c.post("/api/jobs", json=dict(good, where="modal"))
    assert r.status_code == 400 and set(r.get_json()["errors"]) == {"where"}
    from web.app import _parse_submit
    assert "where" not in _parse_submit(dict(good, tier="dev"))[0]


def test_loc_rms_autonomous_after_first_odom(client):
    from web import app as web_app
    rows = [{"ekf_err_m": 0.0, "phase": 1, "auto": 0}, {"ekf_err_m": 9.0, "phase": 1, "auto": 0},
            {"ekf_err_m": 0.0, "auto": 1}, {"ekf_err_m": 3.0, "phase": 2, "auto": 1}, {"ekf_err_m": 4.0, "phase": 2, "auto": 1}]
    s = web_app._loc_stats(rows)   # manual phase dropped; a 0.0 after the first odom is a real sample
    assert s["n"] == 3 and s["ss"] == pytest.approx(25.0) and s["rms"] == pytest.approx(math.sqrt(25 / 3), abs=1e-3)
    assert web_app._loc_stats([{"ekf_err_m": 0.0, "phase": 2, "auto": 1}]) is None   # no odom yet
    c, ids = client
    d = c.get(f"/api/jobs/{ids['mpc-h60']}").get_json()["headline"]["loc_rms"]
    assert d["value"] == pytest.approx(0.05, abs=1e-3) and set(d["per_kart"]) == {"kart1", "kart2", "kart3"}
    assert os.path.exists(os.path.join(store.job_dir(ids["mpc-h60"]), "loc_rms.json")) == (jobs.get(ids["mpc-h60"])["status"] == "done")
    assert c.get(f"/api/jobs/{ids['mpc-h60']}?kart=worst").get_json()["headline"]["loc_rms"]["worst_kart"] in d["per_kart"]
    key = c.get(f"/api/jobs/{ids['mpc-h60']}").get_json()["breakdown"]["laps"]["nominal"]["kart1"]["center"]["key"]
    assert c.get(f"/api/episodes/{ids['mpc-h60']}/{key}").get_json()["loc"] == {"rms": 0.05, "p95": 0.05}
    assert c.get(f"/api/jobs/{ids['mpc-h80']}").get_json()["headline"]["loc_rms"]["value"] is not None
    assert not os.path.exists(os.path.join(store.job_dir(ids["mpc-h80"]), "loc_rms.json"))   # running: never cached incomplete


def test_password_gates_submit_and_delete(client, monkeypatch):
    c, ids = client
    good = {"recipe_name": "pw", "ref": fixture_data.WT, "params": {}, "tier": "smoke", "where": "local", "user": "u", "notes": ""}
    monkeypatch.setenv("AUTONOMOUS_SIM_PASSWORD", "s3cret")
    assert c.get("/api/meta").get_json()["password"] is True and c.get("/api/jobs").status_code == 200   # viewing stays open
    for body, headers in ((good, {}), (dict(good, password="nope"), {}), (good, {"X-Password": "nope"}), (dict(good, password=5), {})):
        r = c.post("/api/jobs", json=body, headers=headers)
        assert r.status_code == 403 and r.get_json()["errors"] == {"password": "Wrong or missing password"}
    r = c.post("/api/jobs", json=dict(good, password="s3cret"))
    assert r.status_code == 200 and "s3cret" not in open(os.path.join(store.job_dir(r.get_json()["id"]), "job.json")).read()
    jid = c.post("/api/jobs", json=good, headers={"X-Password": "s3cret"}).get_json()["id"]
    assert c.delete(f"/api/jobs/{jid}").status_code == 403 and store.has_job(jid)
    assert c.delete(f"/api/jobs/{jid}", headers={"X-Password": "s3cret"}).status_code == 200 and not store.has_job(jid)
    monkeypatch.delenv("AUTONOMOUS_SIM_PASSWORD")
    assert c.post("/api/jobs", json=good).status_code == 200 and c.get("/api/meta").get_json()["password"] is False


def test_submit_ideas_takes_cuda_recipes(client):
    c, ids = client
    r = c.post("/api/jobs", json={"recipe_name": "i1", "ref": fixture_data.WT, "params": {}, "tier": "smoke", "where": "ideas",
                                  "user": "u", "notes": "", "sim": {"cuda": True}})
    assert r.status_code == 200 and jobs.get(r.get_json()["id"])["where"] == "ideas"


def test_submit_ideas_defaults_to_cuda(client):
    c, ids = client
    body = {"recipe_name": "i2", "ref": fixture_data.WT, "params": {}, "tier": "smoke", "user": "u", "notes": ""}
    j = jobs.get(c.post("/api/jobs", json=dict(body, where="ideas")).get_json()["id"])
    assert j["recipe"]["sim"] == {"cuda": True}
    j = jobs.get(c.post("/api/jobs", json=dict(body, where="ideas", sim={"cuda": False})).get_json()["id"])
    assert j["recipe"]["sim"] == {"cuda": False}
    j = jobs.get(c.post("/api/jobs", json=dict(body, recipe_name="i3", where="local")).get_json()["id"])
    assert "cuda" not in j["recipe"].get("sim", {})


def test_meta_falls_back_to_main_without_the_worktrees(client, monkeypatch):
    from web import app as web_app
    c, ids = client
    monkeypatch.setattr(web_app, "_worktree_ok", lambda ref: not ref.startswith("worktree:"))
    m = c.get("/api/meta").get_json()
    assert m["default_ref"] == "main" and m["worktrees"] is False
    assert not web_app._worktree_ok("worktree:/no/such/dir") and web_app._worktree_ok("main")


def test_gunicorn_entry_starts_the_runner(monkeypatch):
    import web.app as web_app
    from runner import local
    started = []
    monkeypatch.setattr(local.Runner, "start", lambda self: started.append(self.workers) or self)
    assert web_app.serve() is web_app.app and started == [1]
