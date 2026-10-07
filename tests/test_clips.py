import collections

from harness import clips


def test_accepted_counts():
    ids = clips.accepted_ids()
    assert len(ids) == 129 and ids == sorted(ids)
    got = collections.Counter(clips.load(i)["cls"] for i in ids)
    assert got == {"misaligned_on_line": 37, "offset_aligned": 33, "recoverable_failure": 31, "oscillation": 28}


def test_every_clip_has_prehandoff_and_times():
    for i in clips.accepted_ids():
        c = clips.load(i)
        assert c["series"]["t"][0] < 0 and "handoff_t" in c and "end_t" in c


def test_series_at_handoff_matches_sample():
    for i in clips.accepted_ids():
        c = clips.load(i)
        s = c["series"]
        k = min(range(len(s["t"])), key=lambda j: abs(s["t"][j]))
        assert abs(s["t"][k]) < 1e-6
        got = clips.series_at(c, 0.0)
        for key in ("x", "y", "v", "cte", "steer"):
            assert abs(got[key] - s[key][k]) < 1e-6


def test_series_at_unwraps_yaw():
    c = {"series": {"t": [0.0, 1.0], "yaw": [3.1, -3.1]}}
    y = clips.series_at(c, 0.5)["yaw"]
    assert abs(abs(y) - 3.14159) < 0.05


def test_missing_clips_are_fetched_at_the_tier_revision(tmp_path, monkeypatch):
    import json
    import huggingface_hub
    calls = []

    def fake(repo_id, repo_type, revision, allow_patterns, local_dir):
        calls.append((repo_id, repo_type, revision))
        d = tmp_path / "clips"
        d.mkdir(exist_ok=True)
        (d / "index.json").write_text(json.dumps([{"id": "c2"}, {"id": "c1"}]))
        (d / "c1.json").write_text(json.dumps({"id": "c1"}))

    monkeypatch.setattr(clips, "CLIPS", str(tmp_path / "clips"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert clips.accepted_ids("rev1") == ["c1", "c2"]
    assert calls == [("ShayManor/Sim_data", "dataset", "rev1")]
    (tmp_path / "clips" / "c1.json").unlink()
    assert clips.load("c1") == {"id": "c1"} and len(calls) == 2
