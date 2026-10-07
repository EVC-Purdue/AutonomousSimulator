import contextlib
import gzip
import io
import os
import subprocess
import tarfile

import pytest

from harness import config, tiers
from runner import jobs, local, modal_app

REAL = {n: getattr(local, n) for n in ("resolve_ref", "missing_sim_support")}   # before fake_docker patches them
LAUNCH = "src/autonomous_kart/autonomous_kart/launch/bringup_simdev.launch.py"
PARAMS = "src/autonomous_kart/autonomous_kart/params/pathfinder.yaml"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, path, text, msg, date=None):
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _git(repo, "add", path)
    env = dict(os.environ, GIT_COMMITTER_DATE=date) if date else None
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg],
                   check=True, env=env)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def ak(tmp_path, monkeypatch):
    """An upstream repo and a clone of it as the AutonomousKart checkout; main has the sim-support launch file."""
    up = tmp_path / "up"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, PARAMS, "pathfinder_node:\n  ros__parameters:\n    mpc.num_samples: 1\n", "params")
    main = _commit(up, LAUNCH, "# sim\n", "sim support")
    _git(up, "tag", "v1")
    clone = tmp_path / "ak"
    subprocess.run(["git", "clone", "-q", str(up), str(clone)], check=True)
    monkeypatch.setattr(local, "AK_REPO", str(clone))
    return {"up": up, "clone": clone, "main": main}


@pytest.fixture
def real_git(fake_docker, monkeypatch):
    """fake_docker's builds and batches, with the real ref resolution and sim-support check."""
    for n, f in REAL.items():
        monkeypatch.setattr(local, n, f)
    return fake_docker


def _recipe(ref, name="r1"):
    return {"name": name, "source": {"repo": "AutonomousKart", "ref": ref}, "params": {}, "notes": "n"}


def test_resolves_branch_tag_sha_and_origin_branch(ak):
    assert local.resolve_ref("main") == ak["main"] and local.resolve_ref("v1") == ak["main"]
    assert local.resolve_ref(ak["main"][:10]) == ak["main"] and local.resolve_ref("worktree:/x") is None
    _git(ak["up"], "checkout", "-q", "-b", "feature-x")
    fx = _commit(ak["up"], "src/f.txt", "f", "feature")
    assert local.resolve_ref("feature-x") == fx and local.resolve_ref("origin/feature-x") == fx   # fetched at submit
    assert _git(ak["clone"], "rev-parse", "--abbrev-ref", "HEAD") == "main"   # the checkout is never touched
    assert "feature-x" not in _git(ak["clone"], "branch", "--format=%(refname:short)").split()


@pytest.mark.parametrize("bad", ["nope", "-x", "main:src", ""])
def test_bad_ref_raises(ak, bad):
    with pytest.raises(ValueError):
        local.resolve_ref(bad)


def test_submit_stores_commit_and_a_moved_branch_gets_new_keys(ak, tmp_results):
    a = jobs.submit("smoke", _recipe("main"), "u", "x")
    first = _commit(ak["clone"], "src/g.txt", "g", "move main")   # a local branch wins over origin/<name>
    b = jobs.submit("smoke", _recipe("main"), "u", "x")
    ra, rb = jobs.get(a)["recipe"]["source"], jobs.get(b)["recipe"]["source"]
    assert ra == {"repo": "AutonomousKart", "ref": "main", "commit": ak["main"]}
    assert rb["ref"] == "main" and rb["commit"] == first != ak["main"]
    keys = [{config.cache_key(e) for e in tiers.expand("smoke", dict(_recipe("main"), source=dict(s, tree=s["commit"])))}
            for s in (ra, rb)]
    assert not keys[0] & keys[1]


def test_job_builds_the_submitted_commit_not_the_moved_branch(ak, tmp_results, real_git, monkeypatch):
    built = []
    monkeypatch.setattr(local, "build_stack", lambda ref, h: built.append(h) or real_git.build_stack(ref, h))
    jid = jobs.submit("smoke", _recipe("main"), "u", "x")
    _commit(ak["up"], "src/h.txt", "h", "move main")
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "done" and built == [ak["main"]]


def test_branch_without_sim_support_is_build_failed(ak, tmp_results, real_git):
    _git(ak["up"], "checkout", "-q", "-b", "old", ak["main"] + "~1")
    _commit(ak["up"], "src/o.txt", "o", "no sim support")
    jid = jobs.submit("smoke", _recipe("old"), "u", "x")
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "build_failed" and "bringup_simdev.launch.py" in open(job["build_log"]).read()
    assert real_git.build_calls == 0


def test_recent_refs_newest_first(ak):
    _git(ak["up"], "checkout", "-q", "-b", "newer")
    _commit(ak["up"], "src/n.txt", "n", "newer", date="2099-01-01T00:00:00")
    local.resolve_ref("newer")   # fetches
    refs = local.recent_refs()
    assert refs[0] == "origin/newer" and "main" in refs and "origin/main" in refs and "origin/HEAD" not in refs
    assert "origin" not in refs


def test_modal_ships_the_commit_archive(ak, tmp_results, real_git, monkeypatch):
    calls = []

    class R:
        def remote(self, eps, src=None):
            calls.append(src)
            from tests.test_modal import FakeRemote
            return FakeRemote().remote(eps)
    monkeypatch.setattr(modal_app, "run_batch", R())
    monkeypatch.setattr(modal_app.app, "run", contextlib.nullcontext)
    jid = jobs.submit("smoke", _recipe("v1"), "u", "x", "modal")
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "done" and calls and all(c == calls[0] for c in calls)
    names = tarfile.open(fileobj=io.BytesIO(gzip.decompress(calls[0])), mode="r:").getnames()
    assert LAUNCH in names


def test_modal_baked_worktree_sends_no_archive(tmp_results, fake_docker, monkeypatch):
    calls = []
    from tests.test_modal import FakeRemote
    fr = FakeRemote()
    monkeypatch.setattr(modal_app, "run_batch", type("R", (), {"remote": lambda self, eps, *a: calls.append(a) or fr.remote(eps)})())
    monkeypatch.setattr(modal_app.app, "run", contextlib.nullcontext)
    jobs.submit("smoke", _recipe(modal_app.REF), "u", "x", "modal")
    local.Runner(workers=1).run_once()
    assert calls and all(a == () for a in calls)


def test_gautschi_builds_per_commit(ak, tmp_results, real_git, gautschi_remote):
    jid = jobs.submit("smoke", _recipe("main"), "u", "x", "gautschi")
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "done"
    assert f"AutoSim/stacks/{ak['main']}" in gautschi_remote.scripts[0]
    assert any(c[0] == "rsync" and c[-1].endswith(f"AutoSim/stacks/{ak['main']}/") for c in gautschi_remote.calls)
