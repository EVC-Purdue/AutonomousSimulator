"""Runs tools/profile_episode.py on Modal (runner/modal_app.py image plus py-spy): kart 1 nominal center lap, stack-default.

.venv/bin/python tools/profile_modal.py <out_dir> [all] [deadline]
Each clock mode is one Modal call; outputs (profile.json, pyspy_*.txt, stdout.txt) land in <out_dir>/<mode>/.
"""
import os
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = None  # the container gets its image from the app definition; runner/ exists only locally
if modal.is_local():
    sys.path.insert(0, ROOT)
    from runner import modal_app
    image = (modal_app.image.run_commands("pip3 install py-spy")
             .add_local_file(os.path.join(ROOT, "tools", "profile_episode.py"), "/prof/profile_episode.py", copy=True))
app = modal.App("evc-prof", image=image)


@app.function(cpu=2, memory=3072, timeout=900)
def profile(batch, mode):
    import json
    import subprocess
    os.makedirs("/out", exist_ok=True)
    with open("/batch.json", "w") as f:
        json.dump(batch, f)
    r = subprocess.run(["bash", "-lc", "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash"
                        f" && python3 /prof/profile_episode.py /batch.json /out --clock {mode} --pyspy-s 40"],
                       capture_output=True, text=True)
    files = {"stdout.txt": (r.stdout + r.stderr).encode()}
    for n in os.listdir("/out"):
        p = os.path.join("/out", n)
        if os.path.isfile(p):
            files[n] = open(p, "rb").read()
    files["nproc.txt"] = subprocess.run(["bash", "-lc", "nproc; cat /sys/fs/cgroup/cpu.max 2>&1; lscpu | head -20"],
                                        capture_output=True, text=True).stdout.encode()
    return files


def main():
    from harness import config
    out, modes = sys.argv[1], sys.argv[2:] or ["all"]
    r = config.resolve_episode("kart1", "nominal", "stack-default", {"type": "lap", "line": "center", "laps": 1}, 0, "smoke")
    batch = [{"key": config.cache_key(r), "resolved": r}]
    app_id = None
    try:
        with app.run():
            app_id = app.app_id
            calls = {m: profile.spawn(batch, m) for m in modes}
            for m, c in calls.items():
                d = os.path.join(out, m)
                os.makedirs(d, exist_ok=True)
                for n, b in c.get().items():
                    with open(os.path.join(d, n), "wb") as f:
                        f.write(b)
                print(m, "done", flush=True)
    finally:
        if app_id:
            from runner import local
            local.stop_modal_app(app_id)  # app.run() stops it on exit; this is the backstop


if __name__ == "__main__":
    main()
