"""python3 -m simnode.batch <episodes.json> <out_root>: runs each resolved episode into <out_root>/<cache_key>/. Optional third arg: ROS_DOMAIN_ID."""
import json
import os
import sys
import traceback


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0]) as f:
        episodes = json.load(f)
    if len(argv) > 2:
        os.environ["ROS_DOMAIN_ID"] = argv[2]  # before simnode.stack copies the environment
    from simnode.episode import run_episode
    for item in episodes:   # {"key": host cache key, "resolved": episode}
        resolved = item["resolved"]
        out = os.path.join(argv[1], item["key"])
        try:
            o = run_episode(resolved, out)
            print(f"{os.path.basename(out)} {o['status']} {o['cause']}", flush=True)
        except Exception:
            traceback.print_exc()
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "outcome.json"), "w") as f:
                json.dump({"status": "error", "cause": "exception", "t_end": 0.0, "context_valid": None, "wall_s": 0.0,
                           "sim_s": 0.0, "handoff_t": None, "laps_completed": 0, "laps_attempted": 0, "peak_mu": None}, f)


if __name__ == "__main__":
    main()
