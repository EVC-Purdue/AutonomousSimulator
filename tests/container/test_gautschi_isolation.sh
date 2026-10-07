#!/usr/bin/env bash
# On a Gautschi node: workers launched as runner/gautschi_launch.py does must not see each other's ROS graph, /tmp, /dev/shm or ports
# usage: test_gautschi_isolation.sh <sif> <stack dir>
set -uo pipefail
SIF=$1; STACK=$2; W=$(mktemp -d /dev/shm/AutoSim-iso.XXXX); trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$W"' EXIT
run() {   # worker index, command: the same flags as gautschi_launch.worker_cmd
  mkdir -p "$W/$1"
  apptainer exec --contain --cleanenv --ipc --pid --net --network none --workdir "$W/$1" --env ROS_DOMAIN_ID=$(( ${1%%[a-z]*} + 1 )) \
    --env ROS_HOME=/tmp/ros --env TMPDIR=/tmp --bind "$STACK:/ws:ro" "$SIF" bash -lc \
    "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && $2"
}
PROBE="touch /tmp/probe_A /dev/shm/probe_A; python3 -m http.server 8000 >/dev/null 2>&1 & ros2 topic pub -r 10 /iso_probe std_msgs/msg/String '{data: A}' >/dev/null"
run 0 "$PROBE & sleep 6; ros2 topic list --no-daemon --spin-time 3 | grep -c iso_probe; sleep 20" > "$W/a.out" 2>"$W/a.err" &
sleep 4
LOOK="sleep 4; python3 -m http.server 8000 >/dev/null 2>&1 & sleep 1; ros2 topic list --no-daemon --spin-time 3 | grep -c iso_probe; ls /tmp /dev/shm | grep -c probe_A; python3 -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000').status)\""
run 1 "$LOOK" > "$W/b.out" 2>"$W/b.err" &   # another domain
run 0b "$LOOK" > "$W/c.out" 2>"$W/c.err" &  # same domain id as A: only the namespaces separate them
wait %2 %3
sleep 8
a=$(head -1 "$W/a.out"); b=$(cat "$W/b.out" | tr '\n' ' '); c=$(cat "$W/c.out" | tr '\n' ' ')
echo "A sees its own probe: $a (expect 1)"
echo "B (domain 2) probe topics, probe files, own :8000 http: $b (expect 0 0 200)"
echo "C (domain 1) probe topics, probe files, own :8000 http: $c (expect 0 0 200)"
[ "$a" = 1 ] && [ "$b" = "0 0 200 " ] && [ "$c" = "0 0 200 " ] && echo ISOLATION PASS || { echo ISOLATION FAIL; tail -5 "$W"/*.err; exit 1; }
