#!/usr/bin/env bash
# Builds the AutonomousKart-sim worktree in evc-sim:dev and checks the stack boots at sim time against emulated devices
set -euo pipefail
WT=${WT:-$HOME/PycharmProjects/AutonomousKart-sim}
docker volume create evc-stack-boot >/dev/null
t0=$SECONDS
docker run --rm -v evc-stack-boot:/ws -v "$WT/src":/ws/src:ro evc-sim:dev bash -lc \
  'source /opt/ros/humble/setup.bash && cd /ws && colcon build --packages-select autonomous_kart'
echo "build wall time: $((SECONDS - t0)) s"
docker run --rm -i --cpus 2 -m 3g -v evc-stack-boot:/ws evc-sim:dev bash -s <<'SH'
{
source /opt/ros/humble/setup.bash && source /ws/install/setup.bash
mkdir -p /tmp/ep
cat > /tmp/ep/episode.yaml <<'EOF'
e_comms_node:
  ros__parameters:
    can_channel: /tmp/can0
gps_node:
  ros__parameters:
    serial_device: socket://127.0.0.1:5017
    rtcm_enabled: false
    use_dual_heading: false
imu_node:
  ros__parameters:
    imu_driver: sim
    calibration_cache: /tmp/ep/imu_calibration.json
EOF
socat pty,raw,echo=0,link=/tmp/can0 pty,raw,echo=0,link=/tmp/can1 2>/dev/null &
while [ ! -e /tmp/can1 ]; do sleep 0.1; done
cat /tmp/can1 > /tmp/can1.log &
# /clock in 1 ms sim steps at 10x real time, plus a TCP NMEA source sending GGA and VTG every 50 ms sim at the start of center
cd /opt/sim && python3 - > /tmp/devices.log 2>&1 <<'PY' &
import math, socket, threading, time
import rclpy
from rosgraph_msgs.msg import Clock
from harness import geo, lines
L = lines.load_line("center")
lat, lon = geo.to_latlon(float(L.x[0]), float(L.y[0]))
course = (90.0 - math.degrees(float(L.psi[0]))) % 360.0
def nmea(body):
    cs = 0
    for c in body:
        cs ^= ord(c)
    return f"${body}*{cs:02X}\r\n".encode()
def dm(v, w):
    a = abs(v)
    return f"{int(a) * 100 + (a - int(a)) * 60:0{w}.6f}"
clients = []
def serve():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 5017))
    srv.listen()
    while True:
        c, _ = srv.accept()
        clients.append(c)
        threading.Thread(target=lambda c=c: [None for _ in iter(lambda: c.recv(4096), b"")], daemon=True).start()
threading.Thread(target=serve, daemon=True).start()
rclpy.init()
node = rclpy.create_node("boot_clock")
pubs = [node.create_publisher(Clock, "/sim/clock/" + n, 10)  # bringup_simdev remaps each node's /clock
        for n in ("pathfinder_node", "master_api", "localization_node", "e_comms_node", "gps_node", "imu_node")]
step_ns, t_ns, wall0 = 1_000_000, 0, time.monotonic()
while True:
    t_ns += step_ns
    msg = Clock()
    msg.clock.sec, msg.clock.nanosec = divmod(t_ns, 1_000_000_000)
    for pub in pubs:
        pub.publish(msg)
    if t_ns % 50_000_000 == 0:
        s = t_ns / 1e9
        hms = f"{int(s // 3600):02d}{int(s % 3600 // 60):02d}{s % 60:05.2f}"
        out = nmea(f"GPGGA,{hms},{dm(lat, 11)},{'N' if lat >= 0 else 'S'},{dm(lon, 12)},{'E' if lon >= 0 else 'W'},4,20,0.6,180.0,M,-33.0,M,1.0,0000")
        out += nmea(f"GPVTG,{course:.2f},T,,M,0.000,N,0.000,K,D")
        for c in list(clients):
            try:
                c.sendall(out)
            except OSError:
                clients.remove(c)
    lag = wall0 + t_ns / 1e10 - time.monotonic()
    if lag > 0:
        time.sleep(lag)
PY
cd /
ros2 launch autonomous_kart bringup_simdev.launch.py episode_params:=/tmp/ep/episode.yaml > /tmp/launch.log 2>&1 &
start=$SECONDS
state=0; nodes=0; slcan=0; gps=0
while [ $((SECONDS - start)) -lt 15 ]; do
  [ $state = 1 ] || { curl -s -m 1 localhost:8000/get_state | grep -q . && state=1 && echo "get_state at $((SECONDS - start)) s"; } || true
  [ $slcan = 1 ] || { grep -q $'O\r' /tmp/can1.log && slcan=1 && echo "slcan O at $((SECONDS - start)) s"; } || true
  [ $nodes = 1 ] || { n=$(timeout -s KILL 5 ros2 node list 2>/dev/null | grep -cxE "/(pathfinder_node|master_api|localization_node|e_comms_node|gps_node|imu_node)"); [ "$n" = 6 ] && nodes=1 && echo "six nodes at $((SECONDS - start)) s"; } || true
  [ $gps = 1 ] || { timeout -s KILL 5 ros2 topic echo --once gps nav_msgs/msg/Odometry > /tmp/gps.log 2>&1 && gps=1 && echo "gps at $((SECONDS - start)) s"; } || true
  [ $state$nodes$slcan$gps = 1111 ] && break
  sleep 0.2
done
echo "get_state=$state nodes=$nodes slcan=$slcan gps=$gps"
if [ $state$nodes$slcan$gps != 1111 ]; then
  echo "--- node list"; timeout -s KILL 5 ros2 node list || true
  echo "--- devices"; tail -20 /tmp/devices.log
  echo "--- launch"; tail -60 /tmp/launch.log
  echo FAIL; exit 1
fi
head -12 /tmp/gps.log
echo PASS
} </dev/null
SH
