"""Lockstep round trip with no physics and no stack: /clock -> a 60 Hz sim-time timer in another process -> heartbeat back.

python3 /prof/profile_rtt.py [--idle N] [--clock all|deadline] [--ticks K] [--busy-us U]
Measures the DDS plus executor floor of one tick and the CPU each idle use_sim_time node burns per /clock message.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

GRID_NS = 5_000_000
STEP_NS = 500_000


def child(role):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import Header
    rclpy.init()
    node = Node(f"{role}_{os.getpid()}", parameter_overrides=[Parameter("use_sim_time", value=True)])
    if role == "pf":
        pub = node.create_publisher(Header, "rtt/tick", 50)

        def beat():
            m = Header()
            m.stamp = node.get_clock().now().to_msg()
            pub.publish(m)
        node.create_timer(1.0 / 60.0, beat)
    print("ready", flush=True)
    rclpy.spin(node)


def cpu_of(pids):
    tck = os.sysconf("SC_CLK_TCK")
    out = 0.0
    for p in pids:
        with open(f"/proc/{p}/stat") as f:
            st = f.read().rsplit(")", 1)[1].split()
        out += (int(st[11]) + int(st[12])) / tck
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=int, default=0)
    ap.add_argument("--clock", default="all", choices=("all", "deadline"))
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--busy-us", type=float, default=0.0)  # stand-in physics cost per 0.5 ms step
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        return child(a.child)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    kids = [subprocess.Popen([sys.executable, __file__, "--child", "pf"], stdout=subprocess.PIPE, text=True)]
    kids += [subprocess.Popen([sys.executable, __file__, "--child", "idle"], stdout=subprocess.PIPE, text=True)
             for _ in range(a.idle)]
    for k in kids:
        k.stdout.readline()
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Header
    rclpy.init()
    node = Node("rtt_sim", parameter_overrides=[Parameter("use_sim_time", value=False)])
    pub = node.create_publisher(Clock, "/clock", 10)
    hb = {"ns": -1, "wall": 0.0}

    def on_tick(m):
        hb["ns"], hb["wall"] = m.stamp.sec * 1_000_000_000 + m.stamp.nanosec, time.perf_counter()
    node.create_subscription(Header, "rtt/tick", on_tick, 50)
    exe = SingleThreadedExecutor()
    exe.add_node(node)
    while node.count_subscribers("/clock") < len(kids) or node.count_publishers("rtt/tick") < 1:
        exe.spin_once(timeout_sec=0.01)
    time.sleep(1.0)
    step, last, rtt, tick_wall, n_pub = 0, -1, [], [], 0
    w_all = time.perf_counter()
    c0 = cpu_of([k.pid for k in kids[1:]]) if a.idle else 0.0
    for k in range(1, a.ticks + 1):
        end = math.ceil(k * 1_000_000_000 / 60 / STEP_NS)
        dl = k * int(1e9 / 60)
        t0 = time.perf_counter()
        while step < end:
            if a.busy_us:
                b = time.perf_counter() + a.busy_us * 1e-6
                while time.perf_counter() < b:
                    pass
            step += 1
            ns = step * STEP_NS
            if step % 2 == 0 or step == end:
                if a.clock == "all" or ns // GRID_NS > last // GRID_NS or step == end:
                    m = Clock()
                    m.clock.sec, m.clock.nanosec = ns // 1_000_000_000, ns % 1_000_000_000
                    pub.publish(m)
                    n_pub += 1
                    last = ns
                    t_last = time.perf_counter()
        w = time.monotonic() + 2.0
        while hb["ns"] < dl:
            exe.spin_once(timeout_sec=0.0)
            if hb["ns"] >= dl:
                break
            if time.monotonic() > w:
                raise SystemExit(f"timeout at tick {k}")
            time.sleep(0.0005)
        rtt.append(hb["wall"] - t_last)
        tick_wall.append(time.perf_counter() - t0)
    wall = time.perf_counter() - w_all
    idle_cpu = (cpu_of([k.pid for k in kids[1:]]) - c0) if a.idle else 0.0
    sim_s = a.ticks / 60
    rtt_ms = sorted(x * 1e3 for x in rtt[10:])
    print(json.dumps({"idle": a.idle, "clock": a.clock, "busy_us": a.busy_us, "wall_per_sim_s": wall / sim_s,
                      "clock_per_sim_s": n_pub / sim_s, "rtt_ms_p50": rtt_ms[len(rtt_ms) // 2],
                      "rtt_ms_p90": rtt_ms[int(len(rtt_ms) * 0.9)], "rtt_ms_mean": sum(rtt_ms) / len(rtt_ms),
                      "idle_node_cpu_per_sim_s": idle_cpu / sim_s / max(a.idle, 1),
                      "idle_node_cpu_us_per_clock": 1e6 * idle_cpu / max(a.idle, 1) / n_pub if a.idle else 0.0}))
    for k in kids:
        k.kill()


if __name__ == "__main__":
    main()
