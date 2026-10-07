"""Per-node /clock schedule: a node is clocked only at the instants where the 1 ms clock would have run one of its callbacks."""
import os

NS = 1_000_000_000


def period_ns(sec):
    return int(float(sec) * NS)  # as rclpy's create_timer


class ClockPlan:
    def __init__(self, periods, poll=None):
        """periods {node: [timer period ns]}, all timers anchored at sim time 0; poll (period ns, nodes) is the GPS
        serial poll, which reads only what the sim wrote, so it is clocked only where a write can be read."""
        self.periods, self.poll = periods, poll
        self.prev, self.w_last = 0, None

    def due(self, ns, writing=False):
        """Nodes to clock at instant ns, given the previous instant; writing: the sim writes NMEA right after this clock."""
        out = [n for n, ps in self.periods.items() if any(ns // p > self.prev // p for p in ps)]
        if self.poll is not None:
            p, nodes = self.poll
            d = ns // p * p
            if d > self.prev and (writing and d == ns or self.w_last is not None and self.w_last >= d - p):
                out += [n for n in nodes if n not in out]  # d == w_last + p also covers a lost race at w_last
        self.prev = ns
        return out

    def wrote(self, ns):
        self.w_last = ns


def _param(recipe, share_dir, node, yaml_name, key, default):
    """Recipe override, else the installed params YAML, else default."""
    rp = recipe.get("params") or {}
    for blk in (rp.get(node, {}), rp.get("/**", {})):
        if blk.get(key) is not None:
            return blk[key]
    try:
        import yaml
        with open(os.path.join(share_dir, "params", yaml_name)) as f:
            v = yaml.safe_load(f)[node]["ros__parameters"].get(key)
        return default if v is None else v
    except (OSError, KeyError, TypeError):
        return default


def clock_plan(system_frequency, recipe, share_dir=None):
    """(periods, poll) from the stack's timers; the constant periods are the ones hard-coded in the nodes."""
    if share_dir is None:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("autonomous_kart")

    def q(node, y, key, default):
        return float(_param(recipe, share_dir, node, y, key, default))
    imu = [period_ns(1.0 / q("imu_node", "imu.yaml", "imu_frequency", 100)),
           period_ns(1.0 / q("imu_node", "imu.yaml", "status_frequency", 1.0))]
    gps_status = _param(recipe, share_dir, "gps_node", "gps.yaml", "status_frequency", None) or \
        _param(recipe, share_dir, "gps_node", "gps.yaml", "gps_frequency", 20)
    periods = {"pathfinder_node": [period_ns(1.0 / system_frequency), period_ns(0.5), period_ns(5.0)],
               "imu_node": imu,
               "localization_node": [imu[0], period_ns(1.0)],  # IMU messages arrive at imu_node's timer instants
               "gps_node": [period_ns(1.0 / float(gps_status))],
               "e_comms_node": [period_ns(q("e_comms_node", "e_comms.yaml", "heartbeat_period_ms", 100.0) / 1000.0),
                                period_ns(0.1), period_ns(5.0)],
               "master_api": [period_ns(1.0)]}  # no timers; reads the clock only for log stamps
    poll = (period_ns(1.0 / q("gps_node", "gps.yaml", "serial_poll_hz", 200)), ["gps_node", "localization_node"])
    return periods, poll
