"""Topic-level interface for stacks whose drivers the sim stands in for (A27): sensor models to ROS messages and back."""
import math

import numpy as np

from harness.geo import to_local
from simnode.opponent import KART_LEN
from simnode.sensors.imu import ACCEL_LSB, ACCEL_SIGMA, GYRO_LSB, GYRO_SIGMA, G

FIX_QUALITY = {0: 0, 1: 1, 2: 2, 4: 4, 5: 5}  # GGA quality to GpsFix.fix_quality
OPPONENT_SIGMA_M = 0.03  # OpponentKart.perception_payload noise


def _deg(dm, hemi, width):
    v = int(dm[:width]) + float(dm[width:]) / 60.0
    return -v if hemi in ("S", "W") else v


def _utc_s(utc):
    return int(utc[0:2]) * 3600 + int(utc[2:4]) * 60 + float(utc[4:])


def gps_fixes(lines, mount):
    """One fix per GGA epoch in the UM982 lines: antenna x, y in the track frame, kart heading ENU when GNHDT is present."""
    out = []
    for line in lines:
        if not line.startswith("$"):
            continue
        f = line[1:line.index("*")].split(",")
        if f[0] == "GNGGA":
            x, y = to_local(_deg(f[2], f[3], 2), _deg(f[4], f[5], 3))
            out.append({"t": _utc_s(f[1]), "fix_quality": FIX_QUALITY.get(int(f[6]), 0), "x": x, "y": y,
                        "sigma_m": 1.0, "speed": 0.0, "heading": 0.0, "heading_valid": False})
        elif not out:
            continue
        elif f[0] == "GNGST":
            out[-1]["sigma_m"] = float(f[2])
        elif f[0] == "GNVTG":
            out[-1]["speed"] = float(f[7]) / 3.6
        elif f[0] == "GNHDT":
            out[-1]["heading"] = math.remainder(math.radians(90.0 + mount["baseline_az_deg"] - float(f[1])), 2 * math.pi)
            out[-1]["heading_valid"] = True
    return out


def imu_body(words, r_mount, gyro_bias):
    """Raw MPU-6050 words to body-frame specific force (m/s^2) and rate (rad/s), gyro bias removed as a calibrated driver would."""
    r = np.asarray(r_mount, float)
    acc = r @ (np.asarray(words[0:3], float) / ACCEL_LSB * G)
    gyr = r @ (np.radians(np.asarray(words[4:7], float) / GYRO_LSB) - np.asarray(gyro_bias, float))
    return acc, gyr


def column_for_steering(steering_rad, wheel_per_column):
    """Column target (deg, + right) for a road-wheel steering angle (rad, + left)."""
    return -math.degrees(steering_rad) / wheel_per_column


def plan_data(msg):
    """kart_interfaces/Plan as pathfinder/plan data in the layout episode.record_plan reads: planner 2 (MPC) with costs, else 1 (pure pursuit)."""
    n, m = len(msg.points), len(msg.cost_terms)
    data = [2.0 if m else 1.0, float(n), *(v for p in msg.points for v in (p.x, p.y))]
    if m:
        data += [float(m), msg.best_cost, *msg.step_costs, float(len(msg.cost_field)), msg.cost_field_start, msg.cost_field_step, *msg.cost_field]
    return data


class TopicIO:
    """Publishers and subscriptions of the topic-level interface, on the sim's nodes."""

    def __init__(self, pub_node, sub_node, on_cmd, on_brake, on_plan):
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from kart_interfaces.msg import DriveCommand, GpsFix, KartTelemetry, ObjectMeasurement, ObjectMeasurementArray, Plan
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Bool, String
        self.GpsFix, self.Imu, self.KartTelemetry, self.Pose = GpsFix, Imu, KartTelemetry, PoseWithCovarianceStamped
        self.ObjectMeasurement, self.ObjectMeasurementArray, self.Bool, self.String = ObjectMeasurement, ObjectMeasurementArray, Bool, String
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        p = pub_node.create_publisher
        self.gps_pub = p(GpsFix, "gps", 10)
        self.imu_pubs = {"front": p(Imu, "imu/front", 10), "back": p(Imu, "imu/back", 10)}
        self.telemetry_pub = p(KartTelemetry, "e_comms/telemetry", 10)
        self.brake_state_pub = p(Bool, "mechanical_brake/state", 10)
        self.pose_pubs = {"slam": p(PoseWithCovarianceStamped, "slam/pose", 10), "gs3d": p(PoseWithCovarianceStamped, "gs3d/pose", 10)}
        self.obs_pub = p(ObjectMeasurementArray, "perception/measurements", 10)
        sub_node.create_subscription(DriveCommand, "cmd_drive", on_cmd, qos_profile_sensor_data)
        sub_node.create_subscription(Bool, "mechanical_brake/command", on_brake, qos_profile_sensor_data)
        sub_node.create_subscription(Plan, "pathfinder/plan", on_plan, qos_profile_sensor_data)

    @staticmethod
    def _stamp(msg, t, frame):
        ns = int(round(t * 1e9))
        msg.header.stamp.sec, msg.header.stamp.nanosec = ns // 1_000_000_000, ns % 1_000_000_000
        msg.header.frame_id = frame
        return msg

    def gps(self, fix):
        m = self._stamp(self.GpsFix(), fix["t"], "odom")
        m.fix_quality, m.x, m.y = fix["fix_quality"], float(fix["x"]), float(fix["y"])
        m.heading, m.heading_valid, m.speed = float(fix["heading"]), fix["heading_valid"], float(fix["speed"])
        s2 = fix["sigma_m"] ** 2
        m.position_covariance = [s2, 0.0, 0.0, s2]
        m.heading_variance = math.radians(0.1) ** 2 if fix["heading_valid"] else 0.0
        m.speed_variance = 0.03 ** 2
        self.gps_pub.publish(m)

    def imu(self, where, t, acc, gyr):
        m = self._stamp(self.Imu(), t, "imu_" + where)
        m.orientation_covariance[0] = -1.0
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (float(v) for v in acc)
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(v) for v in gyr)
        m.linear_acceleration_covariance = [ACCEL_SIGMA ** 2, 0.0, 0.0, 0.0, ACCEL_SIGMA ** 2, 0.0, 0.0, 0.0, ACCEL_SIGMA ** 2]
        m.angular_velocity_covariance = [GYRO_SIGMA ** 2, 0.0, 0.0, 0.0, GYRO_SIGMA ** 2, 0.0, 0.0, 0.0, GYRO_SIGMA ** 2]
        self.imu_pubs[where].publish(m)

    def telemetry(self, t, wheel_speed, steering_rad, steering_current, autonomous, brake_on):
        m = self._stamp(self.KartTelemetry(), t, "base_link")
        m.wheel_speed, m.steering_angle, m.steering_motor_current = float(wheel_speed), float(steering_rad), float(steering_current)
        m.rc_mode = self.KartTelemetry.RC_MODE_AUTONOMOUS if autonomous else self.KartTelemetry.RC_MODE_OVERRIDE
        self.telemetry_pub.publish(m)
        self.brake_state_pub.publish(self.Bool(data=bool(brake_on)))

    def pose(self, which, d):
        m = self._stamp(self.Pose(), d["t"], "odom")
        m.pose.pose.position.x, m.pose.pose.position.y = float(d["x"]), float(d["y"])
        h = 0.5 * float(d["yaw"])
        m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(h), math.sin(h)
        cov = [0.0] * 36
        cov[0] = cov[7] = float(d["pos_sigma_m"]) ** 2
        cov[35] = float(d["yaw_sigma_rad"]) ** 2
        m.pose.covariance = cov
        self.pose_pubs[which].publish(m)

    def opponent(self, t, payload):
        """OpponentKart.perception_payload (body-frame rel x, y first) as one kart measurement."""
        m = self._stamp(self.ObjectMeasurementArray(), t, "base_link")
        if payload:
            o = self.ObjectMeasurement(object_class=self.ObjectMeasurement.CLASS_KART, footprint_size=KART_LEN)
            o.position.x, o.position.y = float(payload[0]), float(payload[1])
            o.position_covariance = [OPPONENT_SIGMA_M ** 2, 0.0, 0.0, OPPONENT_SIGMA_M ** 2]
            m.objects = [o]
        self.obs_pub.publish(m)
