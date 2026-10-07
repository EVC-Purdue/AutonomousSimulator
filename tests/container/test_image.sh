#!/usr/bin/env bash
set -euo pipefail
docker run --rm evc-sim:dev bash -lc 'source /opt/ros/humble/setup.bash && python3 -c "import rclpy, pychrono, pychrono.vehicle; print(\"imports ok\")" && python3 simnode/tools/static_loads.py chrono'
