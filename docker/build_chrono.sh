#!/usr/bin/env bash
# Builds Chrono 10.0.0 with the Vehicle and Python modules for the system Python 3.10
# CH_USE_SIMD=OFF: SIMD means -march=native, which crashes (SIGILL) on Modal hosts older than the image builder
set -euo pipefail
git clone --depth 1 --branch 10.0.0 https://github.com/projectchrono/chrono.git /tmp/chrono
cmake -S /tmp/chrono -B /tmp/chrono-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_DEMOS=OFF -DBUILD_TESTING=OFF -DBUILD_BENCHMARKING=OFF \
  -DCH_USE_SIMD=OFF \
  -DCH_ENABLE_MODULE_VEHICLE=ON -DCH_ENABLE_MODULE_VEHICLE_COSIM=OFF -DCH_ENABLE_MODULE_PYTHON=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3.10 -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCH_INSTALL_PYTHON_PACKAGE=/usr/local/lib/python3.10/dist-packages
cmake --build /tmp/chrono-build -j3 || cmake --build /tmp/chrono-build -j1
cmake --install /tmp/chrono-build
# Chrono 10 installs the Python package under share/chrono/python and ignores CH_INSTALL_PYTHON_PACKAGE
echo /usr/local/share/chrono/python > /usr/local/lib/python3.10/dist-packages/pychrono.pth && ldconfig
python3 -c "import pychrono, pychrono.vehicle; print('pychrono ok', pychrono.__file__)"
rm -rf /tmp/chrono /tmp/chrono-build
