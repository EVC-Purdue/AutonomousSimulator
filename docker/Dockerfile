FROM evc-sim-base:1
COPY harness /opt/sim/harness
COPY simnode /opt/sim/simnode
COPY chrono /opt/sim/chrono
COPY configs /opt/sim/configs
COPY tracks /opt/sim/tracks
COPY data/noise /opt/sim/data/noise
COPY data/clips /opt/sim/data/clips
COPY docker/Dockerfile docker/Dockerfile.base docker/build_chrono.sh docker/requirements-stack.txt /opt/sim/docker/
ENV PYTHONPATH=/opt/sim:/usr/local/lib/python3.10/dist-packages
WORKDIR /opt/sim
RUN python3 -c "from harness.config import engine_hash; print(engine_hash())" > /opt/sim/ENGINE && find /opt/sim -name __pycache__ -prune -exec rm -rf {} +
