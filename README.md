# AutonomousSimulator
A27+ Multi-Config Sim

## Deploy
A push to `main` builds `docker/Dockerfile.web` (Flask under gunicorn plus the ideas and Gautschi runners, no ROS or Chrono) into `ghcr.io/evc-purdue/autonomoussim-web:<sha>`, then the self-hosted runner labelled `autonomoussim` runs `deploy.sh`. The app listens on `127.0.0.1:5060`, and every job from the web form runs on ideas8 over Tailscale.

The mini PC needs:
- Docker, Tailscale, and a GitHub Actions runner with the label `autonomoussim`
- `~/AutonomousKart`: a mirror clone (`git clone --mirror git@github.com:EVC-Purdue/AutonomousKart.git ~/AutonomousKart`), so a fetch moves its branches; the form builds the stack from a branch or commit of it
- `~/.autonomoussim-secrets/env`: `AUTONOMOUS_SIM_PASSWORD=...` (required to submit or delete jobs), optional `HF_TOKEN=...` (without it results stay local), optional `IDEAS_HOST=...` (default `csuser@100.88.24.43`)
- `~/.autonomoussim-secrets/ssh`, mounted as the container's `~/.ssh`: a key for ideas8 and GitHub under a default name such as `id_ed25519`, `gautschi` for Gautschi, and a `known_hosts` with all three hosts (`ssh-keyscan 100.88.24.43 gautschi.rcac.purdue.edu github.com`)
- `~/autonomoussim/results`, created by `deploy.sh`

Environment: `AUTONOMOUS_SIM_RESULTS` (results dir, default `results/`), `AUTONOMOUS_KART_REPO` (AutonomousKart clone), `AUTONOMOUS_SIM_HF_UPLOAD=0` (skip the Hugging Face upload), `AUTONOMOUS_SIM_PASSWORD` (unset: no password).
