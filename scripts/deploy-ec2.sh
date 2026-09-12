#!/usr/bin/env bash
# Used by .github/workflows/deploy.yml on the EC2 host.
set -euo pipefail

cd ~/data-platform

git pull origin main
~/.local/bin/uv sync --frozen

# Restarting the daemon sends SIGTERM to in-flight ops (e.g. multi-hour screener_job).
# If Dagster reports active runs, sync deps only; restart daemon manually when idle.
export DAGSTER_HOME="${DAGSTER_HOME:-$HOME/.dagster}"
ACTIVE=0
if command -v ~/.local/bin/uv >/dev/null; then
  ACTIVE=$(~/.local/bin/uv run dagster run list --status STARTED --limit 20 2>/dev/null | grep -c STARTED || true)
fi

if [[ "${ACTIVE}" -gt 0 ]]; then
  echo "Dagster has ${ACTIVE} STARTED run(s); skipping daemon restart (code pulled, uv sync done)."
  echo "When idle: sudo systemctl restart dagster-daemon dagster-webserver"
  sudo systemctl restart dagster-webserver || true
  exit 0
fi

sudo systemctl restart dagster-daemon
sudo systemctl restart dagster-webserver
