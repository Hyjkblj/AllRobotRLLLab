#!/usr/bin/env bash
set -Eeuo pipefail

# Stop services started by start_local_stack.sh. Conda must already be
# initialized; no environment activation or installation is performed here.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

API_ENV="${ROBOTLAB_API_ENV:-unitree_g1_train}"
RUNTIME_ROOT="${ROBOTLAB_RUNTIME_DIR:-${PROJECT_ROOT}/runtime}"
PROCESS_ROOT="${RUNTIME_ROOT}/processes"

usage() {
  cat <<'EOF'
Usage: scripts/stop_local_stack.sh [--api-env NAME] [--runtime-dir PATH]

Stops API, Local File worker, MuJoCo and React processes recorded by the
local stack. Runtime data is never deleted.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-env)
      [[ $# -ge 2 ]] || { echo "--api-env requires a value" >&2; exit 2; }
      API_ENV="$2"; shift 2 ;;
    --runtime-dir)
      [[ $# -ge 2 ]] || { echo "--runtime-dir requires a value" >&2; exit 2; }
      RUNTIME_ROOT="$2"; PROCESS_ROOT="${RUNTIME_ROOT}/processes"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Required command not found: conda" >&2
  echo "Initialize Conda in the shell, then rerun this script." >&2
  exit 1
fi

stop_pid_group() {
  local pid="$1"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.2
  done
  kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}

pid_matches_service() {
  local pid="$1"
  local marker="$2"
  # Linux exposes the command line needed to avoid killing an unrelated
  # process after a stale PID file has been reused. On other Unix hosts the
  # PID file is still scoped to this runtime directory, so allow the stop.
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    local command_line
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    [[ "${command_line}" == *"${marker}"* ]]
    return $?
  fi
  return 0
}

stopped=0
for name in mujoco frontend; do
  pid_path="${PROCESS_ROOT}/${name}.pid"
  if [[ -f "${pid_path}" ]]; then
    pid="$(<"${pid_path}")"
    marker="${name}"
    [[ "${name}" == "mujoco" ]] && marker="mujoco_service.py"
    [[ "${name}" == "frontend" ]] && marker="npm"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null && pid_matches_service "${pid}" "${marker}"; then
      stop_pid_group "${pid}"
      stopped=$((stopped + 1))
    elif [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "Skipping PID ${pid} in ${pid_path}; command does not match ${marker}." >&2
    fi
    rm -f "${pid_path}"
  fi
done

if conda run --no-capture-output -n "${API_ENV}" python -m tools.robotlab stop \
  --mode local_file --runtime-dir "${RUNTIME_ROOT}"; then
  echo "Local File API and worker stop requested."
else
  echo "Could not invoke robotlab stop in Conda environment ${API_ENV}." >&2
  echo "Inspect ${PROCESS_ROOT}/api.json and stop the recorded processes manually." >&2
fi

echo "Stopped ${stopped} renderer/frontend process group(s); runtime data preserved at ${RUNTIME_ROOT}."
