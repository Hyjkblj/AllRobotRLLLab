#!/usr/bin/env bash
set -Eeuo pipefail

# Start the API, durable Local File worker, MuJoCo renderer and React UI as
# one foreground process. Conda itself must already be initialized by the
# caller; this script only uses `conda run` to select project environments.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

API_ENV="${ROBOTLAB_API_ENV:-unitree_g1_train}"
MUJOCO_ENV="${ROBOTLAB_MUJOCO_ENV:-allrobotrl-mujoco}"
NODE_ENV="${ROBOTLAB_NODE_ENV:-${MUJOCO_ENV}}"
NODE_ENV_EXPLICIT=0
if [[ -n "${ROBOTLAB_NODE_ENV:-}" ]]; then NODE_ENV_EXPLICIT=1; fi
MODE="${ROBOTLAB_MODE:-local_file}"
RUNTIME_ROOT="${ROBOTLAB_RUNTIME_DIR:-${PROJECT_ROOT}/runtime}"
API_HOST="${ROBOTLAB_API_HOST:-127.0.0.1}"
MUJOCO_HOST="${ROBOTLAB_MUJOCO_HOST:-127.0.0.1}"
FRONTEND_HOST="${ROBOTLAB_FRONTEND_HOST:-127.0.0.1}"
API_PORT="${ROBOTLAB_API_PORT:-8010}"
MUJOCO_PORT="${ROBOTLAB_MUJOCO_PORT:-8787}"
FRONTEND_PORT="${ROBOTLAB_FRONTEND_PORT:-4173}"
START_TIMEOUT="${ROBOTLAB_START_TIMEOUT:-60}"
FRONTEND_ROOT="${PROJECT_ROOT}/frontend-prototype/react-app"
PROCESS_ROOT="${RUNTIME_ROOT}/processes"

API_URL="http://127.0.0.1:${API_PORT}"
MUJOCO_URL="http://127.0.0.1:${MUJOCO_PORT}"
FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"

CHILD_PIDS=()
CHILD_NAMES=()
PLATFORM_PIDS=()
PLATFORM_NAMES=()
CLEANED_UP=0
API_STARTED=0

usage() {
  cat <<'EOF'
Usage: scripts/start_local_stack.sh [options]

Start the local no-database stack in one terminal. Conda initialization and
dependency installation are intentionally outside this script.

Options:
  --api-env NAME       API and worker Conda environment (default: unitree_g1_train)
  --mujoco-env NAME    MuJoCo Conda environment (default: allrobotrl-mujoco)
  --node-env NAME      React/Node Conda environment (default: mujoco env)
  --runtime-dir PATH   Local File runtime directory (default: ./runtime)
  --api-port PORT      FastAPI port (default: 8010)
  --mujoco-port PORT   MuJoCo service port (default: 8787)
  --frontend-port PORT Vite port (default: 4173)
  --help               Show this message

The same values can be supplied with ROBOTLAB_* environment variables.
Press Ctrl-C to stop all services and preserve runtime data.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-env)
      [[ $# -ge 2 ]] || { echo "--api-env requires a value" >&2; exit 2; }
      API_ENV="$2"; shift 2 ;;
    --mujoco-env)
      [[ $# -ge 2 ]] || { echo "--mujoco-env requires a value" >&2; exit 2; }
      MUJOCO_ENV="$2"
      if [[ "${NODE_ENV_EXPLICIT}" -eq 0 ]]; then NODE_ENV="$2"; fi
      shift 2 ;;
    --node-env)
      [[ $# -ge 2 ]] || { echo "--node-env requires a value" >&2; exit 2; }
      NODE_ENV="$2"; NODE_ENV_EXPLICIT=1; shift 2 ;;
    --runtime-dir)
      [[ $# -ge 2 ]] || { echo "--runtime-dir requires a value" >&2; exit 2; }
      RUNTIME_ROOT="$2"; PROCESS_ROOT="${RUNTIME_ROOT}/processes"; shift 2 ;;
    --api-port)
      [[ $# -ge 2 ]] || { echo "--api-port requires a value" >&2; exit 2; }
      API_PORT="$2"; API_URL="http://127.0.0.1:${API_PORT}"; shift 2 ;;
    --mujoco-port)
      [[ $# -ge 2 ]] || { echo "--mujoco-port requires a value" >&2; exit 2; }
      MUJOCO_PORT="$2"; MUJOCO_URL="http://127.0.0.1:${MUJOCO_PORT}"; shift 2 ;;
    --frontend-port)
      [[ $# -ge 2 ]] || { echo "--frontend-port requires a value" >&2; exit 2; }
      FRONTEND_PORT="$2"; FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "${RUNTIME_ROOT}" == "~" ]]; then
  RUNTIME_ROOT="${HOME}"
elif [[ "${RUNTIME_ROOT}" == ~/* ]]; then
  RUNTIME_ROOT="${HOME}/${RUNTIME_ROOT#~/}"
fi
PROCESS_ROOT="${RUNTIME_ROOT}/processes"

for port in "${API_PORT}" "${MUJOCO_PORT}" "${FRONTEND_PORT}"; do
  if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "Invalid TCP port: ${port}" >&2
    exit 2
  fi
done
if [[ ! "${START_TIMEOUT}" =~ ^[0-9]+$ ]] || (( START_TIMEOUT < 1 )); then
  echo "ROBOTLAB_START_TIMEOUT must be a positive integer" >&2
  exit 2
fi

if [[ "${MODE}" != "local_file" ]]; then
  echo "This wrapper supports only ROBOTLAB_MODE=local_file (got ${MODE})" >&2
  exit 2
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    return 1
  fi
}

require_command conda
require_command curl
require_command setsid

if [[ ! -d "${FRONTEND_ROOT}" || ! -f "${FRONTEND_ROOT}/package.json" ]]; then
  echo "React app directory is missing: ${FRONTEND_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${FRONTEND_ROOT}/node_modules/.bin/vite" ]]; then
  echo "React dependencies are not installed in ${FRONTEND_ROOT}." >&2
  echo "Run: conda run -n ${NODE_ENV} npm ci --prefix ${FRONTEND_ROOT}" >&2
  exit 1
fi

check_environment() {
  local environment="$1"
  if ! conda run --no-capture-output -n "${environment}" python --version >/dev/null 2>&1; then
    echo "Conda environment is unavailable: ${environment}" >&2
    return 1
  fi
}

check_environment "${API_ENV}"
check_environment "${MUJOCO_ENV}"
if ! conda run --no-capture-output -n "${NODE_ENV}" node --version >/dev/null 2>&1; then
  echo "Node.js is unavailable in Conda environment: ${NODE_ENV}" >&2
  echo "Install it outside this wrapper with: conda install -n ${NODE_ENV} -c conda-forge nodejs=20 -y" >&2
  exit 1
fi
if ! conda run --no-capture-output -n "${NODE_ENV}" npm --version >/dev/null 2>&1; then
  echo "npm is unavailable in Conda environment: ${NODE_ENV}" >&2
  exit 1
fi

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

for port in "${API_PORT}" "${MUJOCO_PORT}" "${FRONTEND_PORT}"; do
  if port_in_use "${port}"; then
    echo "Port ${port} is already in use; stop the existing service before starting the stack." >&2
    exit 1
  fi
done

mkdir -p "${PROCESS_ROOT}"
export ROBOTLAB_MODE="local_file"
export ROBOTLAB_RUNTIME_DIR="${RUNTIME_ROOT}"
export MOTIONLAB_MUJOCO_HOST="${MUJOCO_HOST}"
export MOTIONLAB_MUJOCO_PORT="${MUJOCO_PORT}"
export VITE_PLATFORM_API_TARGET="http://${API_HOST}:${API_PORT}"
export VITE_MUJOCO_API_TARGET="http://${MUJOCO_HOST}:${MUJOCO_PORT}"

stop_pid_group() {
  local pid="$1"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  if [[ "${CLEANED_UP}" -eq 1 ]]; then
    exit "${exit_code}"
  fi
  CLEANED_UP=1
  trap - EXIT INT TERM
  echo
  echo "Stopping AllRobotRLLLab local stack..."
  for pid in "${CHILD_PIDS[@]}"; do
    stop_pid_group "${pid}"
  done
  rm -f "${PROCESS_ROOT}/mujoco.pid" "${PROCESS_ROOT}/frontend.pid"
  if [[ "${API_STARTED}" -eq 1 ]]; then
    conda run --no-capture-output -n "${API_ENV}" python -m tools.robotlab stop \
      --mode local_file --runtime-dir "${RUNTIME_ROOT}" >/dev/null 2>&1 || true
  fi
  echo "Runtime data preserved at ${RUNTIME_ROOT}"
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

start_child() {
  local name="$1"
  local environment="$2"
  shift 2
  local log_path="${PROCESS_ROOT}/${name}.log"
  local pid_path="${PROCESS_ROOT}/${name}.pid"
  : >>"${log_path}"
  setsid conda run --no-capture-output -n "${environment}" "$@" >>"${log_path}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" >"${pid_path}"
  CHILD_PIDS+=("${pid}")
  CHILD_NAMES+=("${name}")
  echo "Started ${name} (pid ${pid}); log: ${log_path}"
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local deadline=$((SECONDS + START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited before health check; inspect ${PROCESS_ROOT}/${name}.log" >&2
      return 1
    fi
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "${name} is healthy: ${url}"
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for ${name}: ${url}; inspect ${PROCESS_ROOT}/${name}.log" >&2
  return 1
}

echo "Starting AllRobotRLLLab local stack from ${PROJECT_ROOT}"
echo "API=${API_ENV}, MuJoCo=${MUJOCO_ENV}, Node=${NODE_ENV}"

if ! conda run --no-capture-output -n "${API_ENV}" python -m tools.robotlab start \
  --mode local_file --runtime-dir "${RUNTIME_ROOT}" --host "${API_HOST}" --port "${API_PORT}"; then
  echo "Failed to start FastAPI and Local File worker; inspect ${PROCESS_ROOT}/api.log" >&2
  exit 1
fi
API_STARTED=1
for name in api worker; do
  pid_path="${PROCESS_ROOT}/${name}.json"
  if [[ ! -f "${pid_path}" ]]; then
    echo "Local File ${name} PID record is missing: ${pid_path}" >&2
    exit 1
  fi
  pid="$(sed -n 's/.*"pid"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${pid_path}" | head -n 1)"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "Local File ${name} stopped during startup; inspect ${PROCESS_ROOT}/${name}.log" >&2
    exit 1
  fi
  PLATFORM_PIDS+=("${pid}")
  PLATFORM_NAMES+=("${name}")
done

start_child mujoco "${MUJOCO_ENV}" python -u frontend-prototype/mujoco_service.py
MUJOCO_PID="${CHILD_PIDS[0]}"
start_child frontend "${NODE_ENV}" npm --prefix "${FRONTEND_ROOT}" run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}"
FRONTEND_PID="${CHILD_PIDS[1]}"

wait_for_http "MuJoCo" "${MUJOCO_URL}/api/mujoco/health" "${MUJOCO_PID}"
wait_for_http "React" "${FRONTEND_URL}" "${FRONTEND_PID}"
if ! curl -fsS --max-time 3 "${API_URL}/api/v1/health" >/dev/null 2>&1; then
  echo "FastAPI health check failed after startup; inspect ${PROCESS_ROOT}/api.log" >&2
  exit 1
fi

cat <<EOF

AllRobotRLLLab local stack is running.
  React:   ${FRONTEND_URL}
  API:     ${API_URL}/api/v1/health
  MuJoCo:  ${MUJOCO_URL}/api/mujoco/health
  Logs:    ${PROCESS_ROOT}

Press Ctrl-C to stop API, worker, MuJoCo and React. Runtime data is preserved.
EOF

while :; do
  for index in "${!PLATFORM_PIDS[@]}"; do
    pid="${PLATFORM_PIDS[${index}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${PLATFORM_NAMES[${index}]} stopped unexpectedly; inspect ${PROCESS_ROOT}/${PLATFORM_NAMES[${index}]}.log" >&2
      exit 1
    fi
  done
  for index in "${!CHILD_PIDS[@]}"; do
    pid="${CHILD_PIDS[${index}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${CHILD_NAMES[${index}]} stopped unexpectedly; inspect ${PROCESS_ROOT}/${CHILD_NAMES[${index}]}.log" >&2
      exit 1
    fi
  done
  if ! curl -fsS --max-time 2 "${API_URL}/api/v1/health" >/dev/null 2>&1; then
    echo "FastAPI health check failed; stopping the local stack." >&2
    exit 1
  fi
  sleep 2
done
