"""The project-owned deployment and operator CLI.

Commands are intentionally wrappers around documented tools.  They report
commands for missing system components instead of silently changing a host.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from backend.app.infrastructure.robot_registry import LocalRobotRegistry, RobotRegistryError


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "infra" / "compose" / "docker-compose.staging.yml"
ENV_EXAMPLE = ROOT / ".env.staging.example"
DEFAULT_RUNTIME_ROOT = ROOT / "runtime"


def _run(command: list[str], *, capture: bool = False) -> int:
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=capture, text=True)
    except FileNotFoundError:
        return 127
    if capture and completed.stdout:
        print(completed.stdout, end="")
    if capture and completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


def _docker_compose() -> list[str] | None:
    docker = shutil.which("docker")
    if docker:
        probe = subprocess.run([docker, "compose", "version"], cwd=ROOT, capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            return [docker, "compose"]
    compose = shutil.which("docker-compose")
    return [compose] if compose else None


def _api_probe(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/v1/health", timeout=3) as response:
            return {"ok": response.status == 200, "status": response.status}
    except (OSError, urllib.error.URLError) as exc:
        return {"ok": False, "error": type(exc).__name__}


def _wait_for_api(url: str, process: subprocess.Popen[bytes] | subprocess.Popen[str], *, timeout_seconds: float = 30.0) -> bool:
    """Wait until a locally spawned API is healthy or its process exits."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if _api_probe(url).get("ok"):
            return True
        time.sleep(0.25)
    return False


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _mode(args: argparse.Namespace, env_values: dict[str, str] | None = None) -> str:
    configured = getattr(args, "mode", None) or os.getenv("ROBOTLAB_MODE")
    if configured:
        return configured.strip().lower()
    if env_values and env_values.get("ROBOTLAB_MODE"):
        return env_values["ROBOTLAB_MODE"].strip().lower()
    return "local_file"


def _runtime_root(args: argparse.Namespace) -> Path:
    value = getattr(args, "runtime_dir", None) or os.getenv("ROBOTLAB_RUNTIME_DIR")
    return Path(value).expanduser().resolve() if value else DEFAULT_RUNTIME_ROOT.resolve()


def _required(check: dict[str, Any]) -> bool:
    return bool(check.get("required", True))


def _doctor(args: argparse.Namespace) -> int:
    env_path = Path(args.env_file).expanduser() if args.env_file else ROOT / ".env.staging"
    env_values = _load_dotenv(env_path)
    def env(name: str) -> str:
        return os.getenv(name, env_values.get(name, ""))

    mode = _mode(args, env_values)
    runtime_root = _runtime_root(args)
    gpu_required = bool(getattr(args, "gpu", False)) or mode == "compose"
    checks: dict[str, dict[str, Any]] = {"mode": {"ok": mode in {"local_file", "compose", "memory"}, "value": mode, "fix": "Use --mode local_file or --mode compose"}}
    checks["runtime"] = {"ok": True, "path": str(runtime_root), "fix": "Choose a writable ROBOTLAB_RUNTIME_DIR"}
    if mode == "local_file":
        try:
            runtime_root.mkdir(parents=True, exist_ok=True)
            probe = runtime_root / ".write-probe"
            probe.write_text("ok", encoding="ascii")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            checks["runtime"] = {"ok": False, "path": str(runtime_root), "error": type(exc).__name__, "fix": "Choose a writable ROBOTLAB_RUNTIME_DIR"}
        checks["docker"] = {"ok": _docker_compose() is not None, "required": False, "fix": "Docker is optional in Local File Mode"}
        checks["nvidia"] = {"ok": shutil.which("nvidia-smi") is not None, "required": gpu_required, "fix": "Install NVIDIA WSL CUDA for GPU workers"}
    else:
        checks["docker"] = {"ok": _docker_compose() is not None, "fix": "Install Docker Engine/Desktop with Compose v2"}
        checks["nvidia"] = {"ok": shutil.which("nvidia-smi") is not None, "required": gpu_required, "fix": "Install NVIDIA driver and NVIDIA Container Toolkit on GPU hosts"}
    conda_env = os.getenv("CONDA_DEFAULT_ENV", "")
    checks["conda"] = {"ok": bool(conda_env), "value": conda_env or None, "required": mode != "memory", "fix": "conda activate the platform or Isaac runtime environment"}
    variables = ("DATABASE_URL", "REDIS_URL", "MINIO_ENDPOINT", "WORKER_AUTH_TOKEN") if mode == "compose" else ()
    for variable in variables:
        value = env(variable)
        if not value and variable == "DATABASE_URL" and env("POSTGRES_PASSWORD"):
            value = "<derived-from-postgres-settings>"
        if not value and variable == "REDIS_URL":
            value = "<derived-from-compose-service>"
        if not value and variable == "MINIO_ENDPOINT":
            value = "<derived-from-compose-service>"
        placeholder = any(marker in value.lower() for marker in ("replace-with", "changeme", "allrobotrl_dev_only"))
        if variable == "DATABASE_URL":
            placeholder = placeholder or any(marker in env("POSTGRES_PASSWORD").lower() for marker in ("replace-with", "changeme", "allrobotrl_dev_only"))
        checks[variable] = {"ok": bool(value) and not placeholder, "configured": bool(value), "placeholder": placeholder, "fix": f"Set {variable} in .env.staging or the server secret manager"}
    runtime_variables = ("ISAACLAB_PATH", "ISAACSIM_PATH", "GMR_PATH", "GVHMR_PATH", "UNITREE_MUJOCO_PATH")
    for variable in runtime_variables:
        value = env(variable)
        if variable == "ISAACSIM_PATH" and not value:
            probe = subprocess.run([sys.executable, "-c", "import importlib.util,pathlib; s=importlib.util.find_spec('isaacsim'); print(pathlib.Path(s.origin).resolve().parent.parent if s and s.origin and pathlib.Path(s.origin).resolve().parent.name == 'isaacsim' else '')"], cwd=ROOT, capture_output=True, text=True, check=False)
            value = probe.stdout.strip()
        checks[variable] = {"ok": bool(value) and Path(value).expanduser().is_dir(), "required": gpu_required, "path": value or None, "fix": f"Set {variable} to the locked external runtime directory"}
    runtime_script = ROOT / "scripts" / "check_external_runtime.py"
    if all(checks[name].get("ok") for name in runtime_variables):
        runtime_env = os.environ.copy()
        runtime_env.update({name: env(name) for name in ("ISAACLAB_PATH", "ISAACSIM_PATH", "GMR_PATH", "GVHMR_PATH", "UNITREE_MUJOCO_PATH", "UNITREE_RL_LAB_PATH")})
        result = subprocess.run([sys.executable, str(runtime_script)], cwd=ROOT, env=runtime_env, capture_output=True, text=True, check=False)
        checks["external_runtime"] = {"ok": result.returncode == 0, "required": gpu_required, "details": (result.stdout + result.stderr).strip(), "fix": "Check the pinned Git SHA values in README.md"}
    else:
        checks["external_runtime"] = {"ok": False, "required": gpu_required, "state": "not_ready", "fix": "Configure all external runtime paths, then rerun robotlab doctor"}
    checks["api"] = _api_probe(args.api_url)
    if args.json:
        print(json.dumps({"status": "ok" if all(item.get("ok") for item in checks.values() if _required(item)) else "degraded", "mode": mode, "runtime_root": str(runtime_root), "checks": checks}, ensure_ascii=False, indent=2))
    else:
        for name, item in checks.items():
            state = "OK" if item.get("ok") else ("OPTIONAL" if not _required(item) else "MISSING")
            print(f"[{state}] {name}: {item.get('value') or item.get('path') or item.get('error') or item.get('fix', '')}")
    return 0 if all(item.get("ok") for item in checks.values() if _required(item)) else 1


def _compose_args(args: argparse.Namespace) -> list[str]:
    compose = _docker_compose()
    if compose is None:
        raise SystemExit("Docker Compose is not available; install it and rerun the printed command")
    command = compose + ["-f", str(COMPOSE)]
    env_file = Path(args.env_file).expanduser() if args.env_file else ROOT / ".env.staging"
    if env_file.is_file():
        command += ["--env-file", str(env_file)]
    return command


def _init(args: argparse.Namespace) -> int:
    mode = _mode(args)
    runtime_root = _runtime_root(args)
    if mode == "local_file":
        for directory in (runtime_root, runtime_root / "runs", runtime_root / "artifacts", runtime_root / "scheduler", runtime_root / "processes"):
            directory.mkdir(parents=True, exist_ok=True)
        profile = runtime_root / "profile.json"
        if profile.exists() and not args.force:
            print(f"{profile} already exists; use --force to replace it")
            return 1
        profile.write_text(json.dumps({"schema_version": "local_profile.v1", "mode": "local_file", "runtime_root": str(runtime_root)}, indent=2) + "\n", encoding="utf-8")
        print(f"Initialized Local File Mode at {runtime_root}")
        return 0
    destination = ROOT / ".env.staging"
    if destination.exists() and not args.force:
        print(f"{destination} already exists; use --force to replace it")
        return 1
    shutil.copyfile(ENV_EXAMPLE, destination)
    print(f"Created {destination}. Replace every placeholder with server secrets before start.")
    return 0


def _pid_path(runtime_root: Path, name: str) -> Path:
    return runtime_root / "processes" / f"{name}.json"


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError, SystemError):
        return False
    return True


def _start_local(args: argparse.Namespace) -> int:
    runtime_root = _runtime_root(args)
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "processes").mkdir(parents=True, exist_ok=True)
    pid_file = _pid_path(runtime_root, "api")
    if pid_file.exists():
        try:
            record = json.loads(pid_file.read_text(encoding="utf-8"))
            if not _pid_running(int(record["pid"])):
                raise OSError("process is not running")
            print(f"Local API already running with pid {record['pid']}")
            return 0
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pid_file.unlink(missing_ok=True)
    child_env = os.environ.copy()
    child_env["ROBOTLAB_MODE"] = "local_file"
    child_env["ROBOTLAB_RUNTIME_DIR"] = str(runtime_root)
    command = [sys.executable, "-m", "uvicorn", "apps.api.main:app", "--host", args.host, "--port", str(args.port)]
    process = _spawn_local_process(runtime_root, "api", command, child_env)
    worker = _spawn_local_process(runtime_root, "worker", [sys.executable, "-m", "backend.app.workers.local_worker"], child_env)
    probe_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    api_url = f"http://{probe_host}:{args.port}"
    if not _wait_for_api(api_url, process):
        for child in (process, worker):
            if child.poll() is None:
                child.terminate()
        print(f"Local API failed health check; inspect {runtime_root / 'processes' / 'api.log'}", file=sys.stderr)
        return 1
    print(f"Started Local File Mode API on http://{args.host}:{args.port} (pid {process.pid})")
    print(f"Started Local File Mode worker (pid {worker.pid})")
    return 0


def _spawn_local_process(runtime_root: Path, name: str, command: list[str], environment: dict[str, str]):
    log_path = runtime_root / "processes" / f"{name}.log"
    pid_file = _pid_path(runtime_root, name)
    log_stream = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, stdout=log_stream, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
    pid_file.write_text(json.dumps({"schema_version": "local_process.v1", "name": name, "pid": process.pid, "command": command, "started_at": time.time()}, indent=2) + "\n", encoding="utf-8")
    log_stream.close()
    return process


def _stop_local(args: argparse.Namespace) -> int:
    runtime_root = _runtime_root(args)
    stopped = 0
    for pid_file in sorted((runtime_root / "processes").glob("*.json")) if (runtime_root / "processes").exists() else []:
        try:
            record = json.loads(pid_file.read_text(encoding="utf-8"))
            pid = int(record["pid"])
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
            else:
                os.kill(pid, signal.SIGTERM)
            stopped += 1
        except (OSError, ProcessLookupError, PermissionError, SystemError, ValueError, KeyError, json.JSONDecodeError):
            pass
        pid_file.unlink(missing_ok=True)
    print(f"Stopped {stopped} Local File Mode process(es); runtime data was preserved")
    return 0


def _status_local(args: argparse.Namespace) -> int:
    runtime_root = _runtime_root(args)
    processes = []
    process_root = runtime_root / "processes"
    if process_root.exists():
        for pid_file in sorted(process_root.glob("*.json")):
            try:
                record = json.loads(pid_file.read_text(encoding="utf-8"))
                running = _pid_running(int(record["pid"]))
                processes.append({"name": record.get("name"), "pid": record.get("pid"), "state": "running" if running else "stopped"})
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                processes.append({"name": pid_file.stem, "state": "stopped"})
    print(json.dumps({"mode": "local_file", "runtime_root": str(runtime_root), "processes": processes}, ensure_ascii=False, indent=2))
    return 0


def _robot_add(args: argparse.Namespace) -> int:
    try:
        record = LocalRobotRegistry(_runtime_root(args)).add(Path(args.path), robot_id=args.robot_id)
    except RobotRegistryError as exc:
        print(f"Robot registration failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def _robot_list(args: argparse.Namespace) -> int:
    records = LocalRobotRegistry(_runtime_root(args)).list()
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


def _install(args: argparse.Namespace) -> int:
    command = [sys.executable, "-m", "pip", "install", "-e", "."]
    print("Project-only install (third-party runtimes are never downloaded):")
    print(" ".join(command))
    if args.dry_run:
        return 0
    return _run(command, capture=True)


def _request_json(url: str, method: str, payload: dict[str, Any] | None, headers: dict[str, str]) -> int:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(response.read().decode("utf-8"))
            return 0
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"), file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"API request failed: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotlab")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="check host, runtime and API prerequisites")
    doctor.add_argument("--api-url", default=os.getenv("ROBOTLAB_API_URL", "http://127.0.0.1:8000"))
    doctor.add_argument("--env-file", default=None)
    doctor.add_argument("--mode", choices=("local_file", "compose", "memory"), default=None)
    doctor.add_argument("--runtime-dir", default=None)
    doctor.add_argument("--gpu", action="store_true", help="require NVIDIA and external Isaac/GMR runtimes")
    doctor.add_argument("--json", action="store_true")
    init = sub.add_parser("init", help="initialize Local File Mode or a Compose env template")
    init.add_argument("--force", action="store_true")
    init.add_argument("--mode", choices=("local_file", "compose"), default=None)
    init.add_argument("--runtime-dir", default=None)
    robot = sub.add_parser("robot", help="register and inspect user-provided robot assets")
    robot_sub = robot.add_subparsers(dest="robot_command", required=True)
    robot_add = robot_sub.add_parser("add")
    robot_add.add_argument("--path", required=True)
    robot_add.add_argument("--robot-id", default=None)
    robot_add.add_argument("--runtime-dir", default=None)
    robot_sub.add_parser("list").add_argument("--runtime-dir", default=None)
    install = sub.add_parser("install", help="install this repository into the active environment")
    install.add_argument("--dry-run", action="store_true")
    for name, action in (("start", "up"), ("stop", "down"), ("status", "ps")):
        command = sub.add_parser(name)
        command.add_argument("--env-file", default=None)
        command.add_argument("--gpu", action="store_true", help="include the GPU worker profile")
        command.add_argument("--mode", choices=("local_file", "compose"), default=None)
        command.add_argument("--runtime-dir", default=None)
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=8000)
        command.set_defaults(compose_action=action)
    logs = sub.add_parser("logs")
    logs.add_argument("service", nargs="?", default="api")
    logs.add_argument("--env-file", default=None)
    logs.add_argument("--gpu", action="store_true")
    logs.add_argument("--mode", choices=("local_file", "compose"), default=None)
    logs.add_argument("--runtime-dir", default=None)
    run = sub.add_parser("run", help="invoke a run operation through the API")
    run.add_argument("operation", choices=("train", "export", "sim2sim"))
    run.add_argument("run_id")
    run.add_argument("--api-url", default=os.getenv("ROBOTLAB_API_URL", "http://127.0.0.1:8000"))
    run.add_argument("--worker-id", default=os.getenv("ROBOTLAB_WORKER_ID", "robotlab-cli"))
    run.add_argument("--worker-token", default=os.getenv("WORKER_AUTH_TOKEN", ""))
    run.add_argument("--config", type=Path, help="TrainingConfig JSON for train")
    artifact = sub.add_parser("artifact")
    artifact.add_argument("artifact_id")
    artifact.add_argument("--api-url", default=os.getenv("ROBOTLAB_API_URL", "http://127.0.0.1:8000"))
    artifact.add_argument("--user-id", default=os.getenv("ROBOTLAB_USER_ID", "local-user"))
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "init":
        return _init(args)
    if args.command == "robot":
        return _robot_add(args) if args.robot_command == "add" else _robot_list(args)
    if args.command == "install":
        return _install(args)
    if args.command in {"start", "stop", "status", "logs"}:
        if _mode(args) == "local_file":
            if args.command == "start":
                return _start_local(args)
            if args.command == "stop":
                return _stop_local(args)
            if args.command == "status":
                return _status_local(args)
            log_path = _runtime_root(args) / "processes" / f"{args.service}.log"
            if not log_path.exists():
                print(f"No local log found: {log_path}")
                return 1
            return _run(["tail", "-f", str(log_path)], capture=False)
        command = _compose_args(args)
        if args.command == "logs":
            if args.gpu:
                command += ["--profile", "gpu"]
            return _run(command + ["logs", "-f", args.service], capture=False)
        if args.compose_action == "up":
            command += ["--profile", "gpu"] if args.gpu else []
            command += ["up", "-d"]
        else:
            command += [args.compose_action]
        return _run(command, capture=True)
    if args.command == "artifact":
        return _request_json(args.api_url.rstrip("/") + f"/api/v1/artifacts/{args.artifact_id}", "GET", None, {"X-User-Id": args.user_id})
    payload: dict[str, Any] = {}
    if args.operation == "train":
        if not args.config:
            print("--config is required for train", file=sys.stderr)
            return 2
        payload = json.loads(args.config.read_text(encoding="utf-8"))
    return _request_json(args.api_url.rstrip("/") + f"/api/v1/runs/{args.run_id}/{args.operation}", "POST", payload, {"X-Worker-Id": args.worker_id, "X-Worker-Token": args.worker_token})


if __name__ == "__main__":
    raise SystemExit(main())
