"""The project-owned deployment and operator CLI.

Commands are intentionally wrappers around documented tools.  They report
commands for missing system components instead of silently changing a host.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "infra" / "compose" / "docker-compose.staging.yml"
ENV_EXAMPLE = ROOT / ".env.staging.example"


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


def _doctor(args: argparse.Namespace) -> int:
    env_path = Path(args.env_file).expanduser() if args.env_file else ROOT / ".env.staging"
    env_values = _load_dotenv(env_path)
    def env(name: str) -> str:
        return os.getenv(name, env_values.get(name, ""))

    checks: dict[str, dict[str, Any]] = {}
    checks["docker"] = {"ok": _docker_compose() is not None, "fix": "Install Docker Engine/Desktop with Compose v2"}
    checks["nvidia"] = {"ok": shutil.which("nvidia-smi") is not None, "fix": "Install NVIDIA driver and NVIDIA Container Toolkit on GPU hosts"}
    conda_env = os.getenv("CONDA_DEFAULT_ENV", "")
    checks["conda"] = {"ok": bool(conda_env), "value": conda_env or None, "fix": "conda activate the platform or Isaac runtime environment"}
    for variable in ("DATABASE_URL", "REDIS_URL", "MINIO_ENDPOINT", "WORKER_AUTH_TOKEN"):
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
    for variable in ("ISAACLAB_PATH", "ISAACSIM_PATH", "GMR_PATH", "GVHMR_PATH", "UNITREE_MUJOCO_PATH"):
        value = env(variable)
        checks[variable] = {"ok": bool(value) and Path(value).expanduser().is_dir(), "path": value or None, "fix": f"Set {variable} to the locked external runtime directory"}
    runtime_script = ROOT / "scripts" / "check_external_runtime.py"
    if all(checks[name].get("ok") for name in ("ISAACLAB_PATH", "ISAACSIM_PATH", "GMR_PATH", "GVHMR_PATH", "UNITREE_MUJOCO_PATH")):
        runtime_env = os.environ.copy()
        runtime_env.update({name: env(name) for name in ("ISAACLAB_PATH", "ISAACSIM_PATH", "GMR_PATH", "GVHMR_PATH", "UNITREE_MUJOCO_PATH", "UNITREE_RL_LAB_PATH")})
        result = subprocess.run([sys.executable, str(runtime_script)], cwd=ROOT, env=runtime_env, capture_output=True, text=True, check=False)
        checks["external_runtime"] = {"ok": result.returncode == 0, "details": (result.stdout + result.stderr).strip(), "fix": "Check the pinned Git SHA values in README.md"}
    else:
        checks["external_runtime"] = {"ok": False, "state": "not_ready", "fix": "Configure all external runtime paths, then rerun robotlab doctor"}
    checks["api"] = _api_probe(args.api_url)
    if args.json:
        print(json.dumps({"status": "ok" if all(item.get("ok") for item in checks.values()) else "degraded", "checks": checks}, ensure_ascii=False, indent=2))
    else:
        for name, item in checks.items():
            state = "OK" if item.get("ok") else "MISSING"
            print(f"[{state}] {name}: {item.get('value') or item.get('path') or item.get('error') or item.get('fix', '')}")
    return 0 if all(item.get("ok") for item in checks.values()) else 1


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
    destination = ROOT / ".env.staging"
    if destination.exists() and not args.force:
        print(f"{destination} already exists; use --force to replace it")
        return 1
    shutil.copyfile(ENV_EXAMPLE, destination)
    print(f"Created {destination}. Replace every placeholder with server secrets before start.")
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
    doctor.add_argument("--json", action="store_true")
    init = sub.add_parser("init", help="create a local staging env template")
    init.add_argument("--force", action="store_true")
    install = sub.add_parser("install", help="install this repository into the active environment")
    install.add_argument("--dry-run", action="store_true")
    for name, action in (("start", "up"), ("stop", "down"), ("status", "ps")):
        command = sub.add_parser(name)
        command.add_argument("--env-file", default=None)
        command.add_argument("--gpu", action="store_true", help="include the GPU worker profile")
        command.set_defaults(compose_action=action)
    logs = sub.add_parser("logs")
    logs.add_argument("service", nargs="?", default="api")
    logs.add_argument("--env-file", default=None)
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
    if args.command == "install":
        return _install(args)
    if args.command in {"start", "stop", "status", "logs"}:
        command = _compose_args(args)
        if args.command == "logs":
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
