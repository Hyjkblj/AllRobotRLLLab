# 部署与 CI/CD

本项目的部署分为平台服务和 GPU 运行时两条链路。平台服务可以由容器持续交付；Isaac Sim、Isaac Lab、GMR、GVHMR、Unitree MuJoCo 及机器人资产必须在受控服务器上按 `README.md` 的版本和 SHA 安装或只读挂载，不复制到本仓库镜像。

## 1. 流水线边界

```text
Pull Request
  -> Python/数据库/边界检查 + React build
  -> 合并 main
  -> 构建平台镜像并推送私有 Registry
  -> Staging Compose 部署
  -> API/异步任务验收
  -> GPU 验收门（手动或 nightly）
  -> Production 发布
```

`.github/workflows/ci.yml` 是当前仓库的第一条质量流水线。它不执行 Isaac Sim 或真实 GPU 训练；这些任务必须在自托管 GPU Runner 或指定服务器执行，并记录驱动、CUDA、Isaac、资产和 Git SHA。

## 2. 本地构建检查

在 Conda 环境中执行：

```powershell
python scripts/check_repo_boundary.py
python -m pytest -q
docker compose -f infra/compose/docker-compose.p2.yml config
docker compose -f infra/compose/docker-compose.staging.yml --env-file .env.staging config
docker build -f infra/docker/platform.Dockerfile -t allrobotrl-platform:local .
```

`.env.staging` 只放在服务器，不提交 Git；可以从 [`.env.staging.example`](../.env.staging.example) 复制。最小配置如下，生产环境请使用 Secret 管理器注入高强度随机值：

```dotenv
POSTGRES_PASSWORD=replace-with-a-secret
MINIO_ROOT_USER=allrobotrl
MINIO_ROOT_PASSWORD=replace-with-a-secret
MINIO_BUCKET=allrobotrl
API_PORT=8000
```

## 3. Staging 部署

服务器准备 Docker Engine、Compose v2 和外部第三方运行时后，在仓库根目录执行。当前 Compose 覆盖 API、Celery、PostgreSQL、Redis 和 MinIO；`frontend-prototype` 的 MuJoCo 服务仍需外部挂载 G1 资产，待阶段 F 接入正式 API 后再纳入同一生产入口：

```bash
docker login registry.example.com
export PLATFORM_IMAGE=registry.example.com/robot/allrobotrl-platform
export IMAGE_TAG="$GIT_SHA"
docker compose -f infra/compose/docker-compose.staging.yml --env-file .env.staging pull
docker compose -f infra/compose/docker-compose.staging.yml --env-file .env.staging up -d
docker compose -f infra/compose/docker-compose.staging.yml --env-file .env.staging ps
curl --fail http://127.0.0.1:8000/api/v1/health
curl --fail http://127.0.0.1:8000/api/v1/health/infrastructure
```

首次部署时，PostgreSQL 容器会从 `infra/migrations` 执行初始化迁移。后续迁移必须使用向后兼容的版本化脚本，并在 staging 先验证；不要删除持久化卷来“解决”迁移问题。

## 4. 生产 GPU Worker

GPU Worker 与 API 镜像分开发布。它需要额外的 NVIDIA Container Toolkit、GPU 驱动和外部只读挂载：

```bash
export ISAACLAB_PATH=/opt/IsaacLab
export ISAACSIM_PATH=/opt/isaac-sim
export GMR_PATH=/opt/allrobotrl/third_party/GMR-master
export GVHMR_PATH=/opt/allrobotrl/third_party/GVHMR-main
export UNITREE_MUJOCO_PATH=/opt/allrobotrl/third_party/unitree_mujoco-main
celery -A backend.app.workers.celery_app:celery_app worker \
  --loglevel=INFO -Q isaac-gpu,sim2sim-gpu --concurrency=1
```

当前仓库的 Celery task 已建立稳定任务名和幂等键，但真实 Isaac/RSL-RL runner、Unitree MuJoCo adapter 和 GPU lease 仍需在服务器阶段接入。没有 GPU 运行证据时，不能把 Run 标记为 `READY_TO_DOWNLOAD`。

## 5. 发布和回滚

- 镜像使用 Git commit SHA 标记；`latest` 只允许用于本地开发。
- 发布前保留上一版本镜像和数据库备份。
- 回滚优先切回上一镜像 SHA；数据库迁移必须已向后兼容，不能依赖回滚 SQL。
- API、Worker、前端和 GPU Worker 分别记录版本，策略包 manifest 记录所有运行时版本和 SHA-256。
- 生产环境禁止使用 Compose 文件中的开发默认密码，禁止把 `.env.staging`、checkpoint、权重和第三方目录提交到 Git。
