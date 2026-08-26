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
export IMAGE_TAG="sha-$GIT_SHA"
docker compose -f infra/compose/docker-compose.staging.yml --env-file .env.staging pull
docker compose -f infra/compose/docker-compose.staging.yml --env-file .env.staging up -d
docker compose -f infra/compose/docker-compose.staging.yml --env-file .env.staging ps
curl --fail http://127.0.0.1:8000/api/v1/health
curl --fail http://127.0.0.1:8000/api/v1/health/infrastructure
```

服务器首次启动 GPU worker 时必须启用 Compose profile：

```bash
docker compose -f infra/compose/docker-compose.staging.yml --env-file .env.staging --profile gpu up -d
python scripts/collect_runtime_manifest.py --strict --output .runtime/runtime-manifest.json
```

`worker-cpu` 只消费 `cpu` 队列，训练和 sim2sim 的 `isaac-gpu`/`sim2sim-gpu` 队列由带 `gpus: all` 的 `worker-gpu` 消费。没有 NVIDIA Container Toolkit 或外部运行时目录时，GPU profile 应保持停止，不能把 CPU smoke 结果当作真实验收。

`migrate` 一次性服务会在 API/worker 前取得 PostgreSQL advisory lock，按文件名顺序执行未记录的迁移并写入 `schema_migrations`。这使已有数据库卷也能安全接收后续迁移；不要删除持久化卷来“解决”迁移问题。`004_p3_run_state.sql` 持久化训练配置、checkpoint、导出元数据、策略包和 sim2sim 报告。API 与 worker 共享 `staging-runtime` 卷，最终产物仍必须写入 MinIO。

验证 worker 重启恢复：提交一个 async train 后重启 worker，任务重新投递或重试时应保持相同 checkpoint id；随后从新的 worker 进程提交 export/sim2sim，不能依赖旧 API 进程内存。对应的本地契约测试是 `test_p3_state_survives_training_service_recreation` 和 `test_worker_replay_is_idempotent_after_completed_train`。

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

目标服务器如果通过 Conda 安装 `isaacsim==5.1.0.0`，不要求存在
`isaac-sim.sh`；运行时检查脚本会从当前 Python 环境的 `isaacsim` 包自动
发现安装根目录。使用容器或独立 SDK 时仍应设置 `ISAACSIM_PATH`，并将该
目录以只读方式挂载给 GPU worker。

当前仓库的 Celery task 已建立稳定任务名、幂等键、late acknowledgement、worker lost 重投和 durable P3 state；真实 Isaac/RSL-RL runner、Unitree MuJoCo adapter 和 GPU lease 仍需在服务器阶段接入。没有 GPU 运行证据时，不能把 Run 标记为 `READY_TO_DOWNLOAD`。

生产内部写接口要求同时提供 `X-Worker-Id` 和 `X-Worker-Token`。`WORKER_AUTH_TOKEN` 由服务器 Secret 管理器注入，长度至少 32 个字符；staging/production 缺失该变量时 API 会拒绝启动。

## 5. 发布和回滚

- 镜像使用 Git commit SHA 标记；`latest` 只允许用于本地开发。
- 发布前保留上一版本镜像和数据库备份。
- 回滚优先切回上一镜像 SHA；数据库迁移必须已向后兼容，不能依赖回滚 SQL。
- API、Worker、前端和 GPU Worker 分别记录版本，策略包 manifest 记录所有运行时版本和 SHA-256。
- 生产环境禁止使用 Compose 文件中的开发默认密码，禁止把 `.env.staging`、checkpoint、权重和第三方目录提交到 Git。
