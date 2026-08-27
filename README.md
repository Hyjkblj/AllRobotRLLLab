# AllRobotRLLLab

G1 29 DoF RL 平台的分层开发基线。仓库只提交本项目源码、契约、测试、配置、迁移和前端代码；第三方仓库、仿真 SDK、网格、模型权重和检查点不提交到 Git。第三方依赖必须按下面的版本和目录约定安装。

## 快速开始

```powershell
conda env create -f environment-platform.yml
conda activate allrobotrl-platform
python -m pip install -r requirements-platform.txt
python -m pip install -e .
python -m pytest -q
python -m tools.robotlab init --mode local_file
python -m tools.robotlab start --mode local_file
```

API 地址：`http://127.0.0.1:8000/api/v1/health`。默认开发模式是无数据库的 Local File Mode，运行数据写入 `runtime/`，不需要 Docker、PostgreSQL、Redis 或 MinIO。需要团队共享或远程 worker 时，再显式使用 Compose Mode。详细边界、接口和后续 P1/P2 工作见 [docs/P0实现说明.md](docs/P0实现说明.md)、[docs/P2实现说明.md](docs/P2实现说明.md) 与方案文档。

运维入口安装项目自身代码后可直接使用 `robotlab`（也可执行 `python -m tools.robotlab`）：

```bash
robotlab doctor --json
robotlab install
robotlab init --mode local_file
robotlab start --mode local_file
robotlab status
robotlab logs api
robotlab stop
```

`doctor` 只检查并给出缺失组件的安装/配置指引，不会自动修改操作系统、Docker、Conda 或第三方运行时。

Local File Mode 的运行目录可以通过 `ROBOTLAB_RUNTIME_DIR` 指定；`robotlab start --mode local_file` 会启动本地 API 和持久化任务 worker。切换到 PostgreSQL/Redis/MinIO 时使用 `robotlab init --mode compose` 和 staging Compose 配置。

P1 动作编辑和 Reward Builder 的接口与约束见 [docs/P1实现说明.md](docs/P1实现说明.md)。

P2 后端闭环（项目、资源、Run/Attempt、SSE、权限、outbox 和对象存储边界）见 [docs/P2实现说明.md](docs/P2实现说明.md)。

P3 训练、TorchScript/ONNX 导出、三种子 sim2sim、策略包校验和产物下载见 [docs/P3实现说明.md](docs/P3实现说明.md)。本地 Docker 端口与 smoke 闭环命令也在该文档中列出。

长任务异步模式使用 `EXECUTION_MODE=async` 和 Redis/Celery；本地默认保持 `sync_smoke`，便于契约测试。

阶段 A–F 的完成度和剩余生产缺口见 [docs/阶段完成度.md](docs/阶段完成度.md)。

## CI/CD 与部署

仓库的 PR 质量门见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)，包含 Python/数据库测试、仓库边界检查、Compose 校验和 React 构建。平台服务镜像由 [`infra/docker/platform.Dockerfile`](infra/docker/platform.Dockerfile) 构建；staging 编排见 [`infra/compose/docker-compose.staging.yml`](infra/compose/docker-compose.staging.yml)。

标准部署、服务器初始化、GPU Worker、发布和回滚流程见 [docs/部署与CI-CD.md](docs/部署与CI-CD.md)。CI 不下载或提交第三方仓库；真实 Isaac/GMR/GVHMR/Unitree 运行时必须在 GPU 服务器按本 README 的版本锁定清单安装。

## 第三方依赖

第三方依赖不属于本仓库提交范围。以下路径相对于仓库根目录；如果只运行 API contract/smoke tests，至少安装 GMR 的 G1 资产。真实动作恢复、Isaac 训练和 Unitree sim2sim 需要对应的 Linux/GPU 环境。

| 组件 | 版本锁定 | 本地目录 | 安装/获取方式 |
| --- | --- | --- | --- |
| GMR | `0.2.0`, commit `bb1bbe4` | `third_party/GMR-master` | `git clone https://github.com/YanjieZe/GMR.git third_party/GMR-master; git -C third_party/GMR-master checkout bb1bbe4` |
| GVHMR | commit `6ec3ca3` | `third_party/GVHMR-main` | `git clone https://github.com/zju3dv/GVHMR.git third_party/GVHMR-main; git -C third_party/GVHMR-main checkout 6ec3ca3` |
| Mink | 由 GMR lockfile/环境锁定 | `third_party/mink-main` | 在 GMR Python 环境按其 lockfile 安装，不与 Isaac 环境混装 |
| Isaac Lab | tag `v2.3.0`, commit `3c6e67bb5`; Python package `0.47.2` | `third_party/IsaacLab-2.3.0` 或外部 `ISAACLAB_PATH` | `git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab-2.3.0; git -C third_party/IsaacLab-2.3.0 checkout v2.3.0; ./isaaclab.sh --install` |
| Isaac Sim | package `5.1.0.0` | 外部 `ISAACSIM_PATH` | 通过 NVIDIA Omniverse/Isaac Sim 官方安装器安装；需要接受其许可证，不从本仓库下载 |
| Unitree RL Lab | package `0.2.1` | `third_party/unitree_rl_lab-main` 或外部路径 | `git clone https://github.com/unitreerobotics/unitree_rl_lab.git third_party/unitree_rl_lab-main; cd third_party/unitree_rl_lab-main; python -m pip install -e .`；生产环境还需记录实际 Git SHA |
| Unitree MuJoCo | upstream commit `ae6a840` | `third_party/unitree_mujoco-main` 或外部路径 | `git clone https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco-main; git -C third_party/unitree_mujoco-main checkout ae6a840` |
| MuJoCo Python | `3.12.0` Windows wheel；Unitree C++ baseline `3.3.6` | Conda 环境 | `python -m pip install -r frontend-prototype/requirements-mujoco.txt` |
| MuJoCo Menagerie | 按使用的机器人 commit 锁定 | `third_party/mujoco_menagerie-main` | `git clone https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie-main`，使用前记录 commit |
| Unitree ROS | commit `d6f13aa` | `third_party/unitree_ros-master` | `git clone https://github.com/unitreerobotics/unitree_ros.git third_party/unitree_ros-master; git -C third_party/unitree_ros-master checkout d6f13aa` |

GMR/GVHMR 使用独立环境，避免与 Isaac Lab 的 Torch/CUDA 版本冲突：

```bash
# GMR / Mink / MuJoCo / SMPL-X environment
conda create -n allrobotrl-gmr python=3.10 -y
conda activate allrobotrl-gmr
git clone https://github.com/YanjieZe/GMR.git third_party/GMR-master
git -C third_party/GMR-master checkout bb1bbe4
python -m pip install -r third_party/GMR-master/requirements.txt

# GVHMR environment; its baseline is Torch 2.3.0+cu121 and Python 3.10.
conda create -n allrobotrl-gvhmr python=3.10 -y
conda activate allrobotrl-gvhmr
git clone https://github.com/zju3dv/GVHMR.git third_party/GVHMR-main
git -C third_party/GVHMR-main checkout 6ec3ca3
python -m pip install -r third_party/GVHMR-main/requirements.txt
```

Isaac Lab/Isaac Sim 需要 Ubuntu 22.04、Python 3.11、CUDA/Torch 与 NVIDIA 驱动匹配的 GPU 主机；平台 Web/API 环境 `allrobotrl-platform` 不加载 Isaac Sim。运行时路径通过环境变量提供：

```bash
export ISAACLAB_PATH=/opt/IsaacLab
export ISAACSIM_PATH=/opt/isaac-sim
export GMR_PATH=$PWD/third_party/GMR-master
export GVHMR_PATH=$PWD/third_party/GVHMR-main
export UNITREE_MUJOCO_PATH=$PWD/third_party/unitree_mujoco-main
export G1_ISAAC_URDF_PATH=/opt/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf
python scripts/collect_runtime_manifest.py --output .runtime/runtime-manifest.json
```

Isaac Sim 5.1 may be installed as the `isaacsim==5.1.0.0` Python package
inside the active Conda environment instead of an `isaac-sim.sh` checkout. In
that case `check_external_runtime.py`, `collect_runtime_manifest.py` and
`robotlab doctor` discover the package automatically when `ISAACSIM_PATH` is
unset. Set `ISAACSIM_PATH` explicitly when using a mounted SDK directory or a
container image.

在无 X Server 的 GPU 服务器上，先用 physics-only probe 验证 Isaac Sim
启动和 PhysX 更新，不要直接把 RTX shutdown 的段错误当成训练失败：

```bash
python scripts/probe_isaacsim.py \
  --frames 5 \
  --output .runtime/isaacsim-probe.json
# 需要专门验证退出路径时再加 --close
python scripts/probe_isaacsim.py --frames 5 --close
```

输出中的 `startup=passed` 和 `update=passed` 才表示运行时可用于物理仿真；
`--close` 失败应单独记录为 Isaac Sim 上游关闭问题。该 probe 不执行训练、
不下载资产，也不写入第三方目录。

Windows 本地 MuJoCo 原型只需要 G1 MJCF/URDF 和网格位于 `third_party/GMR-master/assets/unitree_g1`；生产部署应将这些上游资产作为镜像或外部只读 volume 挂载。版本、许可证和 SHA-256 必须写入 Run Manifest，不能用未锁定的 `main` 分支替代。

Unitree RL Lab 的 G1 29 DoF mimic 配置使用 Unitree ROS 的
`g1_29dof_rev_1_0.urdf`（通过 Isaac Lab URDF importer 在运行时生成
USD），因此不要求预先提供本地 `G1_USD_PATH`。设置
`G1_ISAAC_URDF_PATH` 后，runtime manifest 会单独记录该训练 URDF 的路径、
大小和 SHA-256；`G1_USD_PATH` 仅用于实际选择 USD spawn 配置的任务。
