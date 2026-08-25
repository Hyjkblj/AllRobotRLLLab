# AllRobotRLLLab

G1 29 DoF RL 平台的分层开发基线。仓库只提交本项目源码、契约、测试、配置、迁移和前端代码；第三方仓库、仿真 SDK、网格、模型权重和检查点不提交到 Git。第三方依赖必须按下面的版本和目录约定安装。

## 快速开始

```powershell
conda env create -f environment-platform.yml
conda activate allrobotrl-platform
python -m pip install -r requirements-platform.txt
python -m pytest -q
uvicorn apps.api.main:app --reload
```

API 地址：`http://127.0.0.1:8000/api/v1/health`。详细边界、接口和后续 P1/P2 工作见 [docs/P0实现说明.md](docs/P0实现说明.md) 与方案文档。

P1 动作编辑和 Reward Builder 的接口与约束见 [docs/P1实现说明.md](docs/P1实现说明.md)。

P2 后端闭环（项目、资源、Run/Attempt、SSE、权限、outbox 和对象存储边界）见 [docs/P2实现说明.md](docs/P2实现说明.md)。

P3 训练、TorchScript/ONNX 导出、三种子 sim2sim、策略包校验和产物下载见 [docs/P3实现说明.md](docs/P3实现说明.md)。本地 Docker 端口与 smoke 闭环命令也在该文档中列出。

长任务异步模式使用 `EXECUTION_MODE=async` 和 Redis/Celery；本地默认保持 `sync_smoke`，便于契约测试。

阶段 A–F 的完成度和剩余生产缺口见 [docs/阶段完成度.md](docs/阶段完成度.md)。

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
```

Windows 本地 MuJoCo 原型只需要 G1 MJCF/URDF 和网格位于 `third_party/GMR-master/assets/unitree_g1`；生产部署应将这些上游资产作为镜像或外部只读 volume 挂载。版本、许可证和 SHA-256 必须写入 Run Manifest，不能用未锁定的 `main` 分支替代。
