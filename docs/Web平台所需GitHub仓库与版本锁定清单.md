# Web 平台所需 GitHub 仓库与版本锁定清单

| 项目 | 内容 |
| --- | --- |
| 用途 | GVHMR/GMR → Isaac Lab/Isaac Sim → Unitree MuJoCo sim2sim → Web 编排平台 |
| 当前服务器 | Ubuntu 22.04，双 RTX 4090 |
| 当前 Conda 环境 | `unitree_g1_train` |
| 当前已确认核心版本 | Isaac Lab `v2.3.0`，Isaac Sim `5.1.0.0`，Python `3.11` |
| 版本原则 | 源码可以保留多版本，运行时只能使用经过验证的版本组合 |

## 1. 必需仓库

### 1.1 Isaac 训练和仿真

| 仓库 | 链接 | 版本锁定 |
| --- | --- | --- |
| Isaac Lab | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | `v2.3.0`，服务器 commit `3c6e67bb5` |
| Isaac Sim | [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) | 服务器 pip 包 `5.1.0.0`；源码 commit 尚未采集 |
| MuJoCo | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | Unitree MuJoCo README 使用 `3.3.6` |
| MuJoCo Menagerie | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 可选机器人 MJCF 资产，按 commit 锁定 |
| Mink | [kevinzakka/mink](https://github.com/kevinzakka/mink) | GMR 的 IK 依赖，需写入 GMR lockfile |

### 1.2 人体动作和机器人重定向

| 仓库 | 链接 | 版本锁定 |
| --- | --- | --- |
| GMR | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | README `0.2.0`；当前本地 commit `bb1bbe4` |
| GVHMR | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | 当前本地 commit `6ec3ca3`；独立 Python 3.10 环境 |

GVHMR 当前 requirements 使用 Torch `2.3.0+cu121`、Python 3.10 和 PyTorch3D 预编译包，不应直接安装到 Isaac Lab 的 Torch `2.7.0+cu128` 环境。

### 1.3 Unitree 适配和 sim2real

| 仓库 | 链接 | 作用/版本 |
| --- | --- | --- |
| Unitree RL Lab | [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | 服务器包 `0.2.1`；用于 Isaac Lab 任务和配置参考 |
| Unitree MuJoCo | [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | 上游基线 commit `ae6a840`；当前工作区有未提交 overlay |
| Unitree ROS | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | URDF 和网格；本地 commit `d6f13aa` |
| Unitree SDK2 | [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) | C++ DDS/低层控制接口 |
| Unitree SDK2 Python | [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | Python DDS/低层控制接口 |
| Unitree ROS2 | [unitreerobotics/unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) | ROS2 控制和 sim2real 示例 |

Unitree MuJoCo README 指定其基于 Unitree SDK2 和 MuJoCo，C++ 版本推荐用于低层 sim2real；Python 版本用于快速验证。

## 2. Web 平台实现仓库

这些仓库是平台代码依赖，不改变 Isaac 运行时版本。实际开发时必须通过 `package-lock.json`、`pnpm-lock.yaml`、`uv.lock` 或容器 digest 锁定。

| 组件 | GitHub | MVP 用途 |
| --- | --- | --- |
| React | [facebook/react](https://github.com/facebook/react) | 前端界面 |
| Vite | [vitejs/vite](https://github.com/vitejs/vite) | React 构建 |
| FastAPI | [fastapi/fastapi](https://github.com/fastapi/fastapi) | Python API |
| Pydantic | [pydantic/pydantic](https://github.com/pydantic/pydantic) | Run Manifest/schema 校验 |
| Redis | [redis/redis](https://github.com/redis/redis) | 作业队列和事件缓存，可选 |
| Celery | [celery/celery](https://github.com/celery/celery) | GPU worker 编排，可选 |

MVP 不要求把 Isaac Sim、GVHMR、GMR 和 Web API 放进同一个 Python 环境。推荐 Web API、GMR、GVHMR、Isaac Lab 分环境或分容器，通过文件产物和 JSON manifest 交接。

## 3. 当前服务器实测版本

服务器环境 `unitree_g1_train` 的已知输出：

```text
isaaclab: 0.47.2
IsaacLab Git tag: v2.3.0
IsaacLab Git commit: 3c6e67bb5
isaacsim: 5.1.0.0
isaacsim-app: 5.1.0.0
unitree-rl-lab: 0.2.1
torch: 2.7.0+cu128
Python: 3.11
mujoco Python package: NOT INSTALLED
```

`isaaclab` 的 `0.47.2` 是 Python 分发包内部版本；平台 manifest 应以 Git tag/commit 作为 Isaac Lab 源码身份，以 pip distribution 作为运行时身份，两者都保存。

## 4. 版本确认命令

在服务器执行：

```bash
source /ai/python/miniconda3/etc/profile.d/conda.sh && conda activate unitree_g1_train && python3 -c 'import importlib.metadata as md; from pathlib import Path; print("IsaacLab_Git="+Path("/ai/users/huangwy/G1/IsaacLab/VERSION").read_text().strip()); print("IsaacLab_Package="+md.version("isaaclab")); print("IsaacSim="+md.version("isaacsim")); print("IsaacSim_App="+md.version("isaacsim-app")); print("UnitreeRL="+md.version("unitree-rl-lab")); print("Python="+__import__("sys").version.split()[0]); print("Torch="+md.version("torch"))'
```

还需要补采集 Unitree RL Lab 和 Isaac Sim 的 Git/source identity：

```bash
git -C /ai/users/huangwy/G1/unitree_rl_lab rev-parse HEAD
python3 -c 'import importlib.metadata as md; print(md.metadata("isaacsim").get("Home-page")); print(md.version("isaacsim"))'
```

## 5. 本机源码 checkout 建议

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab fetch --all --tags
git -C IsaacLab checkout v2.3.0

git clone https://github.com/isaac-sim/IsaacSim.git isaacsim-source
git -C isaacsim-source fetch --all --tags
git -C isaacsim-source lfs install

git clone https://github.com/unitreerobotics/unitree_rl_lab.git
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/YanjieZe/GMR.git
git clone https://github.com/zju3dv/GVHMR.git
git clone https://github.com/google-deepmind/mujoco.git
```

本机可以保留 Isaac Lab 的多个 tags，但运行验证先只对 `v2.3.0 + Isaac Sim 5.1.0.0`。不要将 Isaac Lab `main/develop` 直接用于当前 Unitree RL Lab。

## 6. 平台必须保存的版本字段

```json
{
  "isaac_lab_git": "v2.3.0@3c6e67bb5",
  "isaac_lab_package": "0.47.2",
  "isaac_sim_package": "5.1.0.0",
  "unitree_rl_lab_package": "0.2.1",
  "unitree_rl_lab_git": "<server_sha_required>",
  "unitree_mujoco_git": "ae6a840",
  "unitree_mujoco_overlay": "<project_overlay_sha_required>",
  "mujoco_runtime": "3.3.6",
  "gmr_git": "bb1bbe4",
  "gvhmr_git": "6ec3ca3",
  "python": "3.11",
  "torch": "2.7.0+cu128"
}
```

## 7. 重要限制

- Isaac Sim 5.1.0.0 的运行包已经在服务器中，源码仓库不等于已验证的运行版本；必须保存 pip 包、源码和容器信息的对应关系。
- GMR README 主要测试 Ubuntu 20.04/22.04；GVHMR requirements 明确偏向 Linux/CUDA 环境。
- Unitree MuJoCo 当前工作区存在未提交改动，不能只记录上游 commit。
- Python `mujoco` 未安装在 Isaac Lab 环境中；GMR Python 可视化和 Python 版 Unitree MuJoCo 应使用独立环境。
- SMPL/SMPLX、GVHMR checkpoint、AMASS 等数据不属于 GitHub 源码仓库，必须单独记录下载地址、许可证和 SHA-256。
