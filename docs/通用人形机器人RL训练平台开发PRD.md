# 通用人形机器人 RL 训练平台开发 PRD

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.2（版本锁定与风险补充稿） |
| 文档状态 | Draft，历史基线；本地完整项目形态以 v0.3/v0.4 修订章节为准 |
| 目标用户 | 机器人算法工程师、仿真工程师、研究员、项目导师 |
| 首个支持机器人 | Unitree G1 29 DoF |
| 首个训练闭环 | 视频/动作资源 → GVHMR → GMR → 动作预处理 → Isaac Lab/Isaac Sim → MuJoCo sim2sim → 策略包 |
| 前端 | React + TypeScript |
| 后端 | Python API + 作业编排，训练执行以 Isaac Lab/Isaac Sim 为底座，参考 `unitree_rl_lab` 的组织形式 |
| 运行基线 | Windows WSL2 Ubuntu 22.04 与原生 Linux Ubuntu 22.04 均使用 Docker Compose；双 RTX 4090 为本地或后续 worker 资源 |

> 当前实现补充以 [《通用人形机器人RL训练平台开发PRD-v0.3.md》](./通用人形机器人RL训练平台开发PRD-v0.3.md) 的 v0.4 实现基线和 [《通用关节机器人 RL 训练平台完整闭环技术方案》](./G1人形机器人RL训练平台完整闭环技术方案.md) 为准。本文件保留 v0.2 的历史平台架构和版本约束；本地完整项目、CLI、Docker、WSL2/Linux、无登录和用户自带机器人资产等新决策不得按本文件早期 Web 描述实现。

## 1. 产品概述

### 1.1 背景

当前仓库已经具备一条可运行但依赖命令行和手工配置的 G1 训练链路。本 PRD 将其抽象为可替换的 Isaac Lab/Isaac Sim 执行后端：

1. `GVHMR` 将单目视频恢复为世界坐标系下的人体运动和 SMPL 参数。
2. `GMR` 使用 MuJoCo 模型和 mink IK 将人体运动重定向到机器人关节空间。
3. Isaac Lab 负责通用的人形机器人环境、任务注册、RL 训练和策略导出。
4. Isaac Sim 负责物理仿真运行；训练后可由 MuJoCo 或厂商仿真适配器执行独立 sim2sim 验收。

本地 `unitree_rl_lab` 不作为平台内核，而作为形式参考：它展示了如何组织机器人、任务、环境配置、RSL-RL agent、训练入口、play/export 和 deploy 配置。Isaac Lab/Isaac Sim 的具体版本、容器镜像和许可证需要在部署环境中登记并锁定。

这些步骤涉及多个环境、模型版本、坐标系和配置文件。平台需要将它们包装成可追踪、可复现、可扩展的训练作业，同时保留底层 CLI 的可调试性。

### 1.2 产品定位

平台是“训练工作流编排器和验证界面”，不是新的物理引擎或新的 RL 算法。它负责：

- 管理视频、动作、机器人模型、任务和奖励配置。
- 调用现有 GVHMR、GMR，并通过后端适配器调用 Isaac Lab/Isaac Sim。
- 把中间产物、日志、指标、版本和策略包统一归档。
- 通过机器人适配器把同一任务描述映射到不同厂家的机器人。

核心原则是“任务配置可复用，策略权重按机器人分别训练”。不承诺同一份权重直接跨机器人运行。

## 2. 目标与非目标

### 2.1 MVP 目标

- 用户上传一段动作视频，或选择已有动作资源。
- 在浏览器中选择 G1、动作模仿任务和奖励模板，启动一次完整训练作业。
- 实时查看每个阶段的状态、日志、视频和指标。
- 训练结束后自动导出 JIT/ONNX 权重、部署配置和元数据。
- 在 MuJoCo 中执行 sim2sim 验收，展示是否跌倒、是否崩溃、动作跟踪误差和任务成功率。
- 用统一的 `RobotSpec`、`TaskSpec`、`RewardConfig` 契约，为后续机器人适配器留出扩展点。

### 2.2 MVP 非目标

- 不在 Web 进程中直接启动 Isaac Lab/Isaac Sim 或加载 CUDA 仿真库。
- 不允许用户在网页上传任意 Python 代码或任意 shell 命令。
- 不承诺单目视频能恢复可靠的物体 6D 位姿；箱子任务第一阶段由用户提供箱子尺寸、质量、初始/目标位姿，或使用预定义仿真场景。
- 不在没有厂商 SDK、控制频率和关节定义的情况下支持实机部署。
- 不把“奖励高”作为唯一验收标准，也不把仿真通过等同于实机安全。

## 3. 现状与约束依据

### 3.1 GVHMR 输入输出

- 本地入口：`GVHMR/tools/demo/demo.py`。
- 典型输出：`hmr4d_results.pt`，包含 `smpl_params_global`，包括 `body_pose`、`betas`、`global_orient`、`transl`。
- GMR 的 `load_gvhmr_pred_file` 固定读取这些字段，并按 30 FPS 生成逐帧人体数据。
- GVHMR 使用 SMPL/SMPLX 模型和检查点，必须在作业中记录模型版本和来源。

### 3.2 GMR 重定向契约

- 本地入口：`GMR/scripts/gvhmr_to_robot.py`。
- 参数包括 `--gvhmr_pred_file`、`--robot`、`--save_path`、`--rate_limit` 和 `--record_video`。
- `GeneralMotionRetargeting` 读取目标机器人 MuJoCo XML、关节顺序、身体映射和 IK 配置。
- 原始输出包含 `fps`、`root_pos`、`root_rot`、`dof_pos` 等字段；脚本保存的根旋转为四元数 `xyzw`，而 GMR 内部 IK 使用标量在前的 `wxyz`。
- 坐标系转换、四元数约定和关节顺序必须在数据契约中显式声明，不能依赖调用方猜测。

### 3.3 Isaac Lab + Isaac Sim 目标执行底座

Isaac Lab + Isaac Sim 是平台的通用训练后端。平台不把仿真框架内部实现暴露给前端，而是约束一个稳定的后端适配器接口：

- `list_tasks()`：返回任务 id、机器人能力、观测/动作维度和支持的场景。
- `validate_config(manifest)`：校验机器人模型、动作资源、奖励项和 agent 参数。
- `train(manifest, output_dir)`：启动 headless RL 训练，持续输出结构化指标。
- `play(checkpoint, manifest, output_dir)`：固定种子回放并生成视频/轨迹。
- `export(checkpoint, manifest, output_dir)`：导出权重、归一化参数、控制参数和部署配置。
- `sim2sim(policy_bundle, sim_manifest, output_dir)`：在 Isaac Sim 中运行训练内验收；必要时再调用 MuJoCo 或厂商仿真进行独立 sim2sim。

Isaac Lab 适配器必须支持配置分层，至少包括：机器人模型、场景、观测、动作、命令、奖励、终止条件、域随机化和 PPO/其他 agent 配置。任务通过稳定 `task_id` 注册，具体 Python 模块、容器镜像和 CLI 由适配器维护。

参考执行形式（命令名由 Isaac Lab 版本适配器注入，以下不是固定命令）：

```bash
<isaaclab-cli> train --task <task_id> --manifest <run_manifest.json> --headless
<isaaclab-cli> play --task <task_id> --checkpoint <checkpoint> --manifest <run_manifest.json>
<isaaclab-cli> export --task <task_id> --checkpoint <checkpoint> --manifest <run_manifest.json>
<isaacsim-cli> evaluate --policy-bundle <bundle> --scenario <scenario> --seeds 0,1,2
```

### 3.4 Unitree RL Lab 的参考范围

本地 `unitree_rl_lab` 仅用于借鉴以下工程形式，不作为通用平台的运行依赖：

- 使用 task id 注册机器人任务，并通过配置入口选择环境和 agent。
- 将训练、play、export 分为独立命令，便于服务端重试和审计。
- 为动作模仿提供包含关节、身体位姿和速度的标准 `.npz`。
- 导出包含关节映射、动作缩放、观测归一化、PD 参数和控制步长的部署配置。
- 将每次运行的 env/agent 配置、checkpoint、日志和视频保存到独立目录。

因此，平台保留 Unitree 适配器作为第一个验证实现，但所有前端和核心 API 均面向 Isaac Lab 的通用 `BackendAdapter`。接入时必须锁定 Isaac Lab/Isaac Sim 版本、CLI、配置 schema、容器环境和许可证。

## 4. 目标用户与典型场景

### 4.1 角色

- 算法工程师：配置奖励、超参数、观察训练曲线，下载策略包。
- 仿真工程师：维护机器人适配器、场景、传感器和 sim2sim 验收规则。
- 项目负责人/导师：查看作业进度、对比实验、审阅验收报告。
- 平台管理员：管理 GPU worker、模型许可证和数据存储。

### 4.2 典型用户故事

1. 我上传一段“向前走并伸手”的视频，平台能够预览重定向结果并提示关节越限。
2. 我选择 G1 的动作模仿任务，调整“躯干姿态”和“关节跟踪”奖励权重，启动训练并查看日志。
3. 我把同一动作资源切换到另一个已适配机器人，平台自动提示缺失的身体映射、自由度或控制接口。
4. 我打开 sim2sim 验收结果，能看到三次不同随机种子的回放、失败原因和导出文件校验和。

## 5. 端到端业务流程

```text
创建项目
  → 上传视频/选择动作资源
  → 媒体与许可证校验
  → GVHMR（视频 → hmr4d_results.pt）
  → GMR（人体 → 机器人原始运动 .pkl）
  → 动作编译与质量校验（.pkl → 训练 .npz）
  → 选择机器人、任务、场景、奖励和训练超参数
  → 生成不可变 Run Manifest
  → Isaac Lab/选定 RL Agent 训练
  → checkpoint 与指标归档
  → policy.pt/policy.onnx/deploy.yaml 导出
  → MuJoCo sim2sim 验收
  → 生成验收报告与可部署策略包
```

### 5.1 作业状态机

`CREATED → UPLOADING → UPLOADED → GVHMR_RUNNING → GVHMR_READY → GMR_RUNNING → RETARGET_READY → MOTION_VALIDATING → TRAINING_PREPARING → TRAINING → TRAINING_SUCCEEDED → EXPORTING → SIM2SIM_RUNNING → SIM2SIM_PASSED/ SIM2SIM_FAILED → READY_TO_DOWNLOAD`

任一阶段都可以进入 `FAILED` 或 `CANCELLED`。状态变更必须记录时间、命令、版本、退出码和可读错误摘要。重试应生成新的 attempt，不覆盖原始产物。

## 6. 前端 PRD（React）

### 6.1 页面结构

1. **项目首页**：项目、最近作业、机器人适配状态、GPU worker 状态。
2. **新建训练向导**：资源、机器人、任务、奖励、训练参数、确认启动。
3. **动作预览**：视频播放器、人体关键点/SMPL 预览、GMR 机器人骨架、关节曲线和越限标记。
4. **机器人与任务选择**：显示 DoF、关节/身体覆盖率、控制模式、支持的任务能力和许可证状态。
5. **奖励编辑器**：模板、滑块/开关/数值输入、参数校验、贡献预览和高级 JSON 查看。
6. **训练监控**：阶段进度、实时日志、GPU/显存、回报曲线、每项奖励、episode 长度、跌倒率、视频。
7. **sim2sim 验收**：Isaac 与 MuJoCo 并排回放、三次种子结果、失败诊断、通过/不通过结论。
8. **产物详情**：checkpoint、导出权重、配置、日志、报告、校验和、下载按钮。

### 6.2 关键交互要求

- 上传组件限制文件类型、大小和时长，显示上传进度并支持断点重试。
- 机器人选择器在缺少 XML/URDF、网格、关节映射或部署适配器时阻止启动，并给出缺项。
- 奖励编辑器仅暴露注册过的奖励项；每项显示说明、单位、允许范围、默认值和是否影响终止条件。
- 预览总奖励时同时显示各项贡献，防止某一项权重数量级掩盖其他项。
- 日志和指标通过 SSE/WebSocket 增量刷新；刷新页面后可以从服务端恢复。
- 所有“下载到实机”的文案必须带许可证和安全提示，MVP 只导出部署包，不执行远程实机控制。

### 6.3 推荐技术栈

- React + TypeScript + Vite，React Router。
- TanStack Query 管理服务端状态，Zustand 仅保存向导临时状态。
- Recharts 或 ECharts 绘制训练曲线；Three.js/React Three Fiber 用于机器人和动作预览（可作为 P1）。
- `zod` 校验 API 数据和奖励表单；统一错误码和空状态。

## 7. 后端与作业编排 PRD

### 7.1 总体架构

```text
React Web
   │ REST + SSE/WebSocket
FastAPI API（鉴权、校验、元数据）
   │
PostgreSQL/SQLite ─ 对象存储或隔离文件目录
   │
Job Runner（Redis Queue/Celery 或本地进程队列）
   │
GPU Worker：GVHMR | GMR | Motion Compiler | Isaac Lab | MuJoCo
```

API 进程不导入 Isaac Sim，不在请求线程中运行训练。Worker 以显式工作目录执行白名单 CLI，捕获 stdout/stderr、退出码、环境变量摘要、GPU 编号和版本哈希。

### 7.2 服务模块

- `ProjectService`：项目、成员和作业列表。
- `AssetService`：视频、动作、模型、检查点和产物的上传、校验、生命周期。
- `GVHMRService`：调用 demo，校验 `smpl_params_global`，登记模型与许可证。
- `GMRService`：选择机器人适配器，执行重定向、录像和质量评估。
- `MotionCompilerService`：将原始 `.pkl` 编译成 Isaac Lab mimic 任务所需的标准 `.npz` 或后端声明的等价格式。
- `TaskConfigService`：根据 `TaskSpec` 和 `RewardConfig` 生成 Run Manifest/任务覆盖配置。
- `TrainingService`：启动、暂停（若引擎支持）、取消、恢复和监控 Isaac Lab agent 作业。
- `ExportService`：调用 play/export，打包权重、部署配置和元数据。
- `Sim2SimService`：运行 MuJoCo 验收场景，产出指标、录像和报告。

### 7.3 运行隔离

- 每次运行使用独立目录：`runs/<project_id>/<run_id>/<attempt>/`。
- 子进程命令使用参数数组，不拼接用户输入；路径必须经过工作区白名单校验。
- 训练容器或 worker 记录 Git commit、Python/CUDA/Isaac/MuJoCo 版本和依赖锁定文件。
- 超时、显存不足、进程崩溃和磁盘不足都要转成可恢复的错误码。

## 8. 统一数据契约

### 8.1 `RobotSpec`

```json
{
  "robot_id": "unitree_g1_29dof",
  "vendor": "Unitree",
  "model_version": "g1_29dof",
  "assets": {"mujoco_xml": "...", "urdf": "...", "usd": "..."},
  "joint_names": ["..."],
  "body_names": ["pelvis", "torso_link", "..."] ,
  "actuation": {"mode": "position", "control_dt": 0.02},
  "limits": {"position": "...", "velocity": "...", "torque": "..."},
  "action_scale": "...",
  "deploy_adapter": "unitree_g1_29dof",
  "capabilities": ["mimic", "velocity", "box_reach"]
}
```

适配器还必须提供 GMR 的 IK 配置：人体/机器人身体映射、位置/旋转权重、偏移四元数、人体高度假设、地面高度、坐标系和关节顺序。适配器自检失败时不能创建训练作业。

### 8.2 `RetargetMotion`（GMR 原始输出）

```json
{
  "format_version": "retarget_motion.v1",
  "robot_id": "unitree_g1_29dof",
  "fps": 30,
  "root_pos": "float32[N,3]",
  "root_rot": "float32[N,4]",
  "root_rot_convention": "xyzw",
  "dof_pos": "float32[N,D]",
  "joint_names": ["..."],
  "coord_frame": "world_z_up",
  "source": {"type": "gvhmr", "asset_id": "..."},
  "quality": {"nan_count": 0, "joint_limit_violation_ratio": 0.0}
}
```

### 8.3 `TrainMotionNPZ`

必须兼容 `MotionLoader`：`fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。编译器应验证形状、有限值、四元数归一化、身体名称覆盖率和帧率插值结果，并保存 `manifest.json` 说明来源和转换版本。

### 8.4 `RewardConfig` 与 `TrainingConfig`

```json
{
  "reward_version": "reward.v1",
  "terms": [
    {"id": "tracking.anchor_pos", "enabled": true, "weight": 1.0, "params": {"sigma": 0.3}},
    {"id": "regularization.action_rate", "enabled": true, "weight": -0.01, "params": {}},
    {"id": "task.box_goal_pose", "enabled": false, "weight": 2.0, "params": {}}
  ],
  "terminations": ["timeout", "bad_anchor_orientation"],
  "constraints": {"joint_limit_margin": 0.02, "max_torque_ratio": 1.0}
}
```

`TrainingConfig` 还需包含 `robot_id`、`task_id`、`motion_asset_id`、`scene_id`、`num_envs`、`seed`、`max_iterations`、`resume_checkpoint`、`record_video` 和 `git_commits`。启动时将所有配置冻结为不可变 Run Manifest。

### 8.5 `PolicyBundle`

```text
policy.pt
policy.onnx
params/deploy.yaml
params/env.yaml
params/agent.yaml
export_meta.json
sim2sim_report.json
videos/
checksums.sha256
```

`export_meta.json` 必须包含机器人、任务、观测/动作维度、关节顺序、动作缩放、观测归一化、控制频率、训练种子、依赖版本和许可证信息。

## 9. 任务与奖励设计

### 9.1 奖励分层

奖励按能力分为四类，避免每个动作都从零手写一套函数：

1. **通用稳定项**：存活、躯干朝向、基座高度、关节限位、扭矩、能耗、动作变化率、脚底滑动和非期望接触。
2. **参考跟踪项**：anchor 位置/旋转、身体相对位置/旋转、线速度/角速度、关节位置/速度，来源于 `MotionCommand`。
3. **任务语义项**：目标距离、手与物体相对位姿、接触持续时间、抬升高度、搬运路径、目标放置误差。
4. **安全约束和终止项**：跌倒、危险姿态、越限、碰撞、物体脱落。约束优先于奖励，不通过增大奖励权重替代硬约束。

### 9.2 可交互 Reward Builder

- 首屏使用模板和滑块/开关；显示默认范围、单位和建议值。
- 实时展示每个 term 的平均贡献、标准差和占总回报比例。
- 支持保存为版本化模板，并将模板锁定到 Run Manifest。
- P0 不支持任意 Python；新增奖励必须以注册插件和 schema 形式发布。
- 任务奖励和模仿奖励分开显示，允许按训练阶段启用/退火。

### 9.3 箱子任务路线

首个操作任务按阶段拆分：

`GoToBox → AlignToBox → ReachBox → HugBox → LiftBox → HoldBox → TurnWithBox → WalkWithBox → PlaceBox → ReleaseAndStepBack`

MVP 只验收 `ReachBox` 和 `HugBox`，使用轻量箱体、明确碰撞体和固定目标位姿。完成基础稳定后再加入抬升、行走和放置，避免端到端从随机初始化直接学习完整搬运。

## 10. API 草案

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/projects` | 创建项目 |
| GET | `/api/robots` | 查询机器人适配器和能力 |
| GET | `/api/tasks` | 查询任务模板 |
| POST | `/api/assets` | 上传视频/动作/模型 |
| POST | `/api/motions/compile` | GVHMR/GMR 产物编译和校验 |
| GET | `/api/reward-templates` | 查询奖励模板 |
| POST | `/api/jobs` | 创建训练作业并生成 Run Manifest |
| GET | `/api/jobs/{id}` | 查询状态、配置和指标摘要 |
| GET | `/api/jobs/{id}/events` | SSE/WebSocket 日志和事件流 |
| POST | `/api/jobs/{id}/cancel` | 取消当前 attempt |
| POST | `/api/jobs/{id}/export` | 导出策略包 |
| POST | `/api/jobs/{id}/sim2sim` | 启动 MuJoCo 验收 |
| GET | `/api/artifacts/{id}` | 获取产物元数据和下载地址 |

所有接口返回 `request_id`、结构化错误码和资源版本；长任务接口只创建作业，不同步等待训练结束。

## 11. 验收标准

### 11.1 数据与重定向

- GVHMR 结果存在 `smpl_params_global`，关键数组 shape 正确、无 NaN/Inf。
- GMR 输出的四元数归一化误差小于 `1e-3`，关节名称与 `RobotSpec` 完整匹配。
- 接受的动作片段关节限位违反比例为 0；身体映射缺失时作业阻断并给出具体名称。
- 预览可以回放原视频和机器人重定向视频，并显示根轨迹、关节曲线和质量告警。

### 11.2 训练与导出

- 使用固定样例能通过 Isaac Lab 适配器注册的 G1 mimic task 启动、完成并产出 checkpoint；首个适配器可以映射到现有 `Unitree-G1-29dof-Mimic-*` 任务。
- `policy.pt`、`policy.onnx`、`deploy.yaml`、env/agent 配置和校验和均存在，维度、关节顺序、动作缩放一致。
- 失败作业能够定位到具体阶段、命令和退出码，重试不会覆盖原始 attempt。

### 11.3 sim2sim

- 至少运行 3 个随机种子，过程中无 NaN、进程崩溃或不可恢复穿模。
- G1 模仿样例在固定时长内无跌倒，躯干高度/姿态、动作饱和率和基座漂移低于任务阈值。
- 箱子任务除回报外必须报告：到达目标误差、手/箱相对位姿、箱体高度、脱落次数、接触异常和成功率。
- 报告同时保存 Isaac 与 MuJoCo 的关键曲线和视频，不能只展示单次截图。

## 12. 安全、合规与可运维性

- GVHMR 当前许可证限于教育、研究和非营利用途，并要求衍生修改开源；商业化或对外服务前必须取得许可或替换为有明确商用许可的模型。
- SMPL/SMPLX 权重、动作数据、机器人模型和厂商 SDK 分别登记许可证、来源和用途。
- 上传文件进行扩展名、MIME、大小、时长和路径隔离校验；禁止路径穿越和任意命令执行。
- Web 端不直接下发实机控制；导出页增加人工确认、适配器版本、控制频率和安全检查清单。
- 保存结构化日志、GPU/显存、阶段耗时、奖励分项、代码提交号和依赖锁定信息。
- 产物使用 SHA-256 校验；删除策略默认软删除并保留审计记录。

## 13. 主要风险与缓解措施

| 风险 | 影响 | 缓解措施 |
| --- | --- | --- |
| 单目视频遮挡、深度和接触不确定 | 动作重定向抖动、无法直接用于操作 | 先做动作质量评分；允许上传已有动作资源；关键帧/人体高度可人工修正 |
| GMR IK、坐标系、四元数约定不一致 | 训练姿态错误或 sim2sim 失真 | 在契约中声明 `xyzw/wxyz`、坐标系和关节顺序；编译前后做 round-trip 单测 |
| 机器人 DoF、身体名、执行器不同 | 任务无法复用 | 通过 `RobotSpec`/适配器校验覆盖率；先支持单一 G1，再增加第二个机器人 |
| 参考动作不是物理可行轨迹 | RL 学到抖动或跌倒 | 预处理限位、速度、接触和地面检查；使用课程学习和阶段训练 |
| 奖励冲突/奖励投机 | 回报高但任务失败 | 分项回报、约束终止、成功率和视频联合验收；保存每次 RewardConfig |
| Isaac 与 MuJoCo 模型或控制频率不一致 | sim2sim 失败 | 锁定 XML/USD、PD、dt、decimation、动作缩放；导出包携带完整参数 |
| 训练资源昂贵或不稳定 | 作业排队和成本失控 | GPU worker 队列、配额、断点恢复、早停和小规模 smoke test |
| 导出归一化/关节映射遗漏 | 实机策略行为异常 | `deploy.yaml` schema 校验、ONNX 输入输出维度检查、离线回放测试 |
| GVHMR 或模型许可证限制 | 无法商业交付 | 产品中展示许可证状态；商业版本设计可替换的人体估计接口 |

## 14. 迭代计划

### P0：契约和适配器（1～2 周）

- 固化 `RobotSpec`、`RetargetMotion`、`TrainMotionNPZ`、`RewardConfig`、`PolicyBundle` schema。
- 将 GVHMR、GMR、Motion Compiler、Isaac Lab train/play/export 和 MuJoCo sim2sim 包装成可重试 CLI adapter。
- 用一个 G1 mimic 样例完成本地端到端 smoke test。

### P1：后端闭环（2～3 周）

- FastAPI、作业状态机、文件隔离、日志流、产物索引。
- 训练 worker 能够创建 Run Manifest，通过 Isaac Lab 适配器启动 G1 mimic 任务并导出策略。

### P2：React 工作台（2～3 周）

- 项目、上传、向导、动作预览、训练监控和产物详情。
- Reward Builder 模板、字段校验和实时指标展示。

### P3：操作任务与验收（3～4 周）

- `ReachBox`/`HugBox` 分阶段任务、箱体场景参数和失败诊断。
- MuJoCo 三种子验收、报告和下载包。

### P4：第二机器人适配器（另行评估）

- 仅选择已经公开 MuJoCo/URDF/USD、关节限制和控制接口的机器人。
- 通过同一契约完成动作重定向、训练和 sim2sim，不复制 G1 特有代码。

## 15. 待确认决策

1. GPU worker 是部署在本地工作站、局域网服务器还是容器集群。
2. MVP 是否只允许 G1 29 DoF，是否同时纳入 G1 灵巧手版本。
3. 前端采用独立 Vite SPA，还是并入现有后端的统一部署包。
4. 箱子目标位姿由表单输入、场景文件输入，还是另接视觉检测模块。
5. 商业版是否替换 GVHMR，或先取得其商业授权。
6. sim2sim 通过阈值（跌倒、漂移、姿态、任务成功率）由项目统一设定，还是按任务模板设定。

## 16. 参考实现与文档

- `GMR/README.md`
- `GMR/scripts/gvhmr_to_robot.py`
- `GMR/general_motion_retargeting/motion_retarget.py`
- `GMR/general_motion_retargeting/utils/smpl.py`
- `GVHMR/README.md`
- `GVHMR/LICENSE`
- `unitree_rl_lab/README.md`
- `unitree_rl_lab/scripts/rsl_rl/train.py`
- `unitree_rl_lab/scripts/rsl_rl/play.py`
- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/commands.py`
- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/walk_forward/tracking_env_cfg.py`
- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py`
- `G1箱子A到B搬运任务训练路线.md`
- `宇树G1 Sim2Sim训练技术方案.md`

## 17. 仓库与版本锁定

### 17.1 已确认的运行基线

以下版本以导师服务器的实际环境和当前工作区为准。服务器信息来自 `unitree_g1_train` 环境的包查询；不能用 Python 包内部版本替代 Git release 版本。

| 组件 | GitHub 仓库 | 版本/提交 | 状态 |
| --- | --- | --- | --- |
| Isaac Lab | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | Git `v2.3.0`，服务器 commit `3c6e67bb5`；包版本 `0.47.2` | 已确认 |
| Isaac Sim | [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) | pip 分发 `5.1.0.0` | 已确认版本，源码 commit 待采集 |
| Unitree RL Lab | [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | 服务器包 `0.2.1`；本地源码可见 `0.2.1-11-g4960b84` | 需核对服务器 Git SHA |
| Unitree MuJoCo | [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | 基线 commit `ae6a840`；当前工作区有未提交 overlay | 已确认基线，overlay 待纳入 manifest |
| MuJoCo | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | Unitree MuJoCo README 指向 `3.3.6` | 已确认文档基线 |
| GMR | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | README badge `0.2.0`；本地 commit `bb1bbe4` | 无统一 release tag，按 commit 锁定 |
| GVHMR | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | 本地 commit `6ec3ca3`；Python 3.10、Torch 2.3.0+cu121 依赖 | 无统一 release tag，独立环境运行 |

### 17.2 直接依赖和机器人资产仓库

- [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)：C++ DDS/SDK2 bridge。
- [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)：Python DDS/SDK2 bridge。
- [unitreerobotics/unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)：ROS 2 接口和 sim2real 示例。
- [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros)：URDF、网格和机器人描述；当前本地 commit `d6f13aa`。
- [kevinzakka/mink](https://github.com/kevinzakka/mink)：GMR 的 MuJoCo IK 依赖，版本需在 GMR 环境锁定文件中记录。
- [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)：可选的公开机器人 MJCF 资产，不替代厂商官方模型。

### 17.3 Web 平台基础仓库

这些仓库用于平台实现，不参与仿真版本匹配；在 P0 建立前端/后端仓库后，必须将实际 npm/Python 版本写入 lockfile 和镜像 digest。

- [facebook/react](https://github.com/facebook/react)：React UI。
- [vitejs/vite](https://github.com/vitejs/vite)：前端构建。
- [fastapi/fastapi](https://github.com/fastapi/fastapi)：Python API。
- [redis/redis](https://github.com/redis/redis)：可选的作业队列/事件中间件。
- [celery/celery](https://github.com/celery/celery)：可选的 GPU worker 编排。
- [pydantic/pydantic](https://github.com/pydantic/pydantic)：API 与 Run Manifest schema。

### 17.4 版本锁定规则

1. Isaac Lab 和 Isaac Sim 必须成对锁定，不能在 `main`、`develop` 和 release tag 之间混用。
2. 当前 MVP 使用 `IsaacLab v2.3.0 + IsaacSim 5.1.0.0 + Python 3.11`；新版本升级必须新建兼容性分支和 smoke test。
3. GMR、GVHMR、Unitree MuJoCo 均没有足够的统一 release tag，使用 Git SHA、依赖 lockfile 和模型 hash 三元组锁定。
4. `unitree_mujoco` 当前未提交的修改必须作为 `project_overlay` 记录，不能被误认为上游版本。
5. 每次 Run Manifest 保存所有仓库 URL、Git SHA、容器 digest、Python/CUDA/driver 版本和模型文件 SHA-256。

## 18. Unitree MuJoCo 与原生 MuJoCo 的边界

`unitree_mujoco` 是“原生 MuJoCo 引擎 + Unitree 应用层”，不是 MuJoCo 引擎 fork。它增加：

- Unitree G1、Go2、H1 等 MJCF、网格、惯量、关节限位和电机力矩范围。
- `LowCmd → PD/力矩 → mj_data.ctrl` 的电机控制映射。
- `mj_data.sensordata → LowState/IMU/SportModeState` 的状态映射。
- Unitree SDK2/DDS topic、定时线程、无线手柄和 Python bridge。
- 人形机器人虚拟弹簧挂带、地形工具、交互式 viewer 和 sim2real 示例。

原生 MuJoCo 仍负责模型解析、接触、约束、积分、传感器计算和 `mj_step`。因此平台必须把 Unitree MuJoCo 设计为 `UnitreeSimAdapter`，而不能把其 DDS topic、sensor 顺序、`m->nu > 20` 的机器人判断或弹簧挂带逻辑放进通用后端。

当前工作区还包含额外的本地 overlay：软件摇杆、弹簧力渐变释放、debug 输出、G1 摩擦/质量敏感性场景。这些修改会影响 sim2sim 结果，必须在验收报告中标注“原版/overlay”状态。

## 19. 本版 PRD 补充的遗漏与缺陷

### 19.1 已补齐

- 增加了仓库链接、版本矩阵和服务器实测基线。
- 明确了 Isaac Lab/Isaac Sim 与 Unitree RL Lab 的关系：后者只提供工程组织参考。
- 明确了 Unitree MuJoCo 与原生 MuJoCo 的边界，以及 `UnitreeSimAdapter` 的责任范围。
- 增加了上游 commit、未提交 overlay、模型 hash 和容器 digest 的复现要求。
- 增加了 GMR/GVHMR 独立环境要求，避免把 GVHMR 的 Torch 2.3/CUDA 12.1 依赖装入 Isaac Lab 环境。

### 19.2 仍需在 P0 关闭

1. 服务器上尚未采集 Unitree RL Lab 的 Git SHA，`0.2.1` 包版本不足以证明源码一致。
2. 服务器使用 Isaac Sim pip 包，尚未取得对应 Isaac Sim 源码 commit；源码开发和运行包的对应关系需要登记。
3. 当前 `unitree_mujoco` 有未提交改动，缺少独立 overlay commit 和变更清单。
4. GMR 与 GVHMR 的 Python/CUDA 依赖没有统一 lockfile；应使用独立环境或容器，不与 Isaac Lab 共用环境。
5. 还没有验证 Isaac Lab 动作输出、Unitree MuJoCo 的 XML actuator/sensor 顺序和真实 G1 部署配置的逐项一致性。
6. API 中的视频对象追踪、箱体 6D 位姿和操作任务状态没有具体感知方案；MVP 必须由表单或场景文件提供箱体状态。
7. 尚未定义用户、项目、GPU worker 和产物权限模型；在多用户服务器上不能默认所有作业共享文件目录。
8. 尚未定义训练取消、GPU OOM、Isaac Sim 崩溃和 MuJoCo 超时后的清理及重试策略。

## 20. P0 必须新增的验证项

- **版本 smoke test**：固定 `IsaacLab v2.3.0 + IsaacSim 5.1.0.0`，启动一个空场景、一个 G1 mimic 环境和一个导出流程。
- **动作契约测试**：验证 `root_rot` 的 `xyzw/wxyz` 转换、G1 29 DoF 关节顺序、帧率和 `.npz` 所有键的 shape。
- **控制映射测试**：给每个关节发送单位位置、速度和力矩命令，检查 Unitree MuJoCo 的 `ctrl`、传感器和 DDS 状态是否对应。
- **sim2sim 对齐测试**：固定初始状态和动作序列，对比 Isaac Sim 与 MuJoCo 的根高度、姿态、关节位置、接触和力矩曲线。
- **overlay 可复现测试**：原版 Unitree MuJoCo 和当前 overlay 各运行一次，报告弹簧挂带、摩擦、质量和控制参数差异。
- **部署包测试**：检查 `policy.onnx`/`policy.pt`、`deploy.yaml`、关节映射、动作缩放、观测归一化和控制周期是否完整。
- **失败恢复测试**：人为终止训练、删除 checkpoint、制造 GPU OOM 和超时，确认作业状态与临时目录不会污染下一次运行。

## 21. 本地开发与服务器运行边界

### 本机

- 保存 React、FastAPI、schema、适配器、配置生成和日志解析代码。
- 保存各上游仓库的 source checkout、tags 和版本 manifest。
- 可以运行 GMR/GVHMR 的非仿真单元测试，但不把 Windows 作为完整训练验收环境。

### GPU 服务器

- 使用 `/ai/users/huangwy/G1/IsaacLab` 和 `unitree_g1_train` 环境作为当前基线。
- Isaac Lab/Isaac Sim 训练与 MuJoCo sim2sim 分别由 worker 调用。
- 双 RTX 4090 初期先单卡 smoke test，再验证多 GPU；两张卡不是统一显存池。
- GMR/GVHMR 使用独立 Python 环境，避免与 Isaac Lab 的 Torch/CUDA 依赖冲突。

## 22. 新版 PRD 的决策结论

MVP 不再追求“所有机器人共用一份训练代码”或“所有阶段使用同一个 Python 环境”。稳定的通用范式应是：

```text
统一数据契约
  + 机器人适配器
  + Isaac Lab 任务/奖励模板
  + Isaac Sim 训练运行时
  + Unitree/厂商 sim2sim 适配器
  + 版本与产物 manifest
```

这样可以复用任务描述、奖励 schema、前端流程和验收报告，同时允许不同机器人拥有不同的 XML/USD、关节映射、控制周期、动作缩放和部署协议。
