# 通用关节机器人 RL 训练平台完整闭环技术方案

| 项目 | 决策 |
| --- | --- |
| 文档状态 | 开发执行基线 |
| 基础范式 | 面向具有关节的机器人；G1 是首个完整验证实例，不是范式本身 |
| 首期范围 | Unitree G1 29 DoF，通用人体动作模仿训练 |
| 首期终点 | Isaac Lab 训练、策略双格式导出、MuJoCo sim2sim 三种子验收 |
| 本地运行形态 | 完整项目部署到用户本地；本机浏览器访问 `localhost`，不依赖云端 SaaS |
| Windows 运行基线 | Windows 11 + WSL2 + Ubuntu 22.04 + Docker Desktop/WSL2 集成 + NVIDIA WSL CUDA |
| Linux 运行基线 | Ubuntu 22.04 + Docker Engine + NVIDIA Container Toolkit；具备与 Windows 相同的训练能力 |
| 主要技术栈 | `robotlab` CLI；React + TypeScript + Vite；FastAPI + Pydantic v2；本地文件存储/进程调度（默认），PostgreSQL + Redis + MinIO（可选 Compose 扩展） |
| 训练基线 | Isaac Lab v2.3.0 + Isaac Sim 5.1.0.0 + RSL-RL/PPO |
| 适配器基线 | 通用关节机器人 RobotSpec；G1 适配器、GMR/GVHMR 与 Unitree MuJoCo 为首个实例 |
| 后续方向 | 多机器人适配器、用户自定义任务语义、浏览器 MuJoCo WASM/WebGL viewer |

本文件是对以下资料和本次需求确认的工程化收敛，不替代原 PRD 中的背景说明：

- [通用人形机器人RL训练平台开发PRD-v0.3.md](./通用人形机器人RL训练平台开发PRD-v0.3.md)
- [通用人形机器人RL训练平台开发PRD.md](./通用人形机器人RL训练平台开发PRD.md)
- [Web平台所需GitHub仓库与版本锁定清单.md](./Web平台所需GitHub仓库与版本锁定清单.md)
- [人形机器人RL平台本次对话总结.md](./人形机器人RL平台本次对话总结.md)

## 0. 本次方案的强制决策

以下内容是本方案的执行约束，不再作为普通实现偏好处理：

1. 先完成 G1 的完整闭环，再接入第二个机器人。首期不为了箱体搬运或某一个动作阶段裁剪通用动作模仿能力。
2. “任意人体动作”指任意满足平台输入契约和质量门槛的人体动作。平台必须在训练前给出可定位的拒绝原因，不承诺对损坏、不可识别或严重遮挡的输入强行训练。
3. 首期支持两条输入路径：
   - 视频：`视频 → GVHMR → GMR → Motion Compiler → G1 训练动作`。
   - 直接动作：`.pt/.npz/.csv/.pkl → 格式识别/转换 → G1 关节轨迹或人体姿态识别 → 必要的 GMR/适配 → G1 训练动作`。
4. 直接动作首期必须支持 G1 关节轨迹和人体姿态两类；其他机器人到 G1 的跨机器人映射只保留接口和测试占位，不作为首期验收项。
5. 用户可以完整调整 PPO、观测、奖励和控制参数，但只能使用平台注册的参数 schema、奖励项和安全终止项，不得上传 Python、shell 或其他可执行代码。
6. 训练基线固定为 Isaac Lab + RSL-RL/PPO。`unitree_rl_lab` 只参考任务注册、配置分层、train/play/export/deploy 的组织方式，不作为平台内核。
7. 策略包必须同时包含 TorchScript/JIT 和 ONNX，并绑定归一化参数、动作缩放、PD/控制参数、环境配置、版本 manifest、校验和和 sim2sim 报告。
8. 双 RTX 4090 支持并行训练。调度器采用资源声明和动态装箱，首期每张卡默认最多 3 个训练作业；显存、利用率或 CPU 达到阈值时停止接纳新作业。
9. 首期预览仍采用后端 MuJoCo 离屏渲染 PNG。训练和 sim2sim 稳定后再增加浏览器 MuJoCo WASM/WebGL viewer。
10. 首期不包含真实 G1 上机、DDS 实机控制和安全联调，最终验收终点是 sim2sim。
11. 首期面向单个团队内部使用，默认本地单用户且不要求登录；数据模型、GPU lease、审计、队列和 API 保留未来多用户与负载均衡扩展点。
12. `RobotSpec` 必须与具体厂家、型号和 DoF 数量解耦。G1 只能作为 `RobotSpec` 的一个实例，通过适配器注册进入实验。
13. 机器人资产、关节参数、执行器参数和传动关系由用户录入；平台负责 schema 校验、资产 staging、版本化、派生后端配置和能力自检，不从 URDF/MJCF 静默猜测关键控制参数。
14. 首期基础范式支持主流关节绑定方式：独立关节执行器、同一执行器驱动多个关节、多个执行器驱动一个关节、差动/并联传动、腱绳/连杆等效传动、mimic 关节和串联弹性执行器的等效阻抗模型。厂商 CAN/EtherCAT/DDS 通信协议不属于首期范围。
15. 交付物是完整的本地项目，不是必须连接云端的远程 Web 平台；用户通过封装好的 `robotlab` 命令完成安装、初始化、检查、启动、运行、停止和产物导出。
16. Windows 和 Linux 都必须具备完整训练能力。Windows 的 Linux 运行时统一放在 WSL2 Ubuntu 22.04 内；Linux 使用同一套 Docker Compose 服务边界和契约。
17. 首期默认提供单用户 `Local File Mode`：不依赖 Docker、PostgreSQL、Redis 或 MinIO；同时保留 `Compose Mode` 作为团队共享、远程 GPU 和多用户扩展。两种模式共享 API、契约和产物格式，不能把 Local File Mode 实现成仅能跑通的演示版。
18. 首期默认本地单用户运行，不要求登录；单个用户可以提交多个训练/导出/sim2sim 作业，由本地调度器按实时 GPU 负载并发装箱或排队；项目、动作、配置、日志、checkpoint、策略包和报告保存在本地数据目录，同时保留远程 GPU、多用户和负载均衡扩展点。
19. 机器人资产必须由用户提供并通过 `robotlab robot add --path ...` 注册；平台不替用户下载或假定厂商资产，只负责校验、版本化、staging 和配置生成。

## 1. 目标、边界与成功定义

### 1.1 首期目标

用户在本地执行 `robotlab start`，通过浏览器访问 `localhost`，创建项目并选择已注册机器人（首期为 G1），提供视频或动作文件，完成动作转换和质量检查，调整动作及完整训练参数，启动本地异步 RL 作业，查看日志和指标，导出策略，并在 Unitree MuJoCo 中用至少三个随机种子完成 sim2sim。所有输入、配置、命令、版本、指标和产物可追溯、可复现、可下载。

首期不限制动作内容。动作可以是站立、挥手、深蹲、行走、转身或其他人体动作，只要人体估计/姿态文件和目标 G1 运动学约束满足契约。箱体搬运保留为后续任务语义示例，不作为通用 imitation pipeline 的硬编码分支。

### 1.2 非目标

- 不在 API 进程导入 Isaac Sim、加载 CUDA 仿真库或执行训练。
- 不承诺从单目视频稳定恢复物体 6D 位姿。
- 不允许用户上传任意 reward、环境或 shell 代码。
- 不承诺策略权重跨机器人直接复用。
- 不在首期支持浏览器内权威物理仿真或真实机器人控制。
- 不在首期完成其他机器人到 G1 的跨机器人动作映射。

### 1.3 成功定义

一个 Run 只有同时满足以下条件才可以标记为 `READY_TO_DOWNLOAD`：

1. 输入资源通过格式、许可证、数值、运动学和 G1 关节契约检查。
2. Motion Compiler 生成可被 Isaac Lab MotionLoader 读取的 `TrainMotionNPZ`。
3. Isaac Lab 训练、play 和导出阶段成功，产物 hash 完整。
4. JIT 和 ONNX 都能在独立的 CPU 推理检查中输出正确形状的有限值。
5. MuJoCo sim2sim 至少完成 3 个固定随机种子，满足第 15 节默认阈值。
6. Run Manifest、环境版本、配置、日志、视频、指标、报告和校验和均已登记。

## 2. 端到端业务流程

```text
robotlab start → 浏览器访问 localhost → 创建本地项目
  → 选择 Unitree G1 29 DoF
  → 上传视频或动作文件
  → 对象存储完成 + SHA-256/MIME/许可证校验
  → 输入类型识别
       ├─ 视频：GVHMR → GMR
       └─ 直接动作：格式转换 → G1轨迹或人体姿态分支
  → Motion Compiler：坐标、四元数、帧率、速度、限位、接触质量检查
  → 后端 MuJoCo PNG 预览、时间轴、关节编辑、关键帧和质量告警
  → 保存 MotionEditConfig，重新编译为新版本
  → 用户选择完整 PPO/观测/奖励/控制配置
  → 本地 API 校验所有配置并冻结 Run Manifest
  → 动态 GPU 调度
  → Isaac Lab + Isaac Sim + RSL-RL/PPO 训练
  → play、export：JIT + ONNX + deploy/env/agent 参数
  → MuJoCo sim2sim 三种子
  → 失败分类和回溯，或通过后生成策略包
```

### 2.1 两条输入模式

#### 模式 A：视频输入

输入至少包括一个视频对象和人体估计配置。worker 必须执行：

1. 检查视频容器、编码、帧率、时长、分辨率、音视频同步和人体可见度元数据。
2. 在独立 GVHMR 环境执行 `demo.py`，产生 `hmr4d_results.pt`。
3. 检查 `smpl_params_global.body_pose`、`betas`、`global_orient`、`transl` 等字段、shape、finite 值和 checkpoint 身份。
4. 在独立 GMR 环境加载 G1 XML、Mink 和身体映射，输出 `RetargetMotion`。
5. 统一 `xyzw/wxyz`、坐标轴、关节顺序和 fps，生成 `TrainMotionNPZ`。

#### 模式 B：直接动作输入

首期识别 `.pt`、`.npz`、`.csv`、`.pkl`。识别器不得只依据扩展名判断，必须读取文件头、对象类型、字段名和数组 shape，并输出 `SourceMotionDescriptor`。

支持两类闭环：

- G1 关节轨迹：识别 `joint_pos/qpos/dof_pos` 等字段，匹配 G1 29 个关节名或由用户提供显式映射，补齐根位姿、速度和身体状态后进入 Motion Compiler。
- 人体姿态轨迹：识别 SMPL/SMPL-X 关节位置、旋转或等价人体骨架字段；不经过 GVHMR，但仍调用 GMR 的人体到 G1 IK 重定向，再进入 Motion Compiler。

不满足上述两类的直接文件必须进入 `UNSUPPORTED_SOURCE_TYPE` 或 `SCHEMA_INVALID`，不能静默当成 G1 qpos。未来跨机器人映射器使用同一 `SourceMotionDescriptor` 和 `RobotAdapter` 端口扩展。

### 2.2 “任意动作”的输入契约

平台对动作内容不设语义白名单，但对可处理性设门槛。首期默认门槛如下，项目配置可以收紧，不可通过普通用户配置关闭安全硬约束：

| 项目 | 默认要求 |
| --- | --- |
| 视频格式 | MP4/MOV/MKV，H.264/H.265，单文件 ≤ 2 GB |
| 视频时长 | 0.5 秒至 120 秒；超过时长先切片或拒绝 |
| 帧率 | 15–120 FPS；最终重采样到任务目标 FPS |
| 分辨率 | 短边 ≥ 360 px；人体主体在至少 90% 帧可见 |
| 人体数量 | 默认单主体；多主体必须在上传元数据中指定目标主体 |
| 直接动作 | 文件 ≤ 2 GB；数组 dtype 必须为 float32/float64/int16/int32 可转换类型 |
| 数值 | 不允许 NaN/Inf；四元数范数误差默认 ≤ 1e-3 |
| 轨迹时长 | 0.5 秒至 120 秒；少于 15 帧拒绝 |
| G1 关节 | 29 DoF，名称或显式映射完整覆盖；缺失关节不得静默填 0 |
| 许可证 | 必须记录来源、用途和允许的处理范围 |

门槛失败返回稳定错误码、失败字段、实际值、建议修复和对应 stage。平台必须保留原始对象，失败的派生对象不能标记为可训练。

## 3. 总体架构

### 3.1 逻辑架构

```text
React SPA
  ├─ REST /api/v1
  └─ SSE /api/v1/runs/{id}/events
        │
        ▼
FastAPI API（不加载 Isaac Sim，本地文件或 Compose 存储后端）
  ├─ Local Access / Project / Asset / Robot / Motion / Reward / Run / Artifact API
  ├─ Application Services
  ├─ Domain Contracts / State Machine / Policies
  └─ Storage/Queue profile
       ├─ Local File Mode：文件仓库 + 本地 scheduler
       └─ Compose Mode：PostgreSQL 16 + Redis 7 + MinIO(S3) + Celery
        │
        ▼
Outbox Dispatcher → Celery/Redis 分队列
  ├─ asset-io
  ├─ motion-cpu
  ├─ gvhmr-gpu
  ├─ isaac-gpu
  ├─ sim2sim-gpu
  ├─ report-cpu
  └─ maintenance
        │
        ▼
本地 Docker worker（WSL2/Linux）
  ├─ GVHMR Python 3.10 / Torch 2.3.0+cu121
  ├─ GMR Python 3.10 / Mink / MuJoCo
  ├─ Isaac Lab v2.3.0 / Isaac Sim 5.1.0.0 / Python 3.11
  └─ Unitree MuJoCo / SDK2 / MuJoCo 3.3.6
```

### 3.2 本地运行边界

本地项目启动后，API、前端、GPU worker、训练、导出和 sim2sim 均属于同一个用户工作区。默认 `Local File Mode` 不启动外部数据库和消息队列，由本地 scheduler 编排进程并把事实写入 `runtime/`；`Compose Mode` 才启动 PostgreSQL、Redis、MinIO 和 Celery。两种模式都由本地浏览器访问 localhost，不依赖云端服务。

Windows 运行时要求：

- Windows 11 安装 WSL2、Ubuntu 22.04、Docker Desktop/WSL2 集成和 NVIDIA WSL CUDA 支持；
- 所有 Linux 容器、Isaac Lab、MuJoCo、训练和 sim2sim 任务统一在 WSL2 Ubuntu 22.04 内执行；
- Windows 主机只作为 CLI 入口和浏览器宿主，代码挂载、GPU 映射、模型缓存和训练数据必须经 WSL2/Compose 路径验证；
- Windows WSL2 必须执行与 Linux 相同的训练、导出和 sim2sim 测试，不把 Windows 视为功能缩水的开发模式。

Linux 运行时要求：

- Ubuntu 22.04、NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit；
- 使用与 WSL2 相同的 Compose 文件、环境变量契约、worker 镜像和数据目录结构；
- Linux 主机直接提供 GPU 训练能力，不依赖额外远程 GPU 服务器。

允许联网下载依赖和模型，但所有下载内容必须登记来源 URL、版本或 revision、许可证、文件大小和 SHA-256。未经登记的浮动缓存不能直接进入训练或发布。

未来接入 GPU 服务器或团队共享集群时，只增加远程 worker/调度部署，不修改本地项目的训练、产物和 manifest 契约。

### 3.3 运行目录和对象流

每个 attempt 使用独立目录：

```text
runs/{project_id}/{run_id}/{attempt_id}/
  manifest/manifest.json
  input/                # Local File 为本地只读副本，Compose 为对象存储下载副本
  work/                 # 当前阶段临时文件
  outputs/              # 阶段输出，提交前可校验
  logs/                 # stdout/stderr 和结构化日志
  metrics/              # JSONL/CSV/曲线中间文件
  reports/
```

阶段完成后先写临时文件或临时对象 key，再计算 SHA-256，最后以原子文件提交（Local File Mode）或 PostgreSQL 事务登记（Compose Mode）。API 不将大对象作为响应体中转。

## 4. 目标工作区和模块分工

### 4.1 目录树

```text
AllRobotRLLLab/
├─ apps/
│  ├─ web/                         # React/Vite 生产前端
│  └─ api/                         # FastAPI 启动入口
├─ packages/
│  ├─ contracts/                   # JSON Schema、Pydantic、TS 类型生成
│  ├─ frontend-ui/                 # 无业务通用组件和设计 token
│  └─ motion-formats/              # 前端展示所需的格式描述，不执行转换
├─ backend/
│  ├─ app/
│  │  ├─ api/                      # router、DTO、依赖注入
│  │  ├─ application/              # 用例、事务、命令、查询
│  │  ├─ domain/                   # 实体、值对象、端口、状态机、策略
│  │  ├─ adapters/                 # 通用 RobotSpec、机器人实例及 GVHMR/GMR/Isaac/MuJoCo 实现
│  │  ├─ infrastructure/           # DB、Redis、S3、subprocess、GPU 采集
│  │  ├─ workers/                  # Celery task 和阶段 runner
│  │  └─ config/                   # 配置和运行时版本
│  └─ tests/
│     ├─ unit/
│     ├─ integration/
│     ├─ contract/
│     └─ fixtures/
├─ adapters/
│  ├─ generic_articulated/       # 通用 RobotSpec、传动模型和后端生成端口
│  └─ unitree_g1_29dof/          # G1 的 RobotSpec 实例和厂商 sim2sim 适配器
│     ├─ robot_spec.json
│     ├─ assets.lock.json
│     ├─ gmr_mapping.yaml
│     ├─ isaac_tasks/
│     ├─ mujoco/
│     ├─ deploy_schema.json
│     └─ tests/
├─ workers/
│  ├─ motion-image/                # GMR/GVHMR/Motion Compiler 镜像定义
│  ├─ isaac-image/                 # Isaac Lab/Isaac Sim 镜像或启动脚本
│  └─ sim2sim-image/               # Unitree MuJoCo/SDK2 镜像
├─ cli/                             # robotlab install/init/doctor/start/stop/run 等命令
├─ infra/
│  ├─ compose/                     # 本地和平台服务 compose
│  ├─ gpu-server/                  # GPU worker、systemd、健康检查
│  ├─ migrations/
│  └─ monitoring/
├─ schemas/                        # 发布后的 JSON Schema 快照
├─ runtime/                        # 用户本地运行数据，不提交到源码仓库
├─ assets/                         # 用户注册的机器人资产索引，不复制原始资产
├─ projects/                       # 项目、动作、配置、日志和产物索引
├─ docs/
├─ third_party/                    # 只读上游源码和许可证
└─ frontend-prototype/             # 当前原型，迁移完成前只修 bug，不扩展生产业务
```

迁移原则：现有 `frontend-prototype/react-app` 的 UI 和 `mujoco_service.py` 是可复用原型；生产实现必须拆分为 API router、Motion service、MuJoCo adapter、Artifact service 和前端 feature，不能把原型文件直接作为最终边界。

### 4.2 前后端职责边界

| 工作区 | 负责 | 禁止负责 |
| --- | --- | --- |
| `apps/web` | 页面、表单、预览交互、缓存、SSE 展示、错误可视化 | 训练命令、权威 IK、奖励计算、直接读数据库 |
| `packages/contracts` | 版本化 schema、错误码、枚举、生成类型 | 数据库连接、业务副作用 |
| `backend/api` | HTTP/SSE、本地访问控制、可选项目权限、DTO 转换 | 长任务、Isaac import、复杂动作算法 |
| `backend/application` | 编排用例、事务、幂等、事件 | 具体 GPU 命令、SQL 细节 |
| `backend/domain` | 状态机、不变量、值对象、端口 | FastAPI、Celery、SQLAlchemy、仿真 SDK |
| `backend/adapters` | G1、GVHMR、GMR、Isaac、MuJoCo 具体实现 | 用户权限和页面逻辑 |
| `backend/infrastructure` | DB/Redis/S3/subprocess/GPU 资源 | 任务语义和奖励公式 |
| `backend/workers` | 阶段执行、日志、取消、产物登记 | 接收任意用户代码 |
| `adapters/generic_articulated` | 通用 RobotSpec、关节/执行器/传动模型和后端生成端口 | 厂商专有协议和页面逻辑 |
| `adapters/unitree_g1_29dof` | G1 的资产、映射、任务注册、控制和验收规则 | 修改通用 API 或 domain |

### 4.3 依赖方向

```text
api → application → domain
workers → application → domain
adapters/infrastructure → domain ports
frontend features → contracts/entities/shared
```

禁止反向依赖。领域层中出现 `FastAPI`、`Celery`、`sqlalchemy`、`mujoco`、`omni.isaac` 或具体机器人 SDK 即视为架构违规。

## 5. 前端执行方案

### 5.1 技术栈和状态

- React + TypeScript + Vite + React Router；`strict: true`。
- TanStack Query 管理服务端状态、缓存、失效和重试。
- Zustand 只管理向导草稿、预览相机和编辑器临时状态，不能作为事实源。
- React Hook Form + Zod 管理表单；schema 从 `packages/contracts` 生成或通过 contract test 对齐。
- 当前首期预览使用后端 PNG；前端保留 `RendererProvider` 接口，后续接入 WASM/WebGL 不改页面业务协议。

### 5.2 页面和 feature

```text
/projects                       项目列表和项目成员
/projects/:id/assets            视频/动作资源、上传和版本
/projects/:id/robots            已注册机器人适配器能力与自检
/projects/:id/motion/:id        3D动作预览、时间轴、关节编辑、质量告警
/projects/:id/reward/:id        注册奖励和参数表单
/projects/:id/runs/:id          训练日志、指标、GPU、视频和阶段状态
/projects/:id/sim2sim/:id      三种子报告、曲线、视频和失败诊断
/artifacts/:id                  策略包内容、hash、许可证和下载
```

每个 feature 至少包含：`api.ts`、`queries.ts`、`mutations.ts`、`schemas.ts`、`components/`、`routes/`、`tests/`。页面只组合 feature 组件，不直接调用 `fetch`。

### 5.3 关键交互要求

- 上传显示分片进度、hash 校验、许可证声明和失败原因。
- 动作编辑的任何修改都显示“未提交草稿”，保存时创建新的 `MotionEditConfig` 版本。
- Reward Builder 根据后端注册 schema 动态生成表单，禁用安全终止项关闭操作。
- 训练启动前展示即将冻结的 manifest 摘要，要求用户确认。
- SSE 断开后按 `last_event_id` 恢复，不重复显示已有日志。
- 所有异步页面有 loading、empty、error、retry、权限拒绝和资源已删除状态。

## 6. 后端模块与接口

### 6.1 Domain 模块

| 模块 | 不变量 |
| --- | --- |
| Project | 本地单用户默认拥有本地项目；未来多用户模式下成员只能访问授权项目；项目软删除不影响已发布产物 |
| Asset | 逻辑资源可有多个不可变版本；对象 hash 唯一登记 |
| Robot | 适配器必须通过资产、版本、关节和能力自检 |
| Motion | 坐标、四元数、fps、关节顺序必须显式；非法数据不可训练 |
| Reward | 只能引用注册 term；安全终止不能被普通用户关闭 |
| Training | manifest 冻结后只读；配置改变必须创建新 Run 或新 attempt |
| Job | 状态只能按白名单迁移；重试不覆盖历史 attempt |
| Sim2Sim | 每个 seed 单独记录指标、日志、视频和版本 |
| Artifact | 每个对象有 kind、hash、大小、权限和来源 attempt |

### 6.2 适配器接口

```python
class MotionSourceAdapter(Protocol):
    def detect(self, path: Path) -> SourceMotionDescriptor: ...
    def validate(self, descriptor: SourceMotionDescriptor) -> ValidationResult: ...
    def convert(self, descriptor: SourceMotionDescriptor, output_dir: Path) -> ConversionResult: ...

class RobotAdapter(Protocol):
    def get_spec(self) -> RobotSpec: ...
    def validate_assets(self) -> ValidationResult: ...
    def compile_backend_configs(self, spec: RobotSpec, out_dir: Path) -> CompilationResult: ...
    def validate_motion(self, motion: RetargetMotion) -> ValidationResult: ...
    def compile_motion(self, motion: RetargetMotion, config: TrainingConfig, output_dir: Path) -> TrainMotionResult: ...
    def validate_training_manifest(self, manifest: RunManifest) -> ValidationResult: ...

class TrainingBackendAdapter(Protocol):
    def validate_config(self, manifest: RunManifest) -> ValidationResult: ...
    def train(self, manifest: RunManifest, output_dir: Path) -> ExecutionResult: ...
    def play(self, checkpoint: Path, manifest: RunManifest, output_dir: Path) -> ExecutionResult: ...
    def export(self, checkpoint: Path, manifest: RunManifest, output_dir: Path) -> ExecutionResult: ...

class Sim2SimAdapter(Protocol):
    def validate_bundle(self, bundle: PolicyBundle, manifest: RunManifest) -> ValidationResult: ...
    def evaluate(self, bundle: PolicyBundle, manifest: RunManifest, seed: int, output_dir: Path) -> EvaluationResult: ...
    def build_report(self, evaluations: list[EvaluationResult], output_dir: Path) -> Sim2SimReport: ...
```

所有接口返回结构化结果，不直接抛出供应商异常到 API。供应商异常在 adapter 边界转换为稳定错误码和诊断 payload。

## 7. 核心数据契约

所有契约必须带 `schema_version`/`format_version`，数组必须声明 dtype、shape、坐标系和来源。大数组放对象存储，JSON 只登记 URI 和摘要。

### 7.1 通用 `RobotSpec`（G1 为实例）

`RobotSpec` 是平台唯一的机器人适配事实源，设计目标是覆盖具有关节的主流机器人，而不是复刻 G1 配置。任何厂家、型号和 DoF 数量都必须通过同一份契约描述，再由后端生成 Isaac、Motion/GMR 和 MuJoCo 所需的具体配置。G1 的 29 DoF、关节名称和 Unitree 控制参数只存在于 `adapters/unitree_g1_29dof` 实例中。

用户录入的字段与平台派生字段必须分开保存。用户录入资产 URI、关节/执行器/传动参数、控制周期和初始状态；平台派生后端文件、动作维度、归一化参数、qpos 地址、检查报告和 manifest hash。关键控制参数缺失时必须失败，不能依据 URDF/MJCF 猜测后静默继续。

下面的 JSON 仅展示 G1 适配器如何实例化通用契约；正式 schema 不得把 `dof=29`、Unitree 命名或某一种控制模式设为全局固定值。

```json
{
  "schema_version": "robot_spec.v1",
  "robot_id": "unitree_g1_29dof",
  "vendor": "Unitree",
  "model": "G1",
  "model_version": "g1_29dof",
  "adapter_version": "unitree_g1_adapter.v1",
  "assets": {
    "mujoco_xml_uri": "...",
    "urdf_uri": "...",
    "isaac_usd_uri": "...",
    "asset_sha256": "..."
  },
  "joint_names": ["left_hip_pitch_joint", "..."],
  "dof": 29,
  "body_names": ["pelvis", "torso_link", "left_hand", "right_hand"],
  "actuation": {
    "mode": "position_pd",
    "control_dt": 0.02,
    "policy_dt": 0.02,
    "action_scale": 0.25,
    "kp_uri": "...",
    "kd_uri": "...",
    "torque_limit_uri": "..."
  },
  "limits": {
    "position_uri": "...",
    "velocity_uri": "...",
    "torque_uri": "..."
  },
  "capabilities": ["mimic"],
  "gmr_mapping_version": "...",
  "isaac_task_ids": ["g1_mimic"],
  "sim2sim_adapter": "unitree_g1_mujoco",
  "license": {"status": "research_only", "source_uri": "..."}
}
```

G1 首期必须提供 29 个关节的完整顺序、qpos 地址、body 名称、限位、PD 和动作缩放自动化检查。任何 Isaac 与 MuJoCo 顺序不一致都必须在适配器校验阶段失败。

### 7.1.1 关节、执行器和传动契约

每个关节必须有稳定的逻辑 ID，并同时声明策略顺序、仿真名称和部署名称。执行器参数优先使用关节侧 SI 单位；如用户提供电机侧参数，必须显式标注参考侧和换算方向。

```yaml
joints:
  - id: left_hip_pitch
    sim_name: left_hip_pitch_joint
    deployment_name: left_hip_pitch
    policy_index: 0
    type: revolute                 # revolute | prismatic | fixed | mimic
    axis: [0, 1, 0]
    limits:
      position: [-2.5, 2.0]       # rad 或 m
      velocity: 32.0               # rad/s 或 m/s
      effort: 88.0                 # N m 或 N
    home_position: -0.1
    direction: 1
    zero_offset: 0.0
    actuator_ids: [left_hip_pitch_motor]

actuators:
  - id: left_hip_pitch_motor
    model: implicit_pd             # implicit_pd | explicit_torque | velocity_pd | impedance
    command_space: position        # position | velocity | torque | mixed
    reference_side: joint
    gains: {kp: 100.0, kd: 2.0}
    limits: {effort: 88.0, velocity: 32.0}
    transmission:
      type: direct                  # direct | gear | differential | tendon | linkage | elastic
      gear_ratio: 1.0
      efficiency: 0.95
      direction: 1
    dynamics:
      armature: 0.01
      viscous_friction: 0.0
      coulomb_friction: 0.01
    action: {scale: 0.22, offset: 0.0}

transmissions:
  - id: waist_differential
    type: differential
    actuator_ids: [waist_motor_left, waist_motor_right]
    joint_ids: [waist_roll, waist_pitch]
    mapping_matrix: [[0.5, 0.5], [0.5, -0.5]]
```

必须支持的主流绑定方式包括：

- 一执行器对应一关节：直驱或减速器关节；
- 一执行器驱动多个关节：连杆、腱绳或共享传动；
- 多执行器驱动一关节：并联或冗余驱动；
- 差动/并联传动：使用显式映射矩阵；
- mimic 关节：使用主从关节约束；
- 串联弹性执行器：首期使用等效阻抗模型，保留弹簧刚度、阻尼和传感器延迟扩展字段。

平台不能把动作类型限制为某些预设动作。动作经过关节映射后进入仿真，执行器的力矩、速度和位置限制只作为物理模型和饱和诊断，不得静默删除输入帧。每次编译必须输出关节映射、传动矩阵、动作缩放、饱和比例和超限原因。

### 7.2 `SourceMotionDescriptor`

```json
{
  "schema_version": "source_motion_descriptor.v1",
  "asset_version_id": "uuid",
  "file_format": "npz",
  "detected_type": "g1_joint_trajectory",
  "source_skeleton": "unitree_g1_29dof",
  "fields": {
    "joint_pos": {"path": "joint_pos", "shape": [180, 29], "dtype": "float32"},
    "fps": {"path": "fps", "value": 30}
  },
  "joint_names": ["..."],
  "coord_frame": "world_z_up",
  "quaternion_convention": "xyzw",
  "license": {"status": "declared", "source": "user"},
  "detector_version": "motion-detector.v1"
}
```

识别器需要有注册表：`G1JointTrajectoryDetector`、`HumanPoseDetector`、`GVHMRResultDetector`。同一文件同时匹配多个 detector 时返回歧义错误并要求用户选择，不自动猜测。

### 7.3 `RetargetMotion`

```json
{
  "format_version": "retarget_motion.v1",
  "robot_id": "unitree_g1_29dof",
  "fps": 30.0,
  "frame_count": 180,
  "arrays": {
    "root_pos_uri": "...",
    "root_rot_uri": "...",
    "dof_pos_uri": "..."
  },
  "array_meta": {
    "root_pos": {"dtype": "float32", "shape": [180, 3]},
    "root_rot": {"dtype": "float32", "shape": [180, 4], "convention": "xyzw"},
    "dof_pos": {"dtype": "float32", "shape": [180, 29]}
  },
  "joint_names": ["..."],
  "coord_frame": "world_z_up",
  "source": {"type": "gvhmr|direct_human_pose|direct_g1", "asset_id": "..."},
  "quality": {
    "nan_count": 0,
    "quat_norm_max_error": 0.0001,
    "joint_limit_violation_ratio": 0.0,
    "foot_sliding_ratio": 0.02
  },
  "converter": {"name": "g1-motion-compiler", "version": "..."}
}
```

内部统一使用 `wxyz` 或 `xyzw` 的选择必须固定在代码库中；对外契约必须显式标注。当前 GMR 原始输出的根旋转为 `xyzw`，GMR IK 内部使用 `wxyz`，转换处必须有单元测试。

### 7.4 `TrainMotionNPZ`

最终文件必须兼容 Isaac Lab MotionLoader，至少包括：

```text
fps                 float
joint_pos           float32[N, 29]
joint_vel           float32[N, 29]
body_pos_w          float32[N, B, 3]
body_quat_w         float32[N, B, 4]
body_lin_vel_w      float32[N, B, 3]
body_ang_vel_w      float32[N, B, 3]
```

NPZ 同时写入 `joint_names`、`body_names`、`coord_frame`、`quat_convention`、`robot_id`、`source_motion_hash` 和 `compiler_version`。编译器验证 shape、finite 值、四元数归一化、帧率重采样、位置/速度/加速度/力矩限值、地面高度和身体覆盖率。

### 7.5 `MotionEditConfig`

```json
{
  "schema_version": "motion_edit.v1",
  "source_motion_version_id": "uuid",
  "robot_id": "unitree_g1_29dof",
  "global_transform": {
    "translation": [0.0, 0.0, 0.03],
    "yaw_offset": 0.05,
    "time_scale": 1.0
  },
  "joint_offsets": [
    {"joint_name": "left_shoulder_pitch_joint", "frame_start": 120, "frame_end": 240, "position_offset": 0.08}
  ],
  "ik_targets": [],
  "keyframes": [{"frame": 0, "qpos_uri": "..."}],
  "filters": {"smooth": true, "max_velocity_check": true}
}
```

编辑只产生配置版本，不修改原始资源。后端按 RobotSpec 限位重新计算，前端的临时 qpos 不能直接进入训练。

### 7.6 `RewardConfig` 与奖励注册表

平台注册的 term 必须声明 `id`、说明、单位、参数类型、默认值、最小/最大值、权重范围、适用机器人/任务、是否硬终止和实现版本。用户配置示例：

```json
{
  "schema_version": "reward_config.v1",
  "base_template": "g1_mimic_v1",
  "parent_config_id": null,
  "terms": [
    {"id": "tracking.joint_pos", "enabled": true, "weight": 1.0, "params": {"sigma": 0.25}},
    {"id": "tracking.root_pose", "enabled": true, "weight": 0.5, "params": {"sigma": 0.2}},
    {"id": "regularization.action_rate", "enabled": true, "weight": -0.02, "params": {}},
    {"id": "regularization.torque", "enabled": true, "weight": -0.001, "params": {}}
  ],
  "terminations": ["timeout", "bad_anchor_orientation", "fall", "joint_limit", "nan_inf"],
  "annealing": []
}
```

`fall`、`joint_limit`、`nan_inf`、控制周期和执行器硬限位不能被普通用户关闭。奖励实现只来自平台固定 worker 镜像中的注册表，worker 启动前对 term id、版本、参数范围和实现 hash 做白名单校验。

### 7.7 `TrainingConfig`

训练配置要完整暴露，但每个字段必须有类型、默认值、范围、单位和解释。建议结构：

```json
{
  "schema_version": "training_config.v1",
  "task_id": "g1_mimic",
  "scene_id": "g1_flat",
  "motion_asset_version_id": "uuid",
  "observation": {
    "history_length": 3,
    "include_root_velocity": true,
    "include_projected_gravity": true,
    "include_reference": true,
    "clip_value": 100.0
  },
  "action": {"mode": "joint_position_delta", "scale": 0.25, "clip": 1.0},
  "control": {"decimation": 1, "kp_profile": "g1_default", "kd_profile": "g1_default"},
  "ppo": {
    "algorithm": "rsl_rl_ppo",
    "seed": 1234,
    "num_envs": 4096,
    "max_iterations": 5000,
    "rollout_length": 24,
    "learning_rate": 0.001,
    "schedule": "adaptive",
    "gamma": 0.99,
    "lam": 0.95,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 1.0,
    "max_grad_norm": 1.0,
    "hidden_dims": [512, 256, 128]
  },
  "domain_randomization": {
    "enabled": true,
    "mass_scale": [0.95, 1.05],
    "friction": [0.7, 1.3],
    "motor_strength": [0.95, 1.05]
  },
  "resources": {"gpu_count": 1, "gpu_memory_gb": 8, "cpu_cores": 8, "shared_memory_gb": 8}
}
```

配置验证必须阻止观测维度、动作维度、RobotSpec、MotionLoader 字段或任务注册不一致的组合。改变这些结构字段时不能从旧 checkpoint 继续；仅改变奖励权重、学习率等实验字段时可由适配器声明是否允许 resume。

### 7.8 `RunManifest`

Run 创建成功前生成预览；用户确认后冻结。至少包含：

```json
{
  "schema_version": "run_manifest.v1",
  "project_id": "uuid",
  "run_id": "uuid",
  "attempt_id": "uuid",
  "parent_run_id": null,
  "robot": {"robot_id": "unitree_g1_29dof", "adapter_sha": "...", "spec_sha256": "..."},
  "motion": {"input_mode": "video|direct", "asset_versions": [], "train_motion_sha256": "..."},
  "reward_config_sha256": "...",
  "training_config_sha256": "...",
  "runtime": {
    "isaac_lab_git": "v2.3.0@3c6e67bb5",
    "isaac_lab_package": "0.47.2",
    "isaac_sim_package": "5.1.0.0",
    "unitree_rl_lab_package": "0.2.1",
    "unitree_mujoco_git": "ae6a840",
    "gmr_git": "bb1bbe4",
    "gvhmr_git": "6ec3ca3",
    "mujoco_runtime": "3.3.6",
    "python": "3.11",
    "torch": "2.7.0+cu128",
    "cuda_driver": "...",
    "container_digest": "..."
  },
  "execution": {"seed": 1234, "gpu_uuid": "...", "num_envs": 4096},
  "licenses": [],
  "manifest_sha256": "computed_after_serialization"
}
```

manifest 序列化使用 canonical JSON，字段排序、UTF-8、无多余空白固定，hash 计算后不可修改。

### 7.9 `PolicyBundle`

```text
policy_bundle.tar.zst
  policy.pt
  policy.onnx
  params/deploy.yaml
  params/env.yaml
  params/agent.yaml
  params/normalization.json
  params/action_scale.json
  params/pd.json
  manifest/run_manifest.json
  manifest/manifest.sha256
  reports/sim2sim_report.json
  reports/sim2sim_report.html
  videos/seed-*.mp4
  metrics/*.jsonl
  checksums.sha256
  LICENSES/
```

JIT 和 ONNX 导出后必须执行独立 smoke inference：固定输入 shape、有限值、动作 shape `[batch, 29]`、动作范围和归一化参数与环境一致。ONNX 要登记 opset、输入输出名和导出工具版本。

## 8. 动作转换与质量门禁实现

### 8.1 阶段 runner

每个阶段均实现统一 runner：

```text
validate_input
  → prepare_workspace
  → execute_converter_or_command
  → validate_output
  → upload_artifacts
  → write_metrics_and_event
```

阶段 runner 必须可幂等执行。幂等键为 `input_hash + processor_version + config_hash + robot_version`。重复投递先检查已登记的 output hash；存在完整产物时只补发事件，不重复运行。

### 8.2 视频/GVHMR/GMR

- GVHMR worker 使用独立 Python 3.10 环境，固定 Torch `2.3.0+cu121` 和 PyTorch3D 版本。
- GMR worker 使用独立环境，锁定 GMR `bb1bbe4`、Mink 版本和 MuJoCo 版本；不将 GVHMR 依赖安装到 Isaac 环境。
- 子进程命令使用参数数组；路径来自后端生成的白名单工作目录；禁止 `shell=True`。
- 输出目录必须包含原始命令数组、环境摘要、stdout/stderr、退出码、输入和输出 hash。

### 8.3 直接文件识别与转换

转换器注册接口：

```python
class MotionFormatConverter(Protocol):
    name: str
    version: str
    extensions: tuple[str, ...]

    def inspect(self, path: Path) -> SourceMotionDescriptor: ...
    def convert_to_intermediate(self, path: Path, output_dir: Path) -> IntermediateMotion: ...
```

首期实现：

- `PtTensorConverter`：读取安全的 tensor/dict 元数据；禁止反序列化未知可执行对象。优先使用 `torch.load(..., weights_only=True)` 或受限 loader，失败则拒绝。
- `NpzConverter`：只读取白名单数组名和 JSON metadata；拒绝 object dtype。
- `CsvConverter`：要求 header、单位声明或显式映射；行数、列数和关节名校验。
- `PklConverter`：只允许受信来源或平台生成的 pickle；普通用户上传 pickle 默认进入隔离解析进程，禁止直接 import 任意模块。

转换器输出中间格式后统一进入 `G1MotionNormalizer`，补充 fps、根位姿、速度、body 状态和 G1 关节顺序。

### 8.4 数值和运动学检查

至少执行：

1. shape、dtype、finite 值和空帧检查；
2. 四元数归一化、零四元数和 `xyzw/wxyz` 转换；
3. 关节名称、DoF、qpos 地址和 G1 29 DoF 顺序检查；
4. fps 重采样，默认使用时间戳驱动的线性位置/SLERP 旋转插值；
5. 关节位置、速度、加速度、动作缩放和力矩限位检查；
6. `mj_forward` 后的身体位置/姿态有限值检查；
7. 足端接触和滑动告警；
8. 根高度、地面穿透、异常跳变和动作切片边界检查；
9. 质量报告写入 JSON，不以“告警”代替安全硬失败。

质量门禁结果分为：

- `BLOCKING_ERROR`：不能进入训练；
- `WARNING`：允许用户查看并确认；
- `INFO`：仅记录统计。

## 9. Isaac Lab + RSL-RL/PPO 训练实现

### 9.1 G1 imitation 任务

首期注册稳定 task id：`g1_mimic`。任务配置由以下层组成：

```text
robot config
  → scene config
  → observation config
  → action/control config
  → reference motion config
  → registered reward config
  → safety termination config
  → domain randomization config
  → RSL-RL/PPO agent config
```

训练环境负责从 `TrainMotionNPZ` 读取参考帧，根据 episode 时间、循环或片段采样生成 reference state。任务不把某个动作名称或箱体阶段写入环境核心；动作资源通过 manifest 注入。

### 9.2 观测和动作

观测至少明确：关节位置/速度、根速度、投影重力、上一动作、参考关节状态、参考根姿态/速度和可选历史帧。每一项声明 shape、顺序、缩放和 clip。

动作首期默认 `joint_position_delta`，输出 29 维，再由 `RobotSpec.action_scale`、默认姿态和 PD 控制器转换为目标关节位置。控制周期、decimation、kp/kd 和 torque limit 来自 G1 adapter，用户只能在 schema 允许的范围内调整。

### 9.3 注册奖励

首期建议注册：

- `tracking.joint_pos`：参考与实际关节位置误差；
- `tracking.joint_vel`：参考与实际关节速度误差；
- `tracking.root_pos`、`tracking.root_orientation`；
- `tracking.body_pose`：手、脚、躯干等 body 误差；
- `regularization.action_rate`、`regularization.torque`；
- `stability.contact`、`stability.foot_slip`；
- 硬终止：`nan_inf`、`fall`、`joint_limit`、`bad_anchor_orientation`、`timeout`。

每个 term 的实现、版本和单元测试在固定 worker 镜像中锁定。用户调整只改变 JSON 参数，不改变代码路径。

### 9.4 训练阶段输出

训练 worker 每个固定间隔写入结构化指标：总回报、每个 term 回报、episode length、跌倒率、关节跟踪误差、动作饱和率、显存、GPU 利用率、CPU、checkpoint URI 和视频 URI。指标批量写入，日志用单调 `seq`，避免 SSE 重连时丢失或重复。

### 9.5 自动生成动作模仿配置

每个合法动作进入训练配置页时，后端调用 `ImitationConfigBuilder`，根据动作、G1 适配器和用户覆盖生成一份可审阅的初始配置。生成器不依赖动作名称，也不为箱体、推门等任务写特殊分支。

输入：

- `TrainMotionNPZ` 的 fps、帧数、身体覆盖率、速度和质量报告；
- G1 `RobotSpec` 的 29 DoF 顺序、动作维度、控制周期、限位和默认 PD；
- 用户选择的场景、随机化级别和 PPO 覆盖参数；
- 注册的 `g1_mimic_v1` reward template。

自动推导：

1. 将动作 fps 重采样到 `policy_dt` 的整数倍；不满足时提示重采样损失。
2. 由 G1 DoF 和观测选项计算 observation/action shape，写入配置预览。
3. 根据动作时长计算默认 episode horizon：`ceil(duration / policy_dt)`，并限制在适配器允许范围内。
4. 默认启用 joint/root/body tracking、action-rate 和 torque regularization；安全终止强制启用。
5. 根据动作速度和质量报告初始化 `clip_value`、tracking sigma、domain randomization 上限；这些只是可编辑初值，不覆盖安全硬约束。
6. 选择默认 `num_envs`、rollout length、PPO hidden dims 和训练迭代上限；若资源声明超过当前 GPU 装箱预算，生成器返回可解释的资源警告。

输出 `GeneratedImitationConfig`，包含：

```json
{
  "schema_version": "generated_imitation_config.v1",
  "source_motion_sha256": "...",
  "robot_spec_sha256": "...",
  "derived": {
    "policy_dt": 0.02,
    "episode_horizon": 1500,
    "observation_dim": 312,
    "action_dim": 29,
    "active_reward_terms": ["tracking.joint_pos", "tracking.root_pose", "regularization.action_rate"]
  },
  "defaults": {"...": "..."},
  "user_overrides": {"...": "..."},
  "builder_version": "imitation-config-builder.v1"
}
```

用户修改参数后只写入 `user_overrides`，重新计算派生字段并显示差异。配置最终冻结进 `RunManifest`；如果用户修改导致维度、RobotSpec 或 MotionLoader 不兼容，启动按钮必须阻断。

## 10. 作业状态机、取消、重试和失败分类

### 10.1 Run 与 Attempt

- `Run` 是用户视角的逻辑作业和配置版本。
- `Attempt` 是一次具体执行，重试创建新 attempt，原 attempt 永不覆盖。
- `RunManifest` 在启动确认时冻结，attempt 只补充 GPU、worker 和退出信息。

### 10.2 状态

```text
CREATED
→ UPLOADING → UPLOADED → VALIDATING
→ GVHMR_RUNNING → GVHMR_READY
→ GMR_RUNNING → RETARGET_READY
→ MOTION_COMPILING → MOTION_READY
→ MOTION_EDITING → MOTION_VALIDATING
→ TRAINING_PREPARING → TRAINING
→ TRAINING_SUCCEEDED → EXPORTING
→ EXPORTED → SIM2SIM_QUEUED → SIM2SIM_RUNNING
→ SIM2SIM_PASSED → READY_TO_DOWNLOAD
```

任意可执行状态可以进入 `FAILED` 或 `CANCELLED`。`FAILED_NEEDS_REVIEW` 表示超过自动重试次数或需要人工诊断。

### 10.3 重试策略

- 对象存储网络错误、worker 心跳丢失、报告生成失败：指数退避自动重试。
- schema、许可证、关节映射、奖励参数错误：不自动重试，提示用户修改。
- OOM、Isaac Sim 崩溃、MuJoCo 进程崩溃：默认最多 1 次自动重试；第二次失败保留诊断并进入人工复核。
- 用户取消必须传播到当前子进程，等待清理后再标记 `CANCELLED`；强杀也要记录 signal 和退出码。

### 10.4 失败分类

| 分类 | 例子 | 回退位置 |
| --- | --- | --- |
| `INPUT_INVALID` | 文件字段、MIME、许可证或尺寸错误 | 重新上传 |
| `RETARGET_INVALID` | GMR 映射、四元数、限位、接触质量错误 | 动作转换/编辑 |
| `TRAIN_CONFIG_INVALID` | 观测/动作维度或 PPO 参数错误 | 训练配置 |
| `TRAINING_UNSTABLE` | NaN、持续摔倒、动作饱和 | 奖励/控制/训练参数 |
| `EXPORT_INVALID` | JIT/ONNX shape、opset 或 hash 错误 | 导出阶段 |
| `SIM2SIM_MISMATCH` | 模型、控制周期、PD 或动作缩放不一致 | 仿真适配器 |
| `RESOURCE_OOM` | 显存、CPU、磁盘或共享内存不足 | 调度/资源配置 |

## 11. API 设计

前缀 `/api/v1`。每个响应包含 `request_id`、`resource_version`；错误使用统一结构：

```json
{
  "error": {
    "code": "MOTION_SCHEMA_INVALID",
    "message": "joint_pos shape does not match G1 29 DoF",
    "stage": "direct_motion_detect",
    "details": {"expected": ["N", 29], "actual": [180, 28]},
    "retryable": false,
    "suggested_action": "provide joint mapping or upload a 29 DoF trajectory"
  },
  "request_id": "..."
}
```

### 11.1 资源与上传

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/projects` | 创建项目 |
| `GET` | `/projects/{id}` | 项目和成员 |
| `POST` | `/projects/{id}/assets` | 创建资源版本和上传会话 |
| `POST` | `/assets/{id}/upload-url` | 获取 presigned/multipart URL |
| `POST` | `/assets/{id}/upload-complete` | 完成对象存储上传 |
| `GET` | `/assets/{id}/versions` | 资源版本、hash、许可证 |
| `POST` | `/assets/{id}/validate` | 触发格式/许可证/质量校验 |

`upload-complete` 只表示对象存储上传完成；只有 worker 完成服务端 hash、MIME、内容和许可证校验后，版本才变为 `UPLOADED` 可训练。

### 11.2 机器人、动作和奖励

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/robots` | 适配器、能力、版本和许可证 |
| `GET` | `/robots/{id}/self-check` | XML/URDF/关节/控制映射自检 |
| `POST` | `/motions` | 创建动作处理请求 |
| `POST` | `/motions/{id}/detect` | 识别直接文件类型 |
| `POST` | `/motions/{id}/retarget` | 触发 GVHMR/GMR 或人体姿态重定向 |
| `GET` | `/motions/{id}/preview` | 帧、曲线、质量告警和 PNG URI |
| `POST` | `/motion-edits` | 保存 MotionEditConfig 新版本 |
| `POST` | `/motion-edits/{id}/compile` | 生成 TrainMotionNPZ |
| `GET` | `/reward-templates` | 获取注册 term schema |
| `POST` | `/reward-configs` | 创建 RewardConfig 版本 |
| `GET` | `/training-config/schema` | 获取完整 PPO/观测/控制 schema |

### 11.3 运行、事件和产物

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/runs` | 校验配置、冻结 manifest、创建 Run/Attempt |
| `GET` | `/runs/{id}` | 状态和摘要 |
| `GET` | `/runs/{id}/events` | SSE 日志、指标和状态事件 |
| `POST` | `/runs/{id}/cancel` | 取消当前 attempt |
| `POST` | `/runs/{id}/retry` | 创建新 attempt |
| `POST` | `/runs/{id}/export` | 触发 JIT/ONNX 导出 |
| `POST` | `/runs/{id}/sim2sim` | 触发三种子验收 |
| `GET` | `/runs/{id}/comparison` | 父 Run 与当前 Run 对比 |
| `GET` | `/artifacts/{id}` | 产物元数据和短时下载地址 |

启动 `/runs` 必须使用幂等键。Local File Mode 的 API 只向 `robotlabd` scheduler 提交一次命令，不等待 worker；Compose Mode 在事务中写入 Run/Attempt 和 outbox，再由 dispatcher 投递。

## 12. 持久化、对象存储和事件

平台提供两个可替换 profile：

| Profile | 适用范围 | 持久化与调度实现 |
| --- | --- | --- |
| `local-file`（默认） | 单机单用户、多作业并发 | 本地文件仓库、内容寻址 artifacts、`state.json`/`events.jsonl`/`metrics.jsonl`、`robotlabd` scheduler、PID/lease 文件 |
| `compose`（可选） | 团队共享、远程 GPU、多用户、负载均衡 | PostgreSQL、Redis/Celery、MinIO/S3；与远程部署保持同一服务边界 |

两种 profile 必须实现同一组 domain ports：`ProjectRepository`、`AssetRepository`、`RunRepository`、`ArtifactStore`、`EventStore`、`JobScheduler`。前端 API、Run Manifest、PolicyBundle 和 Sim2SimReport 不得感知具体 profile。

### 12.1 Compose Mode：PostgreSQL 表

核心表：`users`、`projects`、`project_members`、`assets`、`asset_versions`、`robot_adapters`、`robot_versions`、`reward_templates`、`reward_configs`、`motion_edits`、`runs`、`attempts`、`run_manifests`、`metric_points`、`log_events`、`artifacts`、`sim2sim_evaluations`、`evaluation_seeds`、`audit_events`、`outbox_events`。

所有业务状态迁移和审计事件必须在同一事务中写入。`sha256`、manifest hash 和 artifact hash 建唯一索引；历史 manifest 只读。

### 12.1.1 Local File Mode：文件事实源

Local File Mode 不使用 PostgreSQL、Redis 或 MinIO。每个实体以不可变版本目录保存，Run 以 `manifest.json`、`state.json`、`events.jsonl`、`metrics.jsonl` 和 `artifacts/` 组成。`robotlabd` 是唯一的状态写入者，CLI/API 通过本地 IPC 或锁文件提交命令；文件提交使用临时文件、`fsync` 和原子 rename。`index.json` 仅为可重建缓存，不能作为唯一事实源。

### 12.2 对象 key

```text
projects/{project_id}/assets/{asset_id}/versions/{version_id}/source/{random_name}
projects/{project_id}/assets/{asset_id}/versions/{version_id}/derived/{kind}/{random_name}
projects/{project_id}/runs/{run_id}/attempts/{attempt_id}/artifacts/{kind}/{random_name}
```

原始文件名只作为展示字段保存，不直接拼接进路径。Local File Mode 使用内容寻址目录和本地文件流；Compose Mode 使用对象存储 key，视频大于 100 MB 默认 multipart。对象读取先做项目和 Run 状态校验，Compose Mode 可返回短时 presigned URL。

### 12.3 Compose Mode：Redis/Celery/outbox

API 在一个 PostgreSQL 事务中写 `Run`、`Attempt`、初始状态和 `outbox_events`；dispatcher 投递成功后标记 outbox。重复投递由 `attempt_id + stage + input_hash` 幂等键消除。

Redis 只承载 broker、短期事件、锁、心跳、限流和缓存，不作为唯一事实源。SSE 历史事件从 PostgreSQL `log_events`/`metric_points` 恢复。

Local File Mode 不启用 Redis/Celery/outbox。`robotlabd` 直接管理本地子进程和 GPU lease；SSE 历史事件从每个 Run 的 `events.jsonl` 和 `metrics.jsonl` 按 `seq` 恢复。调度器崩溃后依据 lease、PID、心跳和最后事件执行恢复扫描。

### 12.4 本地单用户与未来项目权限

首期默认本地单用户运行，不要求登录，不建设登录页、用户注册或远程账号体系。启动命令生成本地实例标识，API 默认只监听本机回环地址；项目、动作、配置、日志、checkpoint、策略包和报告全部写入用户配置的数据目录。

- 本地模式下所有项目默认为当前实例所有者，前端不展示登录和成员管理流程；
- 单个用户可以同时提交多个训练、导出和 sim2sim 作业；本地调度器按每张 GPU 的实时显存、利用率、CPU、温度和 worker 健康状态决定并发或排队；
- 对象下载、训练启动和产物导出仍必须经过 `project_id`、状态和路径校验，防止跨项目读取；
- `users`、`project_members`、`organization_id`、项目配额、队列优先级和 worker pool 等字段保留为未来多用户/远程部署扩展点；
- 后续接入团队共享服务器时，再启用 `owner`、`editor`、`viewer` 和短期 job token，不修改 Asset、Run、Artifact 或 adapter 契约；
- 本地安装、配置冻结、训练取消、重试、发布和导出仍写入 `audit_events`，便于复现和故障追踪。

## 13. GPU 动态调度和负载均衡

### 13.1 资源声明

每个 GPU 作业在 `TrainingConfig.resources` 中声明：`gpu_count`、`gpu_memory_gb`、预计 utilization、CPU cores、RAM、shared memory、磁盘和最长运行时间。默认 RL 训练：1 GPU、8 GB 显存预算、8 CPU 核、8 GB shared memory；实际值由 worker 启动后回写。

### 13.2 GPU lease

调度器维护每张卡：GPU UUID、显存总量、当前已分配、利用率、温度、健康状态、运行中的 lease。Compose Mode 使用 Redis 锁和 TTL；Local File Mode 使用带过期时间的 lease 文件和进程锁。worker 每 10 秒心跳；失联后由调度器回收并将 attempt 标记为 `WORKER_LOST`，不删除已有产物。

### 13.3 动态装箱规则

1. 每张 RTX 4090 首期软上限 3 个训练作业。
2. 新作业只有在 `allocated_memory + request <= safety_memory_limit`、CPU 和 shared memory 足够且 GPU 健康时才接纳。
3. 采样窗口内 GPU 利用率持续高于 90%、显存使用高于 90% 或温度超过配置阈值时停止接纳新作业。
4. 已运行作业不因新作业到来被驱逐；新作业排队并显示预计资源原因。
5. Isaac Sim、sim2sim 和高显存 export 可以声明 `exclusive_gpu=true`，暂不与其他 GPU 任务共卡。
6. 首期不做跨两卡分布式 PPO；双卡是两个可并行的单卡资源池。
7. 调度器保留项目级并发、每日 GPU 时长和队列优先级字段，便于未来多用户负载均衡。

### 13.4 OOM 和背压

GPU OOM 时记录显存快照、配置 hash、进程命令和 worker 镜像；自动降低并发或重新排队不改变原 manifest。队列超过容量返回 `QUEUE_CAPACITY_EXCEEDED`，不能无限积压。

## 14. 本地部署和运行环境

### 14.1 本地运行模式

项目不是远程 Web SaaS，而是完整交付到用户本地的工程。首期默认使用 `Local File Mode`，不依赖 Docker、PostgreSQL、Redis 或 MinIO；`Compose Mode` 作为团队共享、远程 GPU、多用户和负载均衡的可选扩展。两种模式必须提供相同的 API、前端工作流、Run Manifest、策略包和 sim2sim 报告格式。

```text
Local File Mode:
robotlab CLI
    └─ robotlabd（FastAPI + 本地 scheduler）
       ├─ web（React 本地工作台）
       ├─ file repository / artifact store
       └─ GPU workers（GVHMR、GMR、Isaac、sim2sim）

Compose Mode:
robotlab CLI
    └─ Docker Compose
       ├─ web + api
       ├─ postgres-16 + redis-7 + minio
       ├─ celery-dispatcher/cpu-workers
       └─ GPU workers（GVHMR、GMR、Isaac、sim2sim）
```

Compose Mode 的 API 容器禁止安装或导入 Isaac Sim；各 worker 通过 Compose 网络访问 PostgreSQL、Redis 和 MinIO。Local File Mode 的 API 和 worker 通过本地文件仓库及 scheduler 通信。所有项目数据、模型缓存、日志、checkpoint、策略包和报告写入用户指定的 `runtime/` 数据目录。

### 14.1.1 无数据库本地模式（单用户多作业）

本地最终交付可以不依赖 PostgreSQL、Redis 和 MinIO。`robotlabd`（本地 API 与调度控制器）作为唯一调度状态写入者，使用文件原子提交、进程锁和追加事件日志管理多个并发作业：

```text
用户提交 Run
  → manifest.json（冻结配置）
  → scheduler 读取 resources 声明和 GPU 实时指标
  → 并发启动满足装箱条件的进程，其他 Run 进入 QUEUED
  → 每个 Run 独立 state.json / events.jsonl / metrics.jsonl
  → 产物写入 content-addressed artifacts/ 并登记 hash
```

无数据库模式必须满足：

1. 每个 Run 使用独立目录和锁文件；同一 Run 只能有一个活动 attempt。
2. `state.json` 通过临时文件、`fsync` 和原子 rename 更新；`events.jsonl` 使用单调 `seq`，前端可断线续读。
3. 作业提交、取消、重试和恢复由单一 scheduler 串行化；CLI 与 API 通过本地 IPC/锁提交命令，不能并发修改状态文件。
4. scheduler 每 5–10 秒采集 `nvidia-smi` 指标，按 `gpu_memory_gb`、预计利用率、CPU、shared memory、温度和 `exclusive_gpu` 做动态装箱。
5. 首期每张 RTX 4090 默认最多 3 个训练作业；显存或利用率超过阈值时只停止接纳新作业，不驱逐已运行作业。
6. 进程退出、机器重启或 scheduler 崩溃后，启动恢复扫描依据 PID、心跳和最后事件把 Run 标记为 `INTERRUPTED`、重新排队或保留可恢复 checkpoint。
7. `index.json`、项目列表和统计数据均为可重建缓存，损坏时从 manifest、state 和 artifact 目录重建。

该模式可以保持与数据库模式相同的 API 和前端体验，但只保证单机单用户范围内的并发和恢复能力；不提供跨机器队列、团队权限和高可用复制。

### 14.2 统一命令行入口

首期提供稳定、可脚本化的 `robotlab` 命令，Windows WSL2 Ubuntu 22.04 和原生 Linux Ubuntu 22.04 使用同一套 Linux 运行时：

```text
robotlab install              检查宿主机、WSL2、Docker、GPU 和联网前置条件
robotlab init                 创建本地配置、数据目录、运行 profile 和项目工作区
robotlab doctor               重新检查当前 profile 的组件、版本、GPU、挂载和服务连通性
robotlab start                启动本地 API、前端、scheduler 和 worker（Compose profile 另启数据库服务）
robotlab stop                 停止本地服务，不删除项目数据和产物
robotlab status               查看服务、队列、GPU worker 和运行状态
robotlab robot add --path ... 注册用户提供的机器人资产包
robotlab robot list           查看已注册机器人及自检状态
robotlab run --project ...    从冻结配置启动动作处理、训练或 sim2sim 作业
robotlab logs <run_id>        查看结构化日志和阶段进度
robotlab artifact export ...  导出策略包、manifest、报告和校验和
```

`install` 和 `doctor` 只检查并输出明确安装指引，不自动修改 Windows 驱动、WSL2、内核、Docker、NVIDIA 系统组件或 Conda 环境。Local File Mode 不检查 PostgreSQL、Redis、MinIO 和 Docker；Compose Mode 才检查这些服务。缺少组件时返回稳定错误码、检测版本、要求版本、官方安装地址和下一步命令；用户手动安装后再次运行 `doctor`。

### 14.3 Windows 运行要求

1. Windows 11 安装 WSL2、Ubuntu 22.04 和 NVIDIA WSL CUDA 支持；选择 Compose Mode 时再安装 Docker Desktop/WSL2 集成。
2. Local File Mode 的 Linux 运行时、Isaac Lab、MuJoCo、训练和 sim2sim 任务统一在 WSL2 Ubuntu 22.04 内执行；Compose Mode 额外通过 Docker 容器执行。
3. Windows 主机只作为 CLI 入口和浏览器宿主；代码挂载、GPU 映射、模型缓存和训练数据必须经过 WSL2 路径验证。
4. Windows WSL2 必须具备与 Linux 相同的训练、导出和 sim2sim 能力，不允许把 Windows 版本定义为仅开发模式。

### 14.4 Linux 运行要求

1. Ubuntu 22.04 和 NVIDIA 驱动；选择 Compose Mode 时再安装 Docker Engine 和 NVIDIA Container Toolkit。
2. Local File Mode 使用本地 Conda/虚拟环境和进程 scheduler；Compose Mode 使用与 WSL2 相同的 Compose 文件、环境变量契约、worker 镜像和数据目录结构。
3. Linux 主机直接提供 GPU 训练能力，不依赖额外远程 GPU 服务器。

允许联网下载依赖和模型，但下载内容必须登记来源 URL、版本或 revision、许可证、文件大小和 SHA-256。未经登记的浮动模型缓存不能直接进入训练或发布。

### 14.5 分环境 worker 基线

必须分环境或分容器：

| 环境 | 基线 |
| --- | --- |
| GVHMR | Python 3.10、Torch 2.3.0+cu121、PyTorch3D、GVHMR `6ec3ca3` |
| GMR | Python 3.10、GMR `bb1bbe4`、Mink lockfile、MuJoCo 版本单独锁定 |
| Isaac | Ubuntu 22.04、Python 3.11、Isaac Lab `v2.3.0@3c6e67bb5`、Isaac Lab package `0.47.2`、Isaac Sim `5.1.0.0`、Torch `2.7.0+cu128` |
| sim2sim | Unitree MuJoCo `ae6a840` + overlay commit、MuJoCo runtime `3.3.6`、SDK2；C++ 和 Python 运行时分离 |

manifest 还必须记录 CUDA driver、NVIDIA driver、容器 digest、Unitree RL Lab server SHA、Unitree MuJoCo overlay SHA 和 Isaac Sim source/build identity。未采集的字段不能填假值，使用 `pending` 并阻止发布验收。

### 14.6 本地 doctor 检查链

```text
GPU 设备和驱动可见
→ WSL2/Ubuntu 或 Linux 版本满足要求
→ Local File Mode：Conda/虚拟环境、进程权限和磁盘空间可用
  或 Compose Mode：Docker Engine/Desktop、NVIDIA runtime、挂载和磁盘空间可用
→ Isaac Lab/Isaac Sim smoke import
→ GVHMR checkpoint、SMPL/SMPL-X 模型和许可证存在
→ 已注册机器人三侧资产和 mesh hash 一致
→ MuJoCo EGL renderReady=true
→ Local File Mode：robotlabd、scheduler 和本地 worker 心跳正常
  或 Compose Mode：PostgreSQL/Redis/MinIO 健康、dispatcher 和 worker 心跳正常
```

### 14.7 用户机器人资产注册

平台不替用户下载或假定某个厂家的完整机器人资产。用户必须先注册自己的资产包：

```bash
robotlab robot add --path /data/robots/<robot_asset_package>
```

注册过程生成不可变 `RobotVersion` 和 `RobotSpec`，并检查：

- URDF/MJCF/XML/USD 文件存在性、可解析性和许可证；
- 网格引用、相对路径、纹理和碰撞资源；
- 关节、DoF、body、qpos/qvel 地址和轴向；
- Isaac、Motion/GMR 和 MuJoCo sim2sim 三侧资产声明；
- 关节映射、初始状态、位置/速度/力矩限位；
- 执行器、PD、动作缩放、控制周期和传动关系；
- 资产及派生配置的 SHA-256、来源、版本和许可证。

平台只生成经过校验的 staging 目录和后端配置，不修改用户原始资产，也不从 URDF/MJCF 静默推导缺失的关键控制参数。缺少关键字段时返回字段级失败原因，未通过自检的机器人不能进入实验选择器。

## 15. sim2sim 验收规则

### 15.1 执行方式

每个策略 bundle 启动一个 evaluation batch，固定 3 个种子，例如 `20260101`、`20260102`、`20260103`。每个 seed 使用相同动作、初始状态集合、控制周期和模型 hash，只改变随机化种子。验收 worker 运行 Unitree MuJoCo，必要时先生成 Isaac play 对照轨迹。

评估时长为 `max(30 秒, 参考动作时长)`，循环动作必须记录循环次数。报告保存每个 seed 的视频、指标 JSON、曲线和完整命令。

### 15.2 默认阈值

阈值写入 `Sim2SimPolicy.v1`，普通用户不能放宽；仿真工程师可以通过适配器版本升级修改并留下审计记录。

| 指标 | 默认通过条件 | 硬失败条件 |
| --- | --- | --- |
| 进程完成 | 3/3 seed 正常退出 | 任一崩溃、超时或退出码非 0 |
| 数值稳定 | NaN/Inf = 0 | 任一 NaN/Inf |
| 存活/跌倒 | 每 seed 存活率 ≥ 0.90，三 seed 中位数 ≥ 0.95 | 任一 seed 存活率 < 0.80 |
| 根高度 | 低于安全高度的帧占比 ≤ 5% | 根高度持续低于安全高度或穿透地面 |
| 根姿态 | roll/pitch 超过 35° 的帧占比 ≤ 5% | 连续 0.5 秒超过 50° |
| 关节跟踪 | 29 DoF RMSE ≤ 0.35 rad，中位数；p95 ≤ 0.80 rad | 中位数 > 0.50 rad |
| 根位置/姿态误差 | 根位置 RMSE ≤ 0.20 m；姿态误差 ≤ 20° | 根位置 > 0.40 m 或姿态 > 35° |
| 动作饱和 | torque/action 饱和帧占比 ≤ 5% | 任一关节连续 1 秒饱和 |
| 足端滑动 | 接触期间平均速度 ≤ 0.15 m/s | 接触滑动持续超过 1 秒 |
| 可复现性 | 三 seed 关键指标最大差异 ≤ 20% | 差异 > 35% |

“通过”要求所有硬失败条件为假，且关键指标满足默认条件。动作本身若在质量检查阶段被标记为低质量，报告中必须区分“输入质量不足”和“策略/仿真失败”，不能只给一个红色状态。

### 15.3 失败诊断

- 动作/重定向：参考轨迹跳变、限位、四元数、接触和根高度异常；回到动作编辑。
- 奖励/训练：训练回报异常、跌倒率高、动作饱和或 NaN；回到 Reward Builder/PPO。
- 模型/控制：Isaac 与 MuJoCo 的 joint order、PD、control dt、action scale、质量/摩擦不一致；提交仿真工程师。
- 运行时：显存、驱动、容器、文件或 worker 资源问题；提交运维诊断。

## 16. 策略发布和产物管理

发布前执行：

1. 检查 checkpoint 来源和训练 attempt 状态；
2. 导出 JIT 和 ONNX；
3. CPU smoke inference 和 shape/finite/range 检查；
4. 写入 `deploy.yaml`、`env.yaml`、`agent.yaml`、归一化、动作缩放和 PD；
5. 复制 Run Manifest 和许可证快照；
6. 生成 sim2sim JSON/HTML 报告和视频；
7. 计算每个文件和整个压缩包 SHA-256；
8. 通过权限策略后生成短时下载地址。

策略包页面必须显示：机器人和 adapter 版本、控制周期、动作维度、观测维度、训练 seed、sim2sim seed、阈值结果、许可证、人工复核状态和“不可直接用于真实机器人”的安全提示。

## 17. 代码规范和评审门禁

### 17.1 通用规范

- UTF-8、LF、ASCII 标识符优先；文件路径和环境变量不得来自未校验的用户字符串。
- 所有公共 DTO、端口、事件和错误码必须有版本或稳定 id。
- 不提交权重、视频、checkpoint、secret、`.runtime` 缓存和本地环境目录。
- 外部版本必须进入 lockfile、容器 digest 或 manifest；禁止用浮动 `main` 作为运行身份。
- 使用 Conventional Commits；PR 说明影响模块、迁移、配置、测试和回滚方式。

### 17.2 TypeScript/React

- `strict: true`，禁止隐式 `any`；ESLint、Prettier、import/order。
- 组件只负责显示和事件转发；业务变换放在 feature service/hook。
- API 类型从共享 schema 生成或 contract test 对齐，禁止手写重复 DTO。
- 所有异步状态覆盖 loading、empty、error、retry、forbidden。
- 复杂算法、坐标变换和奖励计算不得放在组件 render 中。

### 17.3 Python

- Ruff、Black、Mypy/Pyright、Pytest；FastAPI DTO 使用 Pydantic v2。
- SQLAlchemy 2 typed mapping；迁移使用 Alembic。
- 子进程使用 `subprocess.run([...], shell=False)`，命令和环境变量通过 adapter 构造。
- worker 必须支持幂等、取消、超时、心跳、结构化日志和错误码。
- Domain 不导入 FastAPI、Celery、SQLAlchemy、Isaac、MuJoCo 和具体 SDK。

### 17.4 注释规范

只在“为什么”不明显时写注释：

- 坐标系、四元数、关节顺序和控制周期边界必须有短注释；
- 线程亲和性、EGL/GLFW 初始化和 renderer 生命周期必须说明原因；
- GPU 装箱阈值和安全硬终止必须说明来源；
- 不写重复代码含义的注释，不用 TODO 代替设计。

## 18. 测试策略和验收清单

### 18.1 单元测试

- `xyzw ↔ wxyz`、归一化、SLERP 和零四元数拒绝；
- G1 29 DoF joint/body/qpos 地址和限位校验；
- `.pt/.npz/.csv/.pkl` detector 的字段识别、恶意/不安全对象拒绝；
- RetargetMotion/TrainMotionNPZ shape、fps、finite、速度/限位校验；
- MotionEditConfig 应用、插值、关键帧、撤销和版本 hash；
- RewardConfig 参数范围、安全终止和版本复制；
- TrainingConfig 观测/动作维度和 PPO 范围；
- Run 状态迁移、幂等键、取消和重试；
- GPU lease、TTL、装箱、OOM 和 worker lost；
- manifest canonical hash、PolicyBundle checksums 和阈值聚合。

### 18.2 集成测试

- 上传 → 对象 hash → 许可证 → 直接格式识别 → TrainMotionNPZ；
- 视频 fixture → GVHMR mock → GMR mock → G1 motion；
- 后端 PNG 预览、关节编辑、关键帧、动作切换 reset；
- MotionEditConfig → compiler → Isaac MotionLoader fixture；
- Reward/TrainingConfig → manifest → worker 参数数组；
- outbox → Celery → 状态和事件；
- SSE 断线按游标恢复；
- JIT/ONNX 导出和独立 smoke inference；
- sim2sim fake adapter 三 seed 报告聚合；
- 项目权限、短时下载地址和软删除。

### 18.3 P0 系统验收

1. 本地 WSL2/Linux smoke：Isaac Lab v2.3.0 + Isaac Sim 5.1.0.0 空场景和 G1 mimic。
2. 视频模式：一条合法视频真实完成 GVHMR、GMR、Motion Compiler、训练和产物生成。
3. 直接模式：G1 `.npz/.csv/.pt/.pkl` 各一条、人体姿态文件至少一条完成闭环。
4. 29 DoF 契约测试通过，包含 joint order、四元数、fps、NPZ shape 和控制周期。
5. 用户调整 PPO/观测/奖励/控制参数后，manifest 和配置版本正确变化。
6. 任意非法输入在训练前阻断并给出稳定错误码。
7. 双 RTX 4090 可同时运行两个或三个低显存 RL 作业，超阈值时新作业排队。
8. 训练失败可定位 stage、命令、退出码、GPU 和日志；重试不覆盖原 attempt。
9. JIT/ONNX、deploy/env/agent、manifest、hash 和报告完整。
10. 三种子 sim2sim 报告满足第 15 节阈值或明确标记失败原因。

### 18.4 发布验收

- API 不加载 Isaac Sim；长任务不在 HTTP 请求线程运行。
- 业务事实源由运行 profile 决定：Compose Mode 使用 Postgres；Local File Mode 使用不可变 manifest、state/event journal 和内容寻址文件。两种模式都必须支持状态恢复，Redis、索引缓存或临时目录丢失不得破坏已提交 Run。
- 原始资源、配置、manifest、attempt 和产物不可变；删除采用软删除。
- GVHMR/GMR/Isaac/sim2sim 版本、许可证和 overlay 可追溯。
- Docker/Conda lockfile、GPU 驱动和容器 digest 在本地 WSL2/Linux 环境可复现；未来远程 worker 复用同一 lockfile。

## 19. 实施顺序和 Definition of Done

### 19.0 阶段 0：本地项目骨架和 CLI

交付 Local File Mode、本地运行目录、配置文件模板、可选 Compose Mode 和 `robotlab install/init/doctor/start/stop/status/run/logs/artifact` 命令。DoD：Windows WSL2 Ubuntu 22.04 与原生 Linux Ubuntu 22.04 均可在无数据库、无 Docker 的 Local File Mode 下完成检查、启动本地服务、打开 `localhost` 工作台、提交多个训练作业并按 GPU 负载并发或排队；Compose Mode 作为可选扩展验证，不自动修改系统组件。

### 19.1 阶段 A：通用契约与 G1 实例适配器

交付 `packages/contracts`、通用关节机器人 `RobotSpec`、关节/身体/执行器/传动/控制映射、错误码、fixtures 和 contract tests，再以 G1 作为第一个实例完成注册。DoD：通用 schema 能表达主流一对一和耦合传动；G1 29 DoF、MJCF/URDF/Isaac 资产 hash 和版本检查全部自动化。

### 19.2 阶段 B：两条动作输入和 Motion Compiler

交付上传校验、四类格式 detector/converter、人体姿态分支、GVHMR/GMR runner、RetargetMotion、TrainMotionNPZ 和质量报告。DoD：合法视频、G1 轨迹和人体姿态各自可生成可读、可加载、可重放的训练动作。

### 19.3 阶段 C：正式 API、数据和作业编排

交付 FastAPI、PostgreSQL、Redis、MinIO、outbox、Celery 队列、SSE、本地单用户访问控制、attempt 和审计。DoD：所有长任务异步化，断线、重试、取消、失败和下载均可恢复，数据和产物只写入本地运行目录。

### 19.4 阶段 D：Isaac Lab/RSL-RL 训练

交付 `g1_mimic` task、完整 TrainingConfig schema、Reward Registry、PPO runner、checkpoint、play 和 GPU 动态装箱。DoD：本地双 4090 或 WSL2 映射 GPU 可并行训练；一个合法动作能在本地 headless worker 生成 checkpoint 和指标。

### 19.5 阶段 E：导出与 sim2sim

交付 JIT/ONNX exporter、CPU smoke inference、Unitree MuJoCo adapter、三 seed evaluator、报告和策略包下载。DoD：满足第 15 节阈值的 Run 才能进入 `READY_TO_DOWNLOAD`。

### 19.6 阶段 F：前端生产工作台

交付项目、资源、动作编辑、奖励、训练监控、sim2sim 和策略包页面；迁移现有原型的真实 MuJoCo PNG 能力。DoD：前端不包含仿真业务规则，所有状态由 API/SSE 驱动，Playwright 覆盖主流程。

## 20. 多机器人和任务语义扩展

### 20.1 第二机器人接入规则

第二机器人只能通过新增 `RobotSpec` 实例和 adapter 完成，不能修改通用契约：

- 新 `RobotSpec` 实例、资产 lock、joint/body/actuator/transmission mapping、控制和动作缩放；
- GMR/人体映射和 Motion Compiler plugin；
- Isaac task 注册和观测/动作 schema；
- 厂商 sim2sim adapter 和阈值；
- adapter contract/integration tests。

不得修改通用 Project、Asset、Motion、Run、Artifact、SSE 和下载协议。若机器人 DoF、控制模式或仿真规则不兼容，适配器必须显式声明能力，而不是在通用代码中加入 `if robot == ...`。

### 20.2 用户自定义任务语义

在 G1 imitation 闭环稳定后，新增 `TaskSpec`、`SceneSpec`、任务对象状态和注册奖励 term，支持搬箱、推门、起身等任务。用户只提交 JSON 场景、目标、奖励参数和终止条件，不提交代码。安全终止、控制周期、物理边界和 sim2sim 硬阈值仍由平台/仿真工程师维护。

## 21. 风险、未决项和关闭条件

| 风险/未决项 | 关闭条件 |
| --- | --- |
| GVHMR 许可证限制 | 在项目 manifest 登记研究/教育限制；商业用途前取得授权或接入替代后端 |
| Unitree RL Lab source SHA 缺失 | 运行环境采集并写入 manifest，缺失时不发布 |
| Isaac Sim source/build identity 缺失 | 记录 pip、容器和 source/build 对应关系 |
| Unitree MuJoCo 未提交 overlay | 形成独立 commit 或 patch manifest |
| GMR/Mink 与 MuJoCo 版本兼容性 | 在 GMR worker 中完成固定 fixture 和版本锁定测试 |
| 双卡并发过载 | GPU 采集器、lease、OOM 回收和压测通过 |
| 任意动作质量差 | 输入质量门禁、失败分类和用户可见诊断完善 |
| sim2sim 阈值过严/过松 | 用至少三类动作 fixture 校准，并通过仿真工程师审阅 |
| 浏览器 WASM viewer 延期 | 首期后端 PNG API 保持稳定的 `RendererProvider` 协议 |

## 22. 交付检查表

开发团队提交一个可验收版本时，必须附：

- 目录和模块依赖图；
- schema、数据库迁移、OpenAPI 和错误码快照；
- 本地 Compose 启动说明；
- Windows WSL2/Linux `robotlab doctor` 检查输出、GPU 映射和 lockfile；
- G1 适配器资产与 hash；
- 视频和两类直接输入 fixture 的运行记录；
- 训练、导出和三 seed sim2sim 结果；
- 策略包、校验和、manifest 和许可证清单；
- 单元、契约、集成和端到端测试命令及结果；
- 已知问题、失败 Run 和回滚步骤。

最终执行原则：先把通用 RobotSpec 在 G1 上的数据契约、动作处理、训练、导出和 sim2sim 做成可重复的本地闭环，再通过同一范式扩展其他具有关节的机器人和任务。任何“为了先跑通”而绕过版本、坐标、关节、许可证、manifest 或安全终止检查的实现都不算完成。
