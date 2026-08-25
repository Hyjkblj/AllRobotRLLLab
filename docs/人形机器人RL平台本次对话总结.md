# 人形机器人 RL 平台本次对话总结

## 1. 用户目标

目标是建设一套面向多厂家人形机器人的 Web RL 训练平台：用户上传视频或动作资源，完成动作恢复与机器人重定向，选择机器人和任务，配置奖励，启动仿真训练，最后在 sim2sim 环节验收并导出可部署策略。

首个示例任务是“搬运小箱子/大箱子”，但架构不能只服务于 Unitree G1。

## 2. 讨论形成的技术链路

```text
视频/动作资源
  → GVHMR：视频恢复人体运动
  → GMR：人体到机器人运动重定向
  → Motion Compiler：重采样、坐标/四元数转换、质量校验
  → Isaac Lab + Isaac Sim：并行 RL、模仿学习和任务训练
  → 策略导出：JIT/ONNX + deploy.yaml + manifest
  → Unitree MuJoCo 或厂商 Sim Adapter：sim2sim 验收
  → 可下载策略包
```

## 3. 关键纠正

### 3.1 Unitree RL Lab 的定位

最初把 Unitree RL Lab 写成了训练底座，后面明确纠正为：

- 训练底座是 Isaac Lab + Isaac Sim。
- Unitree RL Lab 只参考其任务注册、配置分层、训练、play、export 和部署配置组织形式。
- Unitree G1 是第一个机器人适配器，不应把 Unitree 专属 task id 和 DDS 逻辑写进通用核心。

### 3.2 Windows 与服务器

当前只需要本地源码开发，后续部署到导师 GPU 服务器。

- Isaac Lab/Isaac Sim 官方源码支持 Windows 11，但完整链路仍以 Linux GPU 服务器为验证环境。
- 导师服务器是 Ubuntu 22.04、双 RTX 4090。
- 本机主要开发 React、FastAPI、数据契约、适配器和配置生成，不把 Windows 作为正式训练验收环境。

## 4. 服务器已确认事实

服务器 Conda 环境：`unitree_g1_train`。

```text
IsaacLab Git tag: v2.3.0
IsaacLab Git commit: 3c6e67bb5
IsaacLab Python package: 0.47.2
IsaacSim: 5.1.0.0
IsaacSim app: 5.1.0.0
Python: 3.11
Torch: 2.7.0+cu128
Unitree RL Lab package: 0.2.1
ISAACLAB_PATH: /ai/users/huangwy/G1/IsaacLab
```

`isaaclab` 的 `0.47.2` 是 Python 包内部版本，不能取代 Git tag `v2.3.0`。当前运行基线应锁定为：

```text
Isaac Lab v2.3.0 + Isaac Sim 5.1.0.0 + Python 3.11
```

服务器环境中没有 Python `mujoco` 包；这不一定阻止 C++ Unitree MuJoCo，但 GMR Python 可视化和 Python 版 Unitree MuJoCo 需要独立环境。

## 5. 本地仓库和代码事实

### 5.1 GMR/GVHMR

- GMR 本地 remote：`YanjieZe/GMR`，当前 commit `bb1bbe4`，README badge 为 `0.2.0`。
- GVHMR 本地 remote：`zju3dv/GVHMR`，当前 commit `6ec3ca3`。
- GVHMR 安装要求 Python 3.10、Torch 2.3.0+cu121、PyTorch3D 等，不能直接和 Isaac Lab 环境混装。
- GMR 内部使用 MuJoCo/mink IK，存在 `wxyz` 与 `xyzw`、坐标系和关节顺序转换边界。

### 5.2 Unitree RL Lab

- 本地 remote：`unitreerobotics/unitree_rl_lab`。
- 本地 Git describe：`0.2.1-11-g4960b84`。
- 服务器只给出了 Python 包版本 `0.2.1`，服务器源码 Git SHA 仍需补采集。
- 当前任务包括 mimic、velocity 和箱子操作配置，可借鉴其 reward、command、termination 和 deploy schema。

### 5.3 Unitree MuJoCo

- 本地上游基线 commit：`ae6a840`。
- Unitree MuJoCo 不是 MuJoCo 引擎 fork，而是原生 MuJoCo 加上 Unitree SDK2/DDS bridge、Unitree MJCF、控制映射、状态发布、虚拟挂带和 sim2real 示例。
- 控制命令大致映射为：`tau + kp * (q_cmd - q) + kd * (dq_cmd - dq)`。
- G1 XML 定义 29 个执行器、力矩范围、关节/IMU/速度/力矩传感器；bridge 依赖特定传感器名称和顺序。
- 当前工作区有未提交 overlay：软件摇杆、弹簧挂带渐变释放、调试输出、G1 摩擦/质量测试场景。

## 6. PRD 已确定的产品设计

### 前端

- React + TypeScript + Vite。
- 页面包括项目、上传、动作预览、机器人/任务选择、Reward Builder、训练监控、sim2sim 验收和策略包。
- Reward Builder 只允许注册的奖励项，保存完整 `RewardConfig`，不允许上传任意 Python。

### 后端

- FastAPI API + GPU worker + 文件/对象存储 + 元数据数据库。
- API 不能直接导入 Isaac Sim，也不能在请求线程运行训练。
- Worker 通过显式命令、独立工作目录和 Run Manifest 调用 GVHMR、GMR、Isaac Lab、MuJoCo。
- 每个运行保存日志、版本、GPU、配置、checkpoint、视频和 SHA-256。

### 核心数据契约

- `RobotSpec`
- `RetargetMotion`
- `TrainMotionNPZ`
- `RewardConfig`
- `TrainingConfig`
- `PolicyBundle`
- `RunManifest`

### 任务策略

奖励分为通用稳定项、参考跟踪项、任务语义项和安全终止项。

箱子任务采用阶段式路线：

```text
GoToBox → AlignToBox → ReachBox → HugBox → LiftBox
→ HoldBox → TurnWithBox → WalkWithBox → PlaceBox → ReleaseAndStepBack
```

MVP 只先验收 `ReachBox` 和 `HugBox`，避免从随机初始化直接训练完整搬运。

## 7. 已识别风险

- GVHMR 许可证当前限制教育、研究和非营利用途。
- 单目视频无法稳定恢复遮挡、接触和物体 6D 位姿；箱体状态应由场景文件或表单提供。
- GMR 的坐标系、四元数和身体映射错误会造成看似能训练但无法 sim2sim 的策略。
- Isaac Sim、MuJoCo、Unitree MuJoCo 的 XML/USD、PD、控制频率和动作缩放不一致会造成迁移失败。
- Unitree MuJoCo 的 sensor 顺序、DDS topic 和 `m->nu > 20` 判断不适合直接泛化到其它厂家。
- 奖励可能被投机，必须联合使用分项回报、成功率、跌倒率、动作饱和率、视频和目标误差验收。
- 双 RTX 4090 应先单卡 smoke test，再验证多 GPU；两张卡不是统一显存池。
- 当前 Unitree MuJoCo overlay 未提交，无法保证团队成员复现同一 sim2sim 环境。

## 8. 待完成事项

1. 采集服务器 Unitree RL Lab 的 Git SHA。
2. 采集 Isaac Sim pip 包对应的 source/build identity。
3. 将当前 Unitree MuJoCo overlay 提交为独立项目 commit，并写入 manifest。
4. 为 GMR 和 GVHMR 建立独立 lockfile 或容器。
5. 完成 G1 29 DoF 动作顺序、sensor 顺序、PD 参数和部署配置的自动化一致性检查。
6. 完成 Isaac Lab playback 与 Unitree MuJoCo 之间的固定初始状态对比。
7. 设计文件权限、GPU 配额、作业取消、OOM、超时和重试策略。
8. 在第二个机器人适配器接入前，先稳定 G1 的完整闭环。

## 9. 当前结论

可行的通用范式不是让所有机器人共用一套权重，而是复用：

```text
统一数据契约
+ 机器人适配器
+ Isaac Lab 任务/奖励模板
+ Isaac Sim 训练运行时
+ 厂商 sim2sim 适配器
+ 版本与产物 manifest
```

第一阶段应把 G1 做成可复现、可验收、可导出的参考适配器，再把同一套接口扩展到其它人形机器人。
