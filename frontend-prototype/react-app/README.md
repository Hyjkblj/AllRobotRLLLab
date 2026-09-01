# AllRobotRLLLab React 工作台

当前前端包含两层：默认打开的 Platform Workbench 负责项目、资源、动作检测、奖励配置、Run 创建和 SSE 监控；MuJoCo Motion Editor 负责真实 G1 网格的动作预览与局部编辑。两者通过明确的按钮切换，平台页面只调用 FastAPI `/api/v1`，编辑器只调用 `/api/mujoco`。

## API 代理

Vite 开发服务将请求分流到两个后端：

```text
/api/v1/*     → FastAPI Platform API（默认 http://127.0.0.1:8010）
/api/mujoco/* → MuJoCo Renderer（默认 http://127.0.0.1:8787）
/uploads/*    → Local File Mode 上传适配器
/objects/*    → Local File Mode 下载适配器
```

可通过环境变量覆盖目标地址：

```bash
VITE_PLATFORM_API_TARGET=http://127.0.0.1:8010 \
VITE_MUJOCO_API_TARGET=http://127.0.0.1:8787 \
npm run dev -- --host 127.0.0.1 --port 4173
```

生产静态部署使用反向代理将 `/api/v1`、`/api/mujoco`、`/uploads` 和 `/objects` 转发到对应服务；浏览器端不写死服务器文件系统路径。

## 启动

服务器上推荐从仓库根目录使用统一编排脚本。脚本假定当前 shell 已完成
Conda 初始化，但不会执行 `source conda.sh`、激活环境或安装依赖：

```bash
cd ~/AllRobotRLLLab
bash scripts/start_local_stack.sh
```

默认会在同一个终端启动 FastAPI + Local File worker、MuJoCo 离屏服务和
React/Vite，并等待三个 HTTP 探活通过。浏览器访问 `http://127.0.0.1:4173`；
SSH 转发时只需转发 `4173`，前端通过 Vite 代理访问 `8010` 和 `8787`。
按 `Ctrl-C` 会停止所有进程并保留 `runtime/` 数据。需要在另一个终端停止时：

```bash
bash scripts/stop_local_stack.sh
```

环境名和端口可以通过参数或环境变量覆盖。服务器只有一个包含 Node.js 的
项目环境时，默认配置即可；若 Node.js 安装在 `allrobotrl-platform`：

```bash
ROBOTLAB_NODE_ENV=allrobotrl-platform bash scripts/start_local_stack.sh
```

常用覆盖示例：

```bash
bash scripts/start_local_stack.sh \
  --api-env unitree_g1_train \
  --mujoco-env allrobotrl-mujoco \
  --node-env allrobotrl-mujoco \
  --runtime-dir "$PWD/runtime" \
  --api-port 8010 --mujoco-port 8787 --frontend-port 4173
```

脚本不会自动执行 `npm ci`。首次使用前，请在已初始化 Conda 的 shell 中完成
依赖安装：

```bash
conda run -n allrobotrl-mujoco npm ci --prefix frontend-prototype/react-app
```

日志位于 `${ROBOTLAB_RUNTIME_DIR:-./runtime}/processes/`；端口已被占用时，
脚本会直接退出并提示先执行停止命令。

在服务器上使用项目的 `allrobotrl-platform` Conda 环境管理 Node.js，不依赖系统 Node.js：

```bash
source /ai/python/miniconda3/etc/profile.d/conda.sh
conda activate allrobotrl-platform
node --version  # v20.x
npm --version
```

如果该环境已经存在但没有 Node.js：

```bash
source /ai/python/miniconda3/etc/profile.d/conda.sh
conda install -n allrobotrl-platform -c conda-forge nodejs=20 -y
conda activate allrobotrl-platform
```

服务器若只有已经创建好的 `allrobotrl-mujoco` 环境，可直接在该项目环境
中安装同一版本的 Node.js，用于当前 MuJoCo 原型和 React 前端：

```bash
source /ai/python/miniconda3/etc/profile.d/conda.sh
conda install -n allrobotrl-mujoco -c conda-forge nodejs=20 -y
conda activate allrobotrl-mujoco
node --version
npm --version
```

启动 MuJoCo 服务仍使用独立的 `allrobotrl-mujoco` 环境。前端在 `allrobotrl-platform` 环境中启动：

```bash
source /ai/python/miniconda3/etc/profile.d/conda.sh
conda activate allrobotrl-mujoco
cd ~/AllRobotRLLLab
python -m pip install -r frontend-prototype/requirements-mujoco.txt
python frontend-prototype/mujoco_service.py
```

另开一个终端启动 React：

```bash
source /ai/python/miniconda3/etc/profile.d/conda.sh
conda activate allrobotrl-platform
cd ~/AllRobotRLLLab/frontend-prototype/react-app
npm ci
npm run dev -- --host 127.0.0.1 --port 4173
```

Windows 本地开发仍可直接使用 PowerShell：

```powershell
conda activate allrobotrl-mujoco
python frontend-prototype/mujoco_service.py
```

另开一个终端：

```powershell
conda activate allrobotrl-platform
cd frontend-prototype/react-app
npm install --cache D:\npm-cache
npm run dev -- --host 127.0.0.1
```

也可以从项目根目录执行 `frontend-prototype/start_motion_lab.ps1`，它会使用
`D:\conda\envs\allrobotrl-platform\python.exe` 启动 MuJoCo，并在 `4173` 启动 Vite。

## 数据约定

- 服务模型固定优先使用 `third_party/GMR-master/assets/unitree_g1/g1_mocap_29dof.xml`。
- URDF 元数据来自同目录的 `g1_custom_collision_29dof.urdf`。
- `UnitreeG1Dance` 下包含 `qpos` 序列的 `.npz/.csv` 可直接逐帧播放和编辑。
- 现有 `.pt` 文件主要是 TorchScript policy 或训练 checkpoint，不是动作帧；工作台会识别并展示它们。若 `.pt` 内含形如 `[T, 36]`、`[T, 29]` 的 qpos/action tensor，在环境中安装 Torch 后即可被服务解码。

## API

Platform Workbench motion processing:

- `POST /api/v1/motions/{asset_version_id}/process` (async by default; use `X-Execution-Mode: sync_smoke` for local verification)
- `GET /api/v1/motions/{asset_version_id}/pipeline`
- `GET /api/v1/motion-pipelines/{pipeline_id}`

- `GET /api/mujoco/health`
- `GET /api/mujoco/model`
- `GET /api/mujoco/urdf`
- `GET /api/mujoco/actions`
- `POST /api/mujoco/session/reset`
- `GET /api/mujoco/actions/{id}/frames/{frame}`
- `POST /api/mujoco/actions/{id}/frames/{frame}/joints`
- `POST /api/mujoco/actions/{id}/keyframes`
- `POST /api/mujoco/export`

## 交互与无头部署

- 在真实 MuJoCo 视图内按住鼠标左键拖拽可旋转相机，滚轮可缩放；时间轴播放会自动取消已经过期的帧请求，只提交最新姿态。
- 服务端使用离屏 `mujoco.Renderer`，不需要也不会尝试嵌入 MuJoCo 原生 GLFW 桌面窗口。HTTP 服务按单渲染工作线程复用 OpenGL context，避免快速拖动时的 WGL/EGL context 冲突。
- Windows 开发环境：

```powershell
$env:MOTIONLAB_RENDER_BACKEND="glfw"
python frontend-prototype/mujoco_service.py
```

- Linux + NVIDIA GPU 无头服务器优先使用 EGL：

```bash
export MOTIONLAB_RENDER_BACKEND=egl
export MOTIONLAB_MUJOCO_HOST=0.0.0.0
python frontend-prototype/mujoco_service.py
```

- Linux 无 GPU 的软件渲染使用 OSMesa（系统需安装 OSMesa 运行库）：

```bash
export MOTIONLAB_RENDER_BACKEND=osmesa
export MOTIONLAB_MUJOCO_HOST=0.0.0.0
python frontend-prototype/mujoco_service.py
```

`GET /api/mujoco/health` 会返回 `renderBackend`、`headless` 和 `renderReady`，用于部署探活。EGL/OSMesa 只负责服务器离屏渲染，浏览器仍通过 React 视图显示 PNG；如果要在浏览器中运行真正的 WebGL MuJoCo viewer，需要另行构建 Emscripten/WASM 版本。

## 真实 3D 渲染说明

工作台中间的视图不是 Three.js 机器人占位几何，而是服务端 MuJoCo
`Renderer` 对 `third_party/GMR-master/assets/unitree_g1/g1_mocap_29dof.xml`
及其 STL 网格进行离屏渲染后返回的 PNG。每次切换动作帧或提交关节覆盖都会
重新使用同一个 `MjModel/MjData` 执行 `mj_forward` 后渲染，右下角按钮控制
MuJoCo 相机参数。

MuJoCo 原生 GLFW viewer 是桌面窗口，不能作为浏览器 DOM 直接嵌入；当前实现
采用 MuJoCo 官方 Python Renderer 的浏览器嵌入方式。若后续需要浏览器内真正的
实时 WebGL 交互，可基于仓库的 `third_party/mujoco-main/wasm` 构建 MuJoCo
WASM viewer，但它与当前 Conda Python 服务是另一条运行时链路。
