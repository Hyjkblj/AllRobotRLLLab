# Motion Lab React 工作台

## 启动

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
