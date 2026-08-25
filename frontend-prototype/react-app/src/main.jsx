import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  Activity,
  Archive,
  ArrowDownToLine,
  BookmarkPlus,
  Check,
  ChevronDown,
  CircleHelp,
  Download,
  FileCog,
  FileUp,
  Gauge,
  Layers3,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Waypoints,
  X
} from "lucide-react";
import "./styles.css";

const API = "/api/mujoco";

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `请求失败 (${response.status})`);
  return body;
}

function formatBytes(value) {
  if (!value) return "—";
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function shortName(value, length = 36) {
  return value && value.length > length ? `${value.slice(0, length - 1)}…` : value || "未命名";
}

function angleFor(pose, name) {
  return pose?.joints?.find((joint) => joint.name === name)?.angleDeg || 0;
}

// Legacy prototype helpers remain isolated for reference. The rendered
// workspace below uses MuJoCoViewport and never mounts these meshes.
function createPart(length, width, depth, material) {
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, length, depth), material);
  mesh.position.y = -length / 2;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
  return group;
}

function createRobotScene() {
  const normal = new THREE.MeshStandardMaterial({ color: "#697782", metalness: 0.72, roughness: 0.29 });
  const dark = new THREE.MeshStandardMaterial({ color: "#25313b", metalness: 0.8, roughness: 0.2 });
  const accent = new THREE.MeshStandardMaterial({ color: "#3286d7", emissive: "#0b3764", emissiveIntensity: 0.42, metalness: 0.62, roughness: 0.22 });
  const root = new THREE.Group();
  const nodes = {};

  const pelvis = new THREE.Group();
  pelvis.position.set(0, 1.26, 0);
  root.add(pelvis);
  const pelvisMesh = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.28, 0.34), dark);
  pelvisMesh.position.y = -0.14;
  pelvisMesh.castShadow = true;
  pelvis.add(pelvisMesh);

  const torso = createPart(0.92, 0.66, 0.38, normal);
  torso.position.y = 0.04;
  pelvis.add(torso);
  const chest = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.28, 0.4), dark);
  chest.position.y = -0.12;
  torso.add(chest);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.13, 0.13, 20), dark);
  neck.position.y = 0.53;
  torso.add(neck);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 22, 16), normal);
  head.scale.set(0.86, 1.1, 0.86);
  head.position.y = 0.78;
  torso.add(head);

  function makeLeg(side) {
    const sign = side === "left" ? -1 : 1;
    const hip = new THREE.Group();
    hip.position.set(sign * 0.18, -0.14, 0);
    pelvis.add(hip);
    nodes[`${side}_hip_pitch_joint`] = hip;
    nodes[`${side}_hip_roll_joint`] = hip;
    nodes[`${side}_hip_yaw_joint`] = hip;
    const thigh = createPart(0.66, 0.2, 0.2, normal);
    hip.add(thigh);
    const knee = new THREE.Group();
    knee.position.y = -0.66;
    hip.add(knee);
    nodes[`${side}_knee_joint`] = knee;
    knee.add(createPart(0.62, 0.17, 0.17, normal));
    const ankle = new THREE.Group();
    ankle.position.y = -0.62;
    knee.add(ankle);
    nodes[`${side}_ankle_pitch_joint`] = ankle;
    nodes[`${side}_ankle_roll_joint`] = ankle;
    const foot = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.12, 0.4), dark);
    foot.position.set(0, -0.08, 0.08);
    ankle.add(foot);
  }

  function makeArm(side) {
    const sign = side === "left" ? -1 : 1;
    const shoulder = new THREE.Group();
    shoulder.position.set(sign * 0.43, 0.4, 0);
    torso.add(shoulder);
    nodes[`${side}_shoulder_pitch_joint`] = shoulder;
    nodes[`${side}_shoulder_roll_joint`] = shoulder;
    nodes[`${side}_shoulder_yaw_joint`] = shoulder;
    shoulder.add(createPart(0.48, 0.15, 0.15, normal));
    const elbow = new THREE.Group();
    elbow.position.y = -0.48;
    shoulder.add(elbow);
    nodes[`${side}_elbow_joint`] = elbow;
    elbow.add(createPart(0.45, 0.13, 0.13, normal));
    const wrist = new THREE.Group();
    wrist.position.y = -0.45;
    elbow.add(wrist);
    nodes[`${side}_wrist_roll_joint`] = wrist;
    nodes[`${side}_wrist_pitch_joint`] = wrist;
    nodes[`${side}_wrist_yaw_joint`] = wrist;
    const hand = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.24, 0.12), dark);
    hand.position.y = -0.13;
    wrist.add(hand);
  }

  makeLeg("left");
  makeLeg("right");
  makeArm("left");
  makeArm("right");
  nodes.waist_yaw_joint = pelvis;
  nodes.waist_roll_joint = pelvis;
  nodes.waist_pitch_joint = pelvis;

  return { root, nodes, materials: { normal, dark, accent } };
}

function applyPose(rig, pose, selectedJoint) {
  if (!rig || !pose) return;
  const radians = (name) => THREE.MathUtils.degToRad(angleFor(pose, name));
  const set = (name, axis, value) => {
    const node = rig.nodes[name];
    if (node) node.rotation[axis] = value;
  };
  set("left_hip_pitch_joint", "x", radians("left_hip_pitch_joint"));
  set("right_hip_pitch_joint", "x", radians("right_hip_pitch_joint"));
  set("left_hip_roll_joint", "z", -radians("left_hip_roll_joint"));
  set("right_hip_roll_joint", "z", radians("right_hip_roll_joint"));
  set("left_hip_yaw_joint", "y", radians("left_hip_yaw_joint"));
  set("right_hip_yaw_joint", "y", radians("right_hip_yaw_joint"));
  set("left_knee_joint", "x", radians("left_knee_joint"));
  set("right_knee_joint", "x", radians("right_knee_joint"));
  set("left_ankle_pitch_joint", "x", radians("left_ankle_pitch_joint"));
  set("right_ankle_pitch_joint", "x", radians("right_ankle_pitch_joint"));
  set("left_ankle_roll_joint", "z", radians("left_ankle_roll_joint"));
  set("right_ankle_roll_joint", "z", radians("right_ankle_roll_joint"));
  set("left_shoulder_pitch_joint", "x", radians("left_shoulder_pitch_joint"));
  set("right_shoulder_pitch_joint", "x", radians("right_shoulder_pitch_joint"));
  set("left_shoulder_roll_joint", "z", -radians("left_shoulder_roll_joint"));
  set("right_shoulder_roll_joint", "z", radians("right_shoulder_roll_joint"));
  set("left_shoulder_yaw_joint", "y", radians("left_shoulder_yaw_joint"));
  set("right_shoulder_yaw_joint", "y", radians("right_shoulder_yaw_joint"));
  set("left_elbow_joint", "x", radians("left_elbow_joint"));
  set("right_elbow_joint", "x", radians("right_elbow_joint"));

  Object.values(rig.nodes).forEach((node) => node.traverse((child) => {
    if (child.material === rig.materials.accent) child.material = rig.materials.normal;
  }));
  const selectedNode = rig.nodes[selectedJoint];
  if (selectedNode) selectedNode.traverse((child) => { if (child.material) child.material = rig.materials.accent; });
}

function ThreeViewport({ pose, selectedJoint }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#edf0f2");
    scene.fog = new THREE.Fog("#edf0f2", 5, 13);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    camera.position.set(3.4, 2.45, 4.5);
    camera.lookAt(0, 1.1, 0);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 1.15, 0);
    controls.minDistance = 2.2;
    controls.maxDistance = 8;
    const hemi = new THREE.HemisphereLight("#ffffff", "#c4cbd1", 2.1);
    scene.add(hemi);
    const key = new THREE.DirectionalLight("#ffffff", 3.4);
    key.position.set(3, 6, 4);
    key.castShadow = true;
    scene.add(key);
    const rim = new THREE.DirectionalLight("#83b8ed", 1.7);
    rim.position.set(-4, 3, -3);
    scene.add(rim);
    const grid = new THREE.GridHelper(8, 32, "#c7cfd5", "#dfe3e6");
    grid.position.y = -1.1;
    scene.add(grid);
    const rig = createRobotScene();
    rig.root.position.y = 0.03;
    scene.add(rig.root);
    const ground = new THREE.Mesh(new THREE.CircleGeometry(2.3, 64), new THREE.MeshStandardMaterial({ color: "#e0e5e8", transparent: true, opacity: 0.72 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1.08;
    ground.receiveShadow = true;
    scene.add(ground);
    sceneRef.current = { scene, camera, renderer, controls, rig };
    const resize = () => {
      const width = mount.clientWidth || 800;
      const height = mount.clientHeight || 560;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    let animation;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      animation = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(animation);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      scene.traverse((node) => { if (node.geometry) node.geometry.dispose(); });
    };
  }, []);
  useEffect(() => {
    if (sceneRef.current) applyPose(sceneRef.current.rig, pose, selectedJoint);
  }, [pose, selectedJoint]);
  return <div className="viewport" ref={mountRef}><div className="viewport-hint"><span className="live-dot" /> MuJoCo forward kinematics</div><div className="viewport-axis">X <i /> Y <i /> Z <i /></div></div>;
}

function MuJoCoViewport({ assetId, frame, pose }) {
  const [imageUrl, setImageUrl] = useState("");
  const [renderError, setRenderError] = useState("");
  const [rendering, setRendering] = useState(false);
  const [dragging, setDragging] = useState(false);
  const cameraRef = useRef({ azimuth: 135, elevation: -25, distance: 2.35 });
  const dragRef = useRef(null);
  const imageUrlRef = useRef("");
  const renderQueueRef = useRef(null);
  const activeRenderRef = useRef(false);
  const renderControllerRef = useRef(null);
  const dragRenderTimerRef = useRef(null);
  const generationRef = useRef(0);

  const pumpRenderQueue = () => {
    if (activeRenderRef.current || !renderQueueRef.current) return;
    const job = renderQueueRef.current;
    renderQueueRef.current = null;
    activeRenderRef.current = true;
    const controller = new AbortController();
    renderControllerRef.current = controller;
    const query = new URLSearchParams({
      asset: job.assetId,
      frame: String(job.frame),
      width: "900",
      height: "720",
      azimuth: String(job.camera.azimuth),
      elevation: String(job.camera.elevation),
      distance: String(job.camera.distance)
    });
    setRendering(true);
    fetch(`${API}/render?${query.toString()}`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`MuJoCo 渲染失败 (${response.status})`);
        return response.blob();
      })
      .then((blob) => {
        if (job.generation !== generationRef.current) return;
        const nextUrl = URL.createObjectURL(blob);
        if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
        imageUrlRef.current = nextUrl;
        setImageUrl(nextUrl);
        setRenderError("");
      })
      .catch((error) => {
        if (error.name !== "AbortError" && job.generation === generationRef.current) setRenderError(error.message);
      })
      .finally(() => {
        activeRenderRef.current = false;
        renderControllerRef.current = null;
        if (!controller.signal.aborted && !renderQueueRef.current) setRendering(false);
        pumpRenderQueue();
      });
  };

  const queueRender = (nextFrame = frame) => {
    if (!assetId) return;
    renderQueueRef.current = {
      assetId,
      frame: nextFrame,
      camera: { ...cameraRef.current },
      generation: generationRef.current
    };
    pumpRenderQueue();
  };

  const scheduleCameraRender = (flush = false) => {
    if (dragRenderTimerRef.current) window.clearTimeout(dragRenderTimerRef.current);
    if (flush) {
      queueRender();
      return;
    }
    // Coalesce pointer events. The service is intentionally serialized to
    // keep MuJoCo's GL context thread-affine; only the newest camera matters.
    dragRenderTimerRef.current = window.setTimeout(() => queueRender(), 45);
  };

  useEffect(() => {
    generationRef.current += 1;
    renderQueueRef.current = null;
    renderControllerRef.current?.abort();
    cameraRef.current = { azimuth: 135, elevation: -25, distance: 2.35 };
    if (imageUrlRef.current) {
      URL.revokeObjectURL(imageUrlRef.current);
      imageUrlRef.current = "";
    }
    setImageUrl("");
    setRenderError("");
    if (assetId) queueRender(frame);
    return () => {
      generationRef.current += 1;
      renderQueueRef.current = null;
      renderControllerRef.current?.abort();
    };
  }, [assetId]);

  useEffect(() => {
    queueRender(frame);
  }, [frame, pose?.overrideCount]);

  useEffect(() => () => {
    if (dragRenderTimerRef.current) window.clearTimeout(dragRenderTimerRef.current);
    renderControllerRef.current?.abort();
    if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
  }, []);

  const handlePointerDown = (event) => {
    if (event.button !== 0 || event.target.closest("button")) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, ...cameraRef.current };
    setDragging(true);
  };

  const handlePointerMove = (event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    cameraRef.current.azimuth = drag.azimuth - (event.clientX - drag.x) * 0.45;
    cameraRef.current.elevation = Math.max(-89, Math.min(89, drag.elevation + (event.clientY - drag.y) * 0.35));
    scheduleCameraRender();
  };

  const endPointerDrag = (event) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setDragging(false);
    scheduleCameraRender(true);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const handleWheel = (event) => {
    event.preventDefault();
    cameraRef.current.distance = Math.max(1.5, Math.min(8, cameraRef.current.distance * Math.exp(event.deltaY * 0.0015)));
    scheduleCameraRender();
  };

  const rotateCamera = (delta) => {
    cameraRef.current.azimuth += delta;
    scheduleCameraRender(true);
  };

  const zoomCamera = (delta) => {
    cameraRef.current.distance = Math.max(1.5, Math.min(8, cameraRef.current.distance + delta));
    scheduleCameraRender(true);
  };

  return <div className={`viewport real-viewport ${dragging ? "grabbing" : ""}`} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={endPointerDrag} onPointerCancel={endPointerDrag} onWheel={handleWheel}>
    {imageUrl ? <img className="mujoco-render" src={imageUrl} alt="MuJoCo 渲染的 Unitree G1 真实网格" /> : <div className="render-loading"><RefreshCw size={18} className="spin" /><span>{renderError || "正在初始化 MuJoCo 渲染器"}</span></div>}
    <div className="viewport-hint"><span className="live-dot" /> MuJoCo Renderer · G1 mesh</div>
    <div className="render-source-tag">真实 MJCF 网格 · {pose?.qpos?.length || 36} qpos</div>
    {rendering && imageUrl && <div className="render-progress"><RefreshCw size={12} className="spin" /></div>}
    <div className="render-controls" aria-label="MuJoCo 相机控制"><button className="icon-button small" title="向左旋转" onClick={() => rotateCamera(-12)}>↶</button><button className="icon-button small" title="向右旋转" onClick={() => rotateCamera(12)}>↷</button><button className="icon-button small" title="拉近" onClick={() => zoomCamera(-.25)}>＋</button><button className="icon-button small" title="拉远" onClick={() => zoomCamera(.25)}>－</button></div>
  </div>;
}

function Badge({ children, tone = "neutral" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function App() {
  const [health, setHealth] = useState(null);
  const [model, setModel] = useState(null);
  const [urdf, setUrdf] = useState(null);
  const [assets, setAssets] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [pose, setPose] = useState(null);
  const [frame, setFrame] = useState(0);
  const [keyframes, setKeyframes] = useState([]);
  const [selectedJoint, setSelectedJoint] = useState("");
  const [angle, setAngle] = useState(0);
  const [velocity, setVelocity] = useState(1.8);
  const [stiffness, setStiffness] = useState(50);
  const [tab, setTab] = useState("joints");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [playing, setPlaying] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [localUrdf, setLocalUrdf] = useState(null);
  const [importPath, setImportPath] = useState("");
  const [loading, setLoading] = useState(true);
  const frameRequestRef = useRef(null);
  const selectionRequestRef = useRef(null);

  const selectedAsset = useMemo(() => assets.find((asset) => asset.id === selectedId) || null, [assets, selectedId]);
  const visibleAssets = useMemo(() => assets.filter((asset) => {
    const matchesFilter = filter === "all" || (filter === "motion" ? asset.kind === "motion" : asset.kind === "policy");
    const query = search.trim().toLowerCase();
    return matchesFilter && (!query || `${asset.name} ${asset.relativePath}`.toLowerCase().includes(query));
  }), [assets, filter, search]);
  const selectedSpec = model?.joints?.find((joint) => joint.name === selectedJoint) || model?.joints?.[0];
  const selectedAngle = pose ? angleFor(pose, selectedJoint) : angle;
  const totalFrames = selectedAsset?.frameCount || 121;

  const notify = (text) => { setMessage(text); window.setTimeout(() => setMessage(""), 2600); };
  const loadFrame = async (assetId, nextFrame) => {
    if (!assetId) return;
    frameRequestRef.current?.abort();
    const controller = new AbortController();
    frameRequestRef.current = controller;
    try {
      const nextPose = await api(`/actions/${assetId}/frames/${nextFrame}`, { signal: controller.signal });
      if (controller.signal.aborted) return;
      setPose(nextPose);
      setFrame(nextPose.frame);
      const current = nextPose.joints?.find((joint) => joint.name === selectedJoint);
      if (current) setAngle(Number(current.angleDeg.toFixed(2)));
      setError("");
    } catch (loadError) {
      if (loadError.name === "AbortError") return;
      setPose(null);
      setError(loadError.message);
    }
  };

  const refresh = async () => {
    setLoading(true);
    try {
      const [nextHealth, nextModel, nextUrdf, nextActions, nextKeyframes] = await Promise.all([
        api("/health"), api("/model"), api("/urdf"), api("/actions"), api("/keyframes")
      ]);
      setHealth(nextHealth);
      setModel(nextModel);
      setUrdf(nextUrdf);
      setKeyframes(nextKeyframes.keyframes || []);
      const nextAssets = nextActions.assets || [];
      setAssets(nextAssets);
      const currentId = selectedId && nextAssets.some((asset) => asset.id === selectedId) ? selectedId : nextAssets.find((asset) => asset.kind === "motion" && asset.frameCount > 0)?.id || nextAssets[0]?.id || "";
      setSelectedId(currentId);
      setSelectedJoint((current) => current || nextModel.joints?.[0]?.name || "");
      setError("");
    } catch (loadError) {
      setError(`无法连接 MuJoCo 服务：${loadError.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, []);
  useEffect(() => { if (selectedId && selectedAsset?.frameCount > 0) loadFrame(selectedId, frame); }, [selectedId, frame]);
  useEffect(() => () => frameRequestRef.current?.abort(), []);
  useEffect(() => {
    if (!playing || !selectedAsset?.frameCount) return undefined;
    const sourceFps = Number(selectedAsset.fps) || 30;
    // Keep the UI responsive while preserving the source motion's time base.
    // At 30 FPS this renders 15 images/sec and advances two source frames.
    const playbackFps = Math.min(18, Math.max(8, sourceFps));
    const frameStep = Math.max(1, Math.round(sourceFps / playbackFps));
    const timer = window.setInterval(() => setFrame((current) => current + frameStep >= totalFrames ? 0 : current + frameStep), 1000 / playbackFps);
    return () => window.clearInterval(timer);
  }, [playing, selectedAsset, totalFrames]);

  const selectAsset = async (asset) => {
    if (asset.id === selectedId) return;
    frameRequestRef.current?.abort();
    selectionRequestRef.current?.abort();
    const controller = new AbortController();
    selectionRequestRef.current = controller;
    setFrame(0);
    setPlaying(false);
    setPose(null);
    setAngle(0);
    // Unmount the old viewport immediately. No old asset request may share
    // the new action's renderer session.
    setSelectedId("");
    try {
      await api("/session/reset", { method: "POST", body: JSON.stringify({ assetId: asset.id }), signal: controller.signal });
      if (controller.signal.aborted) return;
    } catch (resetError) {
      if (resetError.name !== "AbortError") setError(`MuJoCo 窗口重置失败：${resetError.message}`);
    }
    setSelectedId(asset.id);
    if (asset.kind === "policy" && !asset.frameCount) {
      setError("这是策略/检查点 .pt 文件。它已被识别，但没有可直接播放的 qpos 序列；请选择同目录的导出动作或 .npz 动作参考。");
    } else {
      setError("");
    }
  };

  const applyJoint = async () => {
    if (!selectedId || !selectedJoint) return;
    try {
      const result = await api(`/actions/${selectedId}/frames/${frame}/joints`, { method: "POST", body: JSON.stringify({ joint: selectedJoint, angleDeg: Number(angle) }) });
      setPose(result);
      notify(`${selectedJoint} 已应用到 F${String(frame).padStart(3, "0")}`);
    } catch (applyError) { setError(applyError.message); }
  };

  const resetJoint = async () => {
    if (!selectedId) return;
    try {
      const result = await api(`/actions/${selectedId}/frames/${frame}/joints`, { method: "DELETE" });
      setPose(result);
      setAngle(angleFor(result, selectedJoint));
      notify("当前帧覆盖已恢复");
    } catch (resetError) { setError(resetError.message); }
  };

  const addKeyframe = async () => {
    if (!selectedId) return;
    try {
      const result = await api(`/actions/${selectedId}/keyframes`, { method: "POST", body: JSON.stringify({ frame, label: `G1 · F${String(frame).padStart(3, "0")}` }) });
      setKeyframes((current) => [...current.filter((item) => item.id !== result.id), result].sort((a, b) => a.frame - b.frame));
      notify("关键帧已保存");
    } catch (keyframeError) { setError(keyframeError.message); }
  };

  const exportMotion = async () => {
    if (!selectedId) return;
    try {
      const frames = keyframes.filter((item) => item.assetId === selectedId).map((item) => item.frame);
      const result = await api("/export", { method: "POST", body: JSON.stringify({ assetId: selectedId, frame, frames, metadata: { selectedJoint, velocity, stiffness } }) });
      const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${selectedAsset?.name || "g1-motion"}-edited.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      notify("MuJoCo qpos 已导出");
    } catch (exportError) { setError(exportError.message); }
  };

  const importUrdf = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const xml = new DOMParser().parseFromString(String(reader.result), "application/xml");
      const joints = [...xml.querySelectorAll("joint")].filter((node) => node.getAttribute("type") !== "fixed").map((node) => ({ name: node.getAttribute("name"), type: node.getAttribute("type") }));
      setLocalUrdf({ name: file.name, joints });
      notify(`${file.name} 已解析，${joints.length} 个可动关节`);
    };
    reader.readAsText(file);
  };

  const registerPath = async () => {
    if (!importPath.trim()) return;
    try {
      const asset = await api("/actions/import", { method: "POST", body: JSON.stringify({ path: importPath.trim() }) });
      setAssets((current) => [...current.filter((item) => item.id !== asset.id), asset]);
      setImportPath("");
      notify(`${asset.fileName} 已加入动作目录`);
    } catch (importError) { setError(importError.message); }
  };

  const selectedKeyframes = keyframes.filter((item) => item.assetId === selectedId);

  return <div className="app-shell">
    <aside className="rail">
      <div className="brand-mark">ML</div>
      <div className="rail-line" />
      <button className="rail-button active" title="动作编辑"><Waypoints size={18} /></button>
      <button className="rail-button" title="模型资产"><Layers3 size={18} /></button>
      <button className="rail-button" title="训练任务"><Activity size={18} /></button>
      <div className="rail-spacer" />
      <button className="rail-button" title="设置"><Settings2 size={18} /></button>
      <div className="rail-user">HY</div>
    </aside>

    <main className="main-shell">
      <header className="topbar">
        <div className="crumb"><span className="crumb-muted">Motion Lab</span><span className="crumb-divider">/</span><strong>G1 Motion Editor</strong></div>
        <div className="top-actions">
          <div className={`service-state ${health ? "online" : "offline"}`}><span className="live-dot" /> {health ? `MuJoCo ${health.runtimeVersion}` : "服务离线"}</div>
          <button className="icon-button" title="刷新服务与动作目录" onClick={refresh}><RefreshCw size={16} className={loading ? "spin" : ""} /></button>
          <button className="button button-dark" onClick={exportMotion}><Download size={15} /> 导出动作</button>
        </div>
      </header>

      <section className="workspace-head">
        <div><p className="eyebrow">REAL ROBOT DATA / MUJOCO</p><h1>动作微调工作台</h1><p className="subhead">导入真实 G1 模型，定位关键帧，使用 MuJoCo 限位进行逐关节调整。</p></div>
        <div className="head-metrics"><div><span>模型</span><strong>G1 · 29 DoF</strong></div><div><span>qpos</span><strong>{model?.nq || "—"}</strong></div><div><span>资产</span><strong>{assets.length || "—"}</strong></div></div>
      </section>

      <div className="work-grid">
        <section className="asset-panel panel-surface">
          <div className="panel-heading"><div><span className="panel-kicker">MOTION LIBRARY</span><h2>动作源</h2></div><Badge tone="blue">UnitreeG1Dance</Badge></div>
          <div className="asset-toolbar"><div className="search-box"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索动作或路径" /></div><button className="icon-button small" title="刷新资产" onClick={refresh}><RefreshCw size={14} /></button></div>
          <div className="segmented"><button className={filter === "all" ? "selected" : ""} onClick={() => setFilter("all")}>全部</button><button className={filter === "motion" ? "selected" : ""} onClick={() => setFilter("motion")}>动作</button><button className={filter === "policy" ? "selected" : ""} onClick={() => setFilter("policy")}>策略 .pt</button></div>
          <div className="asset-list">
            {visibleAssets.slice(0, 80).map((asset) => <button key={asset.id} className={`asset-row ${asset.id === selectedId ? "selected" : ""}`} onClick={() => selectAsset(asset)}><span className={`asset-icon ${asset.kind}`}><FileCog size={15} /></span><span className="asset-copy"><strong>{shortName(asset.name, 30)}</strong><small>{shortName(asset.relativePath, 38)}</small></span><span className="asset-meta">{asset.kind === "motion" ? `${asset.frameCount || "—"} F` : ".pt"}<br /><small>{formatBytes(asset.sizeBytes)}</small></span></button>)}
            {!visibleAssets.length && <div className="empty-state">没有匹配资产</div>}
          </div>
          <div className="asset-footer"><span>{visibleAssets.length} 个资产</span><span className="asset-path">根目录 · {shortName("D:/Develop/Project/UnitreeG1Dance", 28)}</span></div>
        </section>

        <section className="editor-panel">
          <div className="editor-topline"><div className="selected-asset"><span className="selected-dot" /><div><strong>{selectedAsset ? shortName(selectedAsset.name, 44) : "等待动作资产"}</strong><span>{selectedAsset ? `${selectedAsset.relativePath} · ${selectedAsset.frameCount || "策略检查点"}` : "请选择一个动作或策略文件"}</span></div></div><div className="editor-actions"><Badge tone={pose ? "green" : "neutral"}>{pose ? "LIVE POSE" : "NO FRAME"}</Badge><button className="icon-button" title="帮助"><CircleHelp size={16} /></button></div></div>
          <MuJoCoViewport key={selectedId || "empty"} assetId={selectedId} frame={frame} pose={pose} />
          <div className="timeline-panel">
            <div className="timeline-head"><div className="timeline-title"><span>时间轴</span><strong>F{String(frame).padStart(3, "0")}</strong><small>{(frame / (selectedAsset?.fps || 30)).toFixed(2)} s</small></div><div className="timeline-controls"><button className="icon-button small" title="上一帧" onClick={() => setFrame((value) => Math.max(0, value - 1))}>‹</button><button className="play-button" title={playing ? "暂停" : "播放"} onClick={() => setPlaying((value) => !value)}>{playing ? <Pause size={15} /> : <Play size={15} fill="currentColor" />}</button><button className="icon-button small" title="下一帧" onClick={() => setFrame((value) => Math.min(totalFrames - 1, value + 1))}>›</button></div></div>
            <div className="timeline-track-wrap"><div className="timeline-track"><div className="timeline-progress" style={{ width: `${totalFrames > 1 ? (frame / (totalFrames - 1)) * 100 : 0}%` }} />{selectedKeyframes.map((item) => <button key={item.id} className="key-marker" title={item.label} style={{ left: `${totalFrames > 1 ? (item.frame / (totalFrames - 1)) * 100 : 0}%` }} onClick={() => setFrame(item.frame)} />)}<input type="range" min="0" max={Math.max(0, totalFrames - 1)} value={Math.min(frame, Math.max(0, totalFrames - 1))} onChange={(event) => setFrame(Number(event.target.value))} aria-label="动作帧" /></div><div className="timeline-scale"><span>F000</span><span>F{String(Math.round(totalFrames / 2)).padStart(3, "0")}</span><span>F{String(Math.max(0, totalFrames - 1)).padStart(3, "0")}</span></div></div>
          </div>
        </section>

        <aside className="inspector panel-surface">
          <div className="panel-heading"><div><span className="panel-kicker">INSPECTOR</span><h2>姿态检查</h2></div><Badge tone="silver">{model?.jointCount || 29} DoF</Badge></div>
          <div className="tabs"><button className={tab === "joints" ? "active" : ""} onClick={() => setTab("joints")}><SlidersHorizontal size={14} /> 关节</button><button className={tab === "keyframes" ? "active" : ""} onClick={() => setTab("keyframes")}><BookmarkPlus size={14} /> 关键帧</button><button className={tab === "source" ? "active" : ""} onClick={() => setTab("source")}><FileUp size={14} /> 导入</button></div>

          {tab === "joints" && <div className="inspector-body"><label className="field-label">当前关节</label><div className="select-wrap"><select value={selectedJoint} onChange={(event) => { setSelectedJoint(event.target.value); setAngle(angleFor(pose, event.target.value)); }}><option value="" disabled>选择关节</option>{(model?.joints || []).map((joint) => <option key={joint.name} value={joint.name}>{joint.name}</option>)}</select><ChevronDown size={14} /></div>{selectedSpec && <><div className="joint-context"><span className="joint-bullet" /><div><strong>{selectedJoint}</strong><small>MuJoCo joint · qpos[{selectedSpec.qposAdr}]</small></div></div><div className="angle-head"><label className="field-label">目标角度</label><output>{Number(selectedAngle).toFixed(1)}°</output></div><input className="range" type="range" min={selectedSpec.lowerDeg} max={selectedSpec.upperDeg} step="0.1" value={Number.isFinite(selectedAngle) ? selectedAngle : 0} onChange={(event) => setAngle(Number(event.target.value))} /><div className="number-line"><input type="number" value={Number.isFinite(angle) ? angle : 0} min={selectedSpec.lowerDeg} max={selectedSpec.upperDeg} step="0.1" onChange={(event) => setAngle(Number(event.target.value))} /><span>°</span><small>[{selectedSpec.lowerDeg.toFixed(1)}°, {selectedSpec.upperDeg.toFixed(1)}°]</small></div><div className="param-block"><div className="angle-head"><label className="field-label">速度上限</label><output>{velocity.toFixed(2)} rad/s</output></div><input className="range" type="range" min="0" max="8" step="0.05" value={velocity} onChange={(event) => setVelocity(Number(event.target.value))} /><div className="angle-head"><label className="field-label">执行刚度</label><output>{stiffness}%</output></div><input className="range" type="range" min="0" max="100" step="1" value={stiffness} onChange={(event) => setStiffness(Number(event.target.value))} /></div><div className="limit-row"><span>关节限位</span><Badge tone="green">ACTIVE</Badge></div><div className="action-buttons"><button className="button button-light" onClick={resetJoint}><RotateCcw size={14} /> 恢复本帧</button><button className="button button-blue" onClick={applyJoint}><Check size={14} /> 应用调整</button></div></>}</div>}

          {tab === "keyframes" && <div className="inspector-body"><div className="keyframe-intro"><span className="keyframe-icon"><BookmarkPlus size={17} /></span><div><strong>动作关键帧</strong><p>将当前 MuJoCo qpos 快照写入编辑会话。</p></div></div><button className="button button-blue wide" onClick={addKeyframe}><BookmarkPlus size={14} /> 保存 F{String(frame).padStart(3, "0")}</button><div className="keyframe-list">{selectedKeyframes.map((item) => <button key={item.id} className={`keyframe-row ${item.frame === frame ? "active" : ""}`} onClick={() => setFrame(item.frame)}><span className="marker-number">{String(item.frame).padStart(3, "0")}</span><span><strong>{item.label}</strong><small>{item.createdAt?.slice(11, 19)} · {item.jointCount} joints</small></span><ChevronDown size={14} /></button>)}{!selectedKeyframes.length && <div className="empty-state">还没有保存关键帧</div>}</div></div>}

          {tab === "source" && <div className="inspector-body"><div className="source-block"><div className="source-title"><Archive size={16} /><span>真实 MJCF / URDF</span></div><code>{shortName(model?.modelPath, 46)}</code><div className="source-meta"><Badge tone="green">MJCF loaded</Badge><span>{model?.jointCount || 29} movable joints</span></div></div><label className="upload-drop"><Upload size={18} /><span>导入本地 URDF</span><small>解析关节名称与限位，不替换服务模型</small><input type="file" accept=".urdf,.xml" onChange={importUrdf} /></label>{(localUrdf || urdf) && <div className="urdf-result"><div><strong>{localUrdf?.name || "G1 custom collision URDF"}</strong><span>{localUrdf?.joints?.length || urdf?.jointCount || 0} 个可动关节</span></div><X size={14} onClick={() => setLocalUrdf(null)} /></div>}<div className="source-block"><div className="source-title"><FileUp size={16} /><span>注册动作路径</span></div><input className="path-input" value={importPath} onChange={(event) => setImportPath(event.target.value)} placeholder="D:\\Develop\\Project\\UnitreeG1Dance\\..." /><button className="button button-light wide" onClick={registerPath}><FileUp size={14} /> 加入动作目录</button><small className="muted-note">服务只接受 MOTIONLAB_ACTION_ROOT 内的 .pt / .npz / .csv / .pkl。</small></div></div>}
        </aside>
      </div>

      <footer className="statusbar"><div><span className="status-label">MODEL</span><strong>{shortName(model?.modelPath, 58) || "loading"}</strong></div><div><span className="status-label">URDF</span><strong>{localUrdf?.name || "g1_custom_collision_29dof.urdf"}</strong></div><div className="status-spacer" /><div className="status-note"><Gauge size={14} /> {pose ? `root z ${(pose.root?.z || 0).toFixed(3)} m` : "等待动作帧"}</div></footer>
    </main>
    {message && <div className="toast toast-success"><Check size={15} /> {message}</div>}
    {error && <div className="toast toast-error"><CircleHelp size={15} /> <span>{error}</span><button onClick={() => setError("")}><X size={14} /></button></div>}
  </div>;
}

const rootElement = document.getElementById("root");
const motionLabRoot = globalThis.__motionLabRoot || createRoot(rootElement);
globalThis.__motionLabRoot = motionLabRoot;
motionLabRoot.render(<App />);
