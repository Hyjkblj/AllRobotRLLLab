import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Box,
  Check,
  CheckCircle2,
  CircleHelp,
  Clock3,
  Cpu,
  Database,
  Download,
  FileCheck2,
  FileArchive,
  FileUp,
  Gauge,
  GitBranch,
  Info,
  Layers3,
  LineChart,
  ListChecks,
  LoaderCircle,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  ServerCog,
  ShieldAlert,
  Settings2,
  ShieldCheck,
  Square,
  Timer,
  TriangleAlert,
  UploadCloud,
  X
} from "lucide-react";
import { platformApi, PlatformApiError } from "../../api/platformClient";

const NAV_ITEMS = [
  { id: "overview", label: "总览", icon: Gauge },
  { id: "assets", label: "项目与资源", icon: Layers3 },
  { id: "motion", label: "动作流水线", icon: GitBranch },
  { id: "training", label: "训练监控", icon: Activity },
  { id: "sim2sim", label: "验收报告", icon: ListChecks },
  { id: "artifacts", label: "策略包", icon: FileArchive }
];

const RUN_STATUS_LABELS = {
  CREATED: "已创建",
  VALIDATING: "输入校验",
  MOTION_COMPILING: "动作编译",
  MOTION_READY: "动作就绪",
  TRAINING_PREPARING: "训练准备",
  TRAINING: "训练中",
  TRAINING_SUCCEEDED: "训练完成",
  EXPORTING: "导出中",
  EXPORTED: "已导出",
  SIM2SIM_QUEUED: "验收排队",
  SIM2SIM_RUNNING: "验收中",
  SIM2SIM_PASSED: "验收通过",
  READY_TO_DOWNLOAD: "可下载",
  FAILED: "失败",
  CANCELLED: "已取消"
};

const STATUS_LABELS = {
  READY: "就绪", UPLOADED: "已上传", UPLOADING: "上传中", VALIDATING: "校验中",
  REJECTED: "已拒绝", QUEUED: "排队中", RUNNING: "处理中", SUCCEEDED: "已完成",
  FAILED: "失败", CANCELLED: "已取消", PENDING: "待处理", CREATED: "已创建",
  CHECK: "待检查", MISSING: "缺失", SELECTED: "已选择", PASSED: "通过",
  EXPORTED: "已导出", ACTIVE: "已启用", DISABLED: "已停用", LOADING: "加载中",
  GVHMR_READY: "人体姿态待确认", GMR_READY: "机器人动作待确认"
};

const STAGE_LABELS = { detect: "内容检测", retarget: "动作重定向", edit: "质量检查", compile: "动作编译", publish: "发布资源" };

function statusLabel(status) {
  return STATUS_LABELS[status] || RUN_STATUS_LABELS[status] || status || "未知";
}

function errorLabel(error) {
  const code = error?.code || error?.error_code;
  const messages = {
    ASSET_NOT_READY: "资源仍在校验，请稍候刷新后再处理。",
    ASSET_KIND_INVALID: "当前资源类型不支持此操作。",
    ASSET_IN_USE: "该资源已被动作流水线或训练任务引用，暂不能删除。",
    PROJECT_HAS_RUNS: "项目中已有训练任务，暂不能永久删除。",
    PROJECT_HAS_PIPELINES: "项目中已有动作流水线，暂不能永久删除。",
    ASSET_OBJECT_DELETE_FAILED: "文件删除失败，请稍后重试。",
    INPUT_NOT_FOUND: "找不到输入文件，请检查路径。",
    UNSUPPORTED_SOURCE_TYPE: "暂不支持该文件格式或内容。",
    GVHMR_RUNTIME_UNAVAILABLE: "视频处理运行环境未就绪，请检查 GPU 服务。",
    RETARGET_RUNTIME_UNAVAILABLE: "动作重定向运行环境未就绪，请检查 GMR 服务。"
  };
  return messages[code] || error?.message || String(error);
}

const DEFAULT_TERMINATIONS = ["timeout", "bad_anchor_orientation", "fall", "joint_limit", "nan_inf"];

function assetKindLabel(kind) {
  return kind === "video" ? "视频" : kind === "motion" ? "动作" : kind === "model" ? "模型" : kind || "资源";
}

function Badge({ children, tone = "neutral" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function formatBytes(value) {
  if (!value) return "—";
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function dateLabel(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

function statusTone(status) {
  if (["TRAINING_SUCCEEDED", "SIM2SIM_PASSED", "READY_TO_DOWNLOAD", "MOTION_READY"].includes(status)) return "green";
  if (["FAILED", "CANCELLED"].includes(status)) return "red";
  if (["TRAINING", "TRAINING_PREPARING", "EXPORTING", "SIM2SIM_RUNNING"].includes(status)) return "blue";
  return "silver";
}

function ErrorBanner({ error, onDismiss }) {
  if (!error) return null;
  return <div className="platform-error" role="alert" aria-live="polite"><AlertTriangle size={16} /><span>{errorLabel(error)}</span><button className="icon-button small" title="关闭" aria-label="关闭错误提示" onClick={onDismiss}><X size={14} /></button></div>;
}

const WORKFLOW_STAGES = [
  { id: "assets", label: "资源" },
  { id: "motion", label: "动作" },
  { id: "training", label: "训练" },
  { id: "sim2sim", label: "验收" },
  { id: "artifacts", label: "策略包" }
];

function workflowStageForRun(run) {
  if (!run) return 0;
  if (["READY_TO_DOWNLOAD"].includes(run.status)) return 4;
  if (["EXPORTED", "SIM2SIM_QUEUED", "SIM2SIM_RUNNING", "SIM2SIM_PASSED"].includes(run.status)) return 3;
  if (["TRAINING_PREPARING", "TRAINING", "TRAINING_SUCCEEDED", "EXPORTING"].includes(run.status)) return 2;
  if (["MOTION_COMPILING", "MOTION_READY"].includes(run.status)) return 1;
  return 0;
}

function NextAction({ view, selectedRun, projects, selectedProjectId, assets, onNavigate }) {
  const readyMotion = assets.some((asset) => (asset.kind || asset.version?.kind) === "motion" && asset.version?.status === "READY" && /train_motion|TrainMotionNPZ/i.test(`${asset.version?.original_filename || ""} ${asset.display_name || ""}`));
  let title = "先从项目资源开始";
  let copy = "创建项目并上传视频，平台会在动作流水线中完成 GVHMR、GMR 和 TrainMotionNPZ 编译。";
  let target = "assets";
  let action = "管理资源";
  if (!selectedProjectId || !projects.length) {
    title = "创建第一个项目";
    copy = "项目是视频、动作版本和训练 Run 的共同边界。";
    target = "assets";
    action = "创建项目";
  } else if (view === "assets" || !assets.length) {
    title = "上传视频或动作源";
    copy = "视频会进入 GVHMR → GMR；已有 NPZ、CSV 或 PT 可直接进入动作编译。";
    target = "assets";
    action = "去上传";
  } else if (!readyMotion) {
    title = "把输入处理成 TrainMotionNPZ";
    copy = "在动作流水线选择已登记的视频，确认目标机器人后提交处理并发布。";
    target = "motion";
    action = "打开动作流水线";
  } else if (!selectedRun || ["FAILED", "CANCELLED"].includes(selectedRun.status)) {
    title = "准备并提交训练 Run";
    copy = "选择动作版本、奖励配置和训练参数，然后提交异步训练。";
    target = "training";
    action = "打开训练台";
  } else if (selectedRun.status === "TRAINING_SUCCEEDED") {
    title = "导出策略";
    copy = "训练已完成，下一步生成 TorchScript 和 ONNX 策略包。";
    target = "training";
    action = "继续导出";
  } else if (selectedRun.status === "EXPORTED") {
    title = "运行三种子验收";
    copy = "导出完成后运行 MuJoCo 三 seed 质量门，只有通过后才能下载。";
    target = "training";
    action = "开始验收";
  } else if (selectedRun.status === "READY_TO_DOWNLOAD") {
    title = "策略包已就绪";
    copy = "验收已通过，可以查看报告并下载最终 artifact。";
    target = "artifacts";
    action = "查看策略包";
  } else {
    title = `训练任务正在${statusLabel(selectedRun.status)}`;
    copy = "状态和事件会自动刷新，完成后页面会提示下一步。";
    target = "training";
    action = "查看运行状态";
  }
  return <div className="next-action-card"><div className="next-action-index">NEXT</div><div className="next-action-copy"><strong>{title}</strong><span>{copy}</span></div><button className="button button-light" onClick={() => onNavigate(target)}>{action}<ArrowRight size={14} /></button></div>;
}

function WorkflowStrip({ view, selectedRun, onNavigate }) {
  const viewIndex = WORKFLOW_STAGES.findIndex((stage) => stage.id === view);
  const runProgress = workflowStageForRun(selectedRun);
  const active = selectedRun ? Math.max(viewIndex, runProgress) : Math.max(viewIndex, 0);
  return <div className="workflow-strip" aria-label="训练闭环进度">
    {WORKFLOW_STAGES.map((stage, index) => {
      const done = Boolean(selectedRun) && index < runProgress;
      const current = index === active;
      return <React.Fragment key={stage.id}>
        <button className={`workflow-step ${done ? "done" : ""} ${current ? "current" : ""}`} onClick={() => onNavigate(stage.id)} aria-current={current ? "step" : undefined}>
          <span className="workflow-index">{done ? <Check size={13} /> : String(index + 1).padStart(2, "0")}</span>
          <span><strong>{stage.label}</strong><small>{done ? "已完成" : current ? "当前阶段" : "待开始"}</small></span>
        </button>
        {index < WORKFLOW_STAGES.length - 1 && <span className={`workflow-connector ${index < active ? "done" : ""}`} />}
      </React.Fragment>;
    })}
  </div>;
}

function MetricCard({ label, value, hint, tone = "neutral", icon: Icon = Activity }) {
  return <div className={`metric-card metric-${tone}`}><div className="metric-icon"><Icon size={15} /></div><div><span>{label}</span><strong>{value}</strong><small>{hint}</small></div></div>;
}

function SectionHeader({ eyebrow, title, description, action }) {
  return <div className="platform-section-head"><div><span className="panel-kicker">{eyebrow}</span><h2>{title}</h2>{description && <p>{description}</p>}</div>{action}</div>;
}

function Overview({ health, infrastructure, projects, robots, robotCheck, runs, onNavigate, onRefresh }) {
  const robot = robots[0];
  const robotLabel = robot?.model || robot?.robot_id || "机器人适配器";
  const latestRun = runs[0];
  return <>
    <SectionHeader eyebrow="PLATFORM CONTROL" title="训练工作台" description="从项目资源到可追溯 Run 的统一入口。" action={<button className="icon-button" title="刷新平台状态" onClick={onRefresh}><RefreshCw size={16} /></button>} />
    <div className="platform-stat-grid">
      <div className="stat-card"><div className="stat-icon blue"><ServerCog size={18} /></div><span>API 服务</span><strong>{health?.status === "ok" ? "在线" : "离线"}</strong><small>{health?.storage_mode || "未连接"}</small></div>
      <div className="stat-card"><div className="stat-icon green"><Database size={18} /></div><span>存储模式</span><strong>{health?.storage_mode === "local_file" ? "本地文件" : health?.storage_mode || "—"}</strong><small>{health?.runtime_root || "等待服务"}</small></div>
      <div className="stat-card"><div className="stat-icon silver"><Box size={18} /></div><span>机器人适配器</span><strong>{robot?.model || robot?.robot_id || "—"}</strong><small>{robot ? `${robot.dof || robot.joints?.length || "—"} DoF` : "未加载"}</small></div>
      <div className="stat-card"><div className="stat-icon amber"><Activity size={18} /></div><span>当前训练任务</span><strong>{latestRun ? statusLabel(latestRun.status) : "—"}</strong><small>{latestRun?.run_id ? latestRun.run_id.slice(0, 8) : "尚未创建"}</small></div>
    </div>
    <div className="platform-columns">
      <section className="platform-panel"><SectionHeader eyebrow="PROJECTS" title="项目" action={<button className="button button-light" onClick={() => onNavigate("assets")}><Plus size={14} /> 新建项目</button>} />
        {projects.length ? <div className="project-list">{projects.slice(0, 6).map((project) => <button className="project-row" key={project.project_id} onClick={() => onNavigate("assets", project.project_id)}><span className="project-mark">{project.name?.slice(0, 1) || "P"}</span><span><strong>{project.name}</strong><small>{project.project_id}</small></span><ArrowRight size={15} /></button>)}</div> : <div className="empty-state">还没有项目</div>}
      </section>
      <section className="platform-panel"><SectionHeader eyebrow="RUNTIME" title="运行时检查" />
        <div className="check-list"><div><span><ShieldCheck size={15} /> {robotLabel} 适配器</span><Badge tone={robotCheck?.valid ? "green" : robot ? "amber" : "red"}>{statusLabel(robotCheck?.valid ? "READY" : robot ? "CHECK" : "MISSING")}</Badge></div><div><span><Cpu size={15} /> GPU / 工作进程</span><Badge tone={infrastructure?.status === "ok" ? "green" : "silver"}>{statusLabel(infrastructure?.status === "ok" ? "READY" : "PENDING")}</Badge></div><div><span><FileCheck2 size={15} /> 契约版本</span><Badge tone="green">v1</Badge></div></div>{robotCheck?.issues?.length > 0 && <div className="check-warning"><TriangleAlert size={14} /> {robotCheck.issues[0].message}</div>}
      </section>
    </div>
    <section className="platform-panel run-summary"><SectionHeader eyebrow="RECENT RUNS" title="最近运行" action={<button className="button button-light" onClick={() => onNavigate("training")}>打开训练台 <ArrowRight size={14} /></button>} />
      {runs.length ? <div className="run-table"><div className="run-table-head"><span>状态</span><span>运行任务</span><span>项目</span><span>更新时间</span></div>{runs.slice(0, 8).map((run) => <button className="run-table-row" key={run.run_id} onClick={() => onNavigate("training", run.run_id)}><Badge tone={statusTone(run.status)}>{statusLabel(run.status)}</Badge><code>{run.run_id.slice(0, 12)}</code><span>{projects.find((item) => item.project_id === run.project_id)?.name || run.project_id}</span><small>{dateLabel(run.updated_at)}</small></button>)}</div> : <div className="empty-state">创建项目后即可提交训练任务</div>}
    </section>
  </>;
}

function Assets({ projects, selectedProjectId, onSelectProject, onProjectCreated, onAssetCreated, assets, onNavigate, onAssetDeleted, onProjectDeleted }) {
  const [projectName, setProjectName] = useState("");
  const [file, setFile] = useState(null);
  const [kind, setKind] = useState("motion");
  const [license, setLicense] = useState("declared");
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState(null);

  const createProject = async (event) => {
    event.preventDefault();
    if (!projectName.trim()) return;
    setBusy(true); setLocalError(null);
    try { const result = await platformApi.createProject(projectName.trim()); setProjectName(""); onProjectCreated(result.item); }
    catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };

  const upload = async () => {
    if (!selectedProjectId || !file) return;
    setBusy(true); setProgress(5); setLocalError(null);
    try { const result = await platformApi.uploadAsset({ projectId: selectedProjectId, file, kind, license: { status: license, source: "user" }, onProgress: setProgress }); onAssetCreated({ ...result, file }); setFile(null); }
    catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };
  const removeAsset = async (asset) => {
    const versionId = asset.version?.asset_version_id;
    if (!versionId) return;
    if (!window.confirm(`确认永久删除资源“${asset.version?.original_filename || asset.display_name}”？文件和资源记录都将被删除。`)) return;
    setBusy(true); setLocalError(null);
    try { await platformApi.deleteAsset(versionId); onAssetDeleted?.(versionId); }
    catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };
  const removeProject = async () => {
    const project = projects.find((item) => item.project_id === selectedProjectId);
    if (!project || !window.confirm(`确认永久删除项目“${project.name}”及其上传文件？已有流水线或训练任务的项目暂不能删除。`)) return;
    setBusy(true); setLocalError(null);
    try { await platformApi.deleteProject(project.project_id); onProjectDeleted?.(project.project_id); }
    catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };

  return <>
    <SectionHeader eyebrow="PROJECTS & ASSETS" title="项目与资源" description="资产进入校验状态后，才可用于动作编译和训练。" />
    {localError && <ErrorBanner error={localError} onDismiss={() => setLocalError(null)} />}
    <div className="platform-columns assets-columns">
      <section className="platform-panel"><SectionHeader eyebrow="PROJECT" title="项目空间" />
        <form className="inline-form" onSubmit={createProject}><input value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="项目名称" maxLength={120} /><button className="button button-dark" type="submit" disabled={busy || !projectName.trim()}><Plus size={14} /> 创建</button></form>
        <div className="project-list compact">{projects.map((project) => <div className="project-row-wrapper" key={project.project_id}><button type="button" className={`project-row ${project.project_id === selectedProjectId ? "selected" : ""}`} onClick={() => onSelectProject(project.project_id)}><span className="project-mark">{project.name?.slice(0, 1) || "P"}</span><span><strong>{project.name}</strong><small>{project.project_id}</small></span>{project.project_id === selectedProjectId && <Badge tone="blue">当前</Badge>}</button>{project.project_id === selectedProjectId && <button type="button" className="project-delete" onClick={removeProject} disabled={busy}>删除</button>}</div>)}</div>
      </section>
      <section className="platform-panel"><SectionHeader eyebrow="UPLOAD" title="添加资源" />
        <div className="upload-zone"><UploadCloud size={24} /><strong>{file ? file.name : "选择视频或动作文件"}</strong><small>{file ? formatBytes(file.size) : "MP4 · NPZ · CSV · PT"}</small><input type="file" accept=".mp4,.mov,.avi,.mkv,.webm,.npz,.csv,.pt,.pkl" onChange={(event) => { const nextFile = event.target.files?.[0] || null; setFile(nextFile); if (nextFile && /\.(mp4|mov|avi|mkv|webm)$/i.test(nextFile.name)) setKind("video"); else if (nextFile && /\.(npz|csv|pt|pkl)$/i.test(nextFile.name)) setKind("motion"); }} /></div>
        <div className="form-grid"><label>资源类型<select value={kind} onChange={(event) => setKind(event.target.value)}><option value="video">视频</option><option value="motion">动作</option><option value="model">模型</option></select></label><label>许可证<select value={license} onChange={(event) => setLicense(event.target.value)}><option value="declared">已声明</option><option value="research_only">研究用途</option><option value="internal">内部资产</option></select></label></div>
        <button className="button button-blue wide" onClick={upload} disabled={!selectedProjectId || !file || busy}>{busy ? <><LoaderCircle size={14} className="spin" /> 上传中 {progress}%</> : <><FileUp size={14} /> 上传并登记</>}</button>
        {!selectedProjectId && <small className="form-note">先选择项目空间</small>}
      </section>
    </div>
    <section className="platform-panel"><SectionHeader eyebrow="ASSET VERSIONS" title="已登记资源" action={<button className="button button-light" onClick={() => onNavigate("motion")}><GitBranch size={14} /> 动作流水线</button>} />
          {assets.length ? <div className="asset-table"><div className="asset-table-head"><span>文件</span><span>类型</span><span>版本</span><span>状态</span><span>操作</span></div>{assets.map((asset) => { const status = asset.version?.status || asset.completed?.item?.status || "UPLOADED"; return <div className="asset-table-row" key={asset.version?.asset_version_id || asset.asset_id}><span><strong>{asset.version?.original_filename || asset.file?.name || asset.display_name}</strong><small>{asset.asset_id || asset.version?.asset_id}</small></span><Badge tone="silver">{assetKindLabel(asset.kind || asset.version?.kind)}</Badge><span>v{asset.version?.version || 1}</span><Badge tone={status === "READY" ? "green" : status === "REJECTED" ? "red" : "blue"}>{statusLabel(status)}</Badge><button className="button button-danger" type="button" onClick={() => removeAsset(asset)} disabled={busy}>删除资源</button></div>; })}</div> : <div className="empty-state">暂无资源</div>}
    </section>
  </>;
}

function MotionPipeline({ assets, robots, selectedRobotId, onRobotSelect, onDetect, detection, onOpenEditor, onRefresh }) {
  const [path, setPath] = useState("");
  const [assetVersionId, setAssetVersionId] = useState("");
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [pipeline, setPipeline] = useState(null);
  const sourceAssets = assets.filter((asset) => ["motion", "video"].includes(asset.kind || asset.version?.kind));
  const selectedAsset = sourceAssets.find((asset) => asset.version?.asset_version_id === assetVersionId);
  const selectedAssetKind = selectedAsset?.kind || selectedAsset?.version?.kind || (/\.(mp4|mov|avi|mkv|webm)$/i.test(selectedAsset?.version?.original_filename || "") ? "video" : undefined);
  const selectedAssetIsCompiled = /train_motion|TrainMotionNPZ/i.test(`${selectedAsset?.version?.original_filename || ""} ${selectedAsset?.display_name || ""}`);
  const selectedRobot = robots.find((item) => item.robot_id === selectedRobotId);
  const robotLabel = selectedRobot?.model || selectedRobot?.robot_id || "目标机器人";
  const nextVideoStage = pipeline?.gmr_output_path ? "compile" : pipeline?.gvhmr_output_path ? "gmr" : "gvhmr";
  const nextVideoAction = nextVideoStage === "compile" ? "编译训练动作" : nextVideoStage === "gmr" ? "启动 GMR" : "开始 GVHMR";
  const videoActionLabel = pipeline?.status === "FAILED" ? `重试${nextVideoAction}` : nextVideoAction;

  useEffect(() => {
    setPipeline(null);
  }, [assetVersionId]);

  useEffect(() => {
    if (!assetVersionId) return undefined;
    let cancelled = false;
    platformApi.getMotionPipeline(assetVersionId).then((result) => {
      if (!cancelled) setPipeline(result.item);
    }).catch((error) => {
      if (!cancelled && error?.status !== 404) setLocalError(error);
    });
    return () => { cancelled = true; };
  }, [assetVersionId]);

  useEffect(() => {
    // A stage-ready state is a user confirmation gate. Stop polling until the
    // user explicitly starts the next stage; this avoids needless GET traffic
    // while preserving polling for queued/running work.
    if (!pipeline?.pipeline_id || ["GVHMR_READY", "GMR_READY", "READY", "FAILED"].includes(pipeline.status)) return undefined;
    let cancelled = false;
    const timer = window.setInterval(() => {
      platformApi.getMotionPipelineById(pipeline.pipeline_id).then((result) => {
        if (cancelled) return;
        setPipeline(result.item);
        if (["READY", "FAILED"].includes(result.item?.status)) onRefresh?.();
      }).catch((error) => { if (!cancelled) setLocalError(error); });
    }, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [onRefresh, pipeline?.pipeline_id, pipeline?.status]);

  const detect = async (event) => {
    event.preventDefault();
    if (selectedAssetKind === "video") {
      setLocalError({ message: "视频只进行格式校验，请直接开始 GVHMR。" });
      return;
    }
    if (!path.trim() && !assetVersionId) return;
    setBusy(true); setLocalError(null);
    try { const result = await platformApi.detectMotion({ path: path.trim() || null, assetVersionId: assetVersionId || null, robotId: selectedRobotId }); onDetect(result.descriptor); }
    catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };

  const process = async () => {
    if (!assetVersionId) return;
    setBusy(true); setLocalError(null);
    try {
      const editConfig = { source_motion_version_id: assetVersionId, robot_id: selectedRobotId || robots[0]?.robot_id, global_transform: { translation: [0, 0, 0], yaw_offset: 0, time_scale: 1 }, joint_offsets: [], ik_targets: [], keyframes: [], filters: { smooth: true, max_velocity_check: true } };
      let result;
      if (selectedAssetKind === "video" && !pipeline?.pipeline_id) result = await platformApi.processMotion(assetVersionId, editConfig, "async", "gvhmr");
      else if (selectedAssetKind === "video" && pipeline?.gmr_output_path) result = await platformApi.compileMotionPipeline(pipeline.pipeline_id);
      else if (selectedAssetKind === "video" && pipeline?.gvhmr_output_path) result = await platformApi.startGmr(pipeline.pipeline_id);
      else if (selectedAssetKind === "video") result = await platformApi.startGvhmr(pipeline.pipeline_id);
      else result = await platformApi.processMotion(assetVersionId, editConfig, "async", "full");
      setPipeline(result.item);
    } catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };

  const stage = (name) => pipeline?.stages?.find((item) => item.name === name);
  const stageTone = (name) => stage(name)?.status === "SUCCEEDED" ? "green" : stage(name)?.status === "FAILED" ? "red" : stage(name)?.status === "RUNNING" ? "blue" : "silver";
  const visualizationLinks = (name, stageName) => (stage(name)?.artifacts || []).filter((item) => /\.(mp4|webm|avi|mov|png|jpg)$/i.test(item)).map((item) => ({ name: item, url: platformApi.getMotionVisualizationUrl(pipeline?.pipeline_id, stageName, item) }));

  return <>
    <SectionHeader eyebrow="STEP 2 · MOTION" title="把输入变成训练动作" description="视频会经过 GVHMR → GMR；已有轨迹文件可直接编译为 TrainMotionNPZ。" action={<button className="button button-light" onClick={onOpenEditor}><Play size={14} /> 打开 MuJoCo 预览</button>} />
    {localError && <ErrorBanner error={localError} onDismiss={() => setLocalError(null)} />}
    <div className="pipeline-track"><div className={`pipeline-step ${stageTone("detect")}`}><span>01</span><strong>检测</strong><small>内容识别</small></div><ArrowRight size={17} /><div className={`pipeline-step ${stageTone("retarget")}`}><span>02</span><strong>重定向</strong><small>{robotLabel} 动作</small></div><ArrowRight size={17} /><div className={`pipeline-step ${stageTone("edit")}`}><span>03</span><strong>编辑</strong><small>质量检查</small></div><ArrowRight size={17} /><div className={`pipeline-step ${stageTone("compile")}`}><span>04</span><strong>编译</strong><small>训练动作</small></div><ArrowRight size={17} /><div className={`pipeline-step ${stageTone("publish")}`}><span>05</span><strong>发布</strong><small>生成资源版本</small></div></div>
    <div className="pipeline-note"><Info size={15} /><span>本地模式支持已登记机器人的 NPZ、CSV、PT 直接轨迹转换；视频和人体姿态输入会根据运行时配置进入 GVHMR/GMR。</span></div>
    <div className="platform-columns">
      <section className="platform-panel"><SectionHeader eyebrow="INPUT CHECK" title="输入检查" /><label className="target-robot-field">目标机器人<select value={selectedRobotId || ""} onChange={(event) => onRobotSelect?.(event.target.value)}><option value="">选择机器人</option>{robots.map((item) => <option key={item.robot_id} value={item.robot_id}>{item.model || item.robot_id} · {item.dof} DoF</option>)}</select></label>
        <div className="stack-form"><label>已登记输入<select value={assetVersionId} onChange={(event) => { setAssetVersionId(event.target.value); if (event.target.value) setPath(""); }}><option value="">选择视频或动作资源</option>{sourceAssets.filter((asset) => asset.version?.asset_version_id).map((asset) => <option key={asset.version.asset_version_id} value={asset.version.asset_version_id}>{assetKindLabel(asset.kind || asset.version?.kind)} · {asset.version.original_filename} · v{asset.version.version} · {statusLabel(asset.version.status)}</option>)}</select></label></div>
        {selectedAssetKind === "video" ? <div className="pipeline-note"><FileCheck2 size={15} /><span>视频上传阶段只做格式和文件可读性校验；人体姿态恢复将在提交处理后单独运行。</span></div> : <form className="stack-form" onSubmit={detect}><label>或服务器路径<input value={path} onChange={(event) => { setPath(event.target.value); if (event.target.value) setAssetVersionId(""); }} placeholder="/data/motions/walk.npz" /></label><button className="button button-blue" type="submit" disabled={busy || (!path.trim() && !assetVersionId)}>{busy ? <><LoaderCircle size={14} className="spin" /> 检测中</> : <><FileCheck2 size={14} /> 检查动作格式</>}</button></form>}
        {detection && <div className="detection-result"><div><Badge tone="green">VALID</Badge><strong>{detection.detected_type}</strong></div><span>{detection.file_format} · {detection.source_skeleton}</span><small>{detection.detector_version}</small></div>}
        <div className="pipeline-submit"><div><span className="panel-kicker">下一步</span><strong>{selectedAssetIsCompiled ? "该资源已经是 TrainMotionNPZ" : selectedAssetKind === "video" ? videoActionLabel : selectedAsset?.version?.status === "READY" ? "准备编译训练动作" : "先选择就绪输入"}</strong><small>{pipeline?.error_code ? `${pipeline.error_code} · ${pipeline.error_message}` : selectedAssetKind === "video" ? "每个阶段完成后可单独检查和确认" : "处理成功后会发布新的 TrainMotionNPZ 资源版本"}</small></div><button className="button button-dark" onClick={process} disabled={busy || !assetVersionId || selectedAssetIsCompiled || selectedAsset?.version?.status !== "READY" || ["QUEUED", "RUNNING", "READY"].includes(pipeline?.status)}>{busy ? <><LoaderCircle size={14} className="spin" /> 处理中</> : <><GitBranch size={14} /> {selectedAssetKind === "video" ? videoActionLabel : "编译并发布"}</>}</button></div>
      </section>
      <section className="platform-panel"><SectionHeader eyebrow="RUNNER STATUS" title="处理器状态" />
        <div className="check-list"><div><span><Activity size={15} /> GVHMR 人体姿态</span><Badge tone={stageTone("detect")}>{statusLabel(stage("detect")?.status || "待开始")}</Badge></div><div><span><GitBranch size={15} /> {robotLabel} GMR 重定向</span><Badge tone={stageTone("retarget")}>{statusLabel(stage("retarget")?.status || "待开始")}</Badge></div><div><span><ShieldCheck size={15} /> 质量检查</span><Badge tone={stageTone("edit")}>{statusLabel(stage("edit")?.status || "待开始")}</Badge></div><div><span><FileCheck2 size={15} /> 训练动作编译</span><Badge tone={stageTone("compile")}>{statusLabel(stage("compile")?.status || "待开始")}</Badge></div></div>
        {pipeline?.pipeline_id && <div className="pipeline-visuals"><div><strong>阶段可视化</strong><small>仅显示运行时实际生成并登记的文件</small></div>{[...visualizationLinks("detect", "gvhmr"), ...visualizationLinks("retarget", "gmr")].map((item) => <a className="button button-light" key={item.url} href={item.url} target="_blank" rel="noreferrer"><Play size={14} /> {item.name.split(/[\\/]/).pop()}</a>)}</div>}
        {pipeline?.quality && <div className="pipeline-quality"><Badge tone={pipeline.quality.status === "PASS" ? "green" : pipeline.quality.status === "BLOCKED" ? "red" : "amber"}>{pipeline.quality.status}</Badge><span>最大速度 {pipeline.quality.stats?.max_joint_velocity_rad_s?.toFixed?.(2) || "—"} rad/s</span><span>限位违规 {((pipeline.quality.stats?.joint_limit_violation_ratio || 0) * 100).toFixed(2)}%</span></div>}
      </section>
    </div>
    <section className="platform-panel"><SectionHeader eyebrow="MOTION SOURCES" title="当前项目输入" />{sourceAssets.length ? <div className="asset-table">{sourceAssets.map((asset) => { const status = asset.version?.status || "UPLOADED"; return <div className="asset-table-row" key={asset.version?.asset_version_id || asset.asset_id}><span><strong>{assetKindLabel(asset.kind || asset.version?.kind)} · {asset.version?.original_filename || asset.file?.name || asset.display_name}</strong><small>{asset.version?.asset_version_id || asset.asset_id}</small></span><Badge tone={status === "READY" ? "green" : status === "REJECTED" ? "red" : "blue"}>{statusLabel(status)}</Badge><button className="button button-light" onClick={onOpenEditor}>预览 <ArrowRight size={14} /></button></div>; })}</div> : <div className="empty-state">先在项目资源中上传视频或动作文件</div>}</section>
  </>;
}

function Training({ projects, selectedProjectId, assets, robots, tasks, selectedRobotId, onRobotSelect, rewardTemplates, runs, selectedRun, onRunCreated, onRunSelect, onRefresh, onProjectRefresh }) {
  const [terms, setTerms] = useState([]);
  const [rewardVersion, setRewardVersion] = useState(null);
  const [motionVersionId, setMotionVersionId] = useState("");
  const [iterations, setIterations] = useState(2000);
  const [gpuMemory, setGpuMemory] = useState(8);
  const [annealing, setAnnealing] = useState("none");
  const [showParameters, setShowParameters] = useState(true);
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [events, setEvents] = useState([]);
  const [lastSeq, setLastSeq] = useState(0);
  const stopSseRef = useRef(null);
  const motionAssets = assets.filter((asset) => (asset.kind || asset.version?.kind) === "motion" && asset.version?.asset_version_id && asset.version?.status === "READY" && /train_motion|TrainMotionNPZ/i.test(`${asset.version?.original_filename || ""} ${asset.display_name || ""}`));
  const robot = robots.find((item) => item.robot_id === selectedRobotId) || robots[0];
  const task = tasks.find((item) => item.robot_id === robot?.robot_id) || tasks[0];

  useEffect(() => {
    setTerms(rewardTemplates.map((item) => ({ ...item, enabled: true, weight: item.default_weight, params: Object.fromEntries(Object.entries(item.parameter_schema || {}).map(([key, schema]) => [key, schema.default])) })));
  }, [rewardTemplates]);

  useEffect(() => () => stopSseRef.current?.(), []);

  useEffect(() => {
    if (!selectedRun?.run_id) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const result = await platformApi.getRun(selectedRun.run_id);
        if (!cancelled && result?.item) {
          if (result.item.status !== selectedRun.status) onProjectRefresh?.();
          onRunSelect(result);
        }
      } catch (error) {
        if (!cancelled && error?.status !== 404) setLocalError(error);
      }
    };
    const timer = window.setInterval(poll, 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [onProjectRefresh, onRunSelect, selectedRun?.run_id, selectedRun?.status]);

  useEffect(() => {
    stopSseRef.current?.();
    setEvents([]); setLastSeq(0);
    if (!selectedRun?.run_id) return undefined;
    stopSseRef.current = platformApi.subscribeRun(selectedRun.run_id, {
      onEvent: (event) => { setEvents((current) => [...current.slice(-99), event]); setLastSeq(event.seq || 0); },
      onError: (error) => setLocalError(error)
    });
    return () => stopSseRef.current?.();
  }, [selectedRun?.run_id]);

  const updateTerm = (id, patch) => setTerms((current) => current.map((term) => term.id === id ? { ...term, ...patch } : term));
  const updateParam = (id, key, value) => setTerms((current) => current.map((term) => term.id === id ? { ...term, params: { ...term.params, [key]: value } } : term));

  const latestMetrics = useMemo(() => {
    const metrics = {};
    events.filter((event) => event.event_type === "metric").forEach((event) => {
      Object.entries(event.payload || {}).forEach(([name, value]) => { if (typeof value === "number") metrics[name] = value; });
      if (event.payload?.name && typeof event.payload.value === "number") metrics[event.payload.name] = event.payload.value;
    });
    return metrics;
  }, [events]);

  const saveReward = async () => {
    setBusy(true); setLocalError(null);
    try {
      if (!terms.some((term) => term.enabled)) throw new PlatformApiError("至少保留一项已注册 shaping reward；安全终止项由平台单独锁定。", { code: "REWARD_TERMS_EMPTY" });
      if (!robot?.robot_id || !task?.task_id) throw new PlatformApiError("请先选择有效的机器人和任务", { code: "TRAINING_SCOPE_REQUIRED" });
      const config = { schema_version: "reward_config.v1", base_template: `${task.task_id}_v1`, terms: terms.filter((term) => term.enabled).map(({ id, weight, params }) => ({ id, enabled: true, weight: Number(weight), params })), terminations: DEFAULT_TERMINATIONS, annealing: annealing === "none" ? [] : [{ term_id: "tracking.joint_pos", start_step: 0, end_step: Number(iterations), mode: annealing }] };
      const validation = await platformApi.validateReward(config, robot?.robot_id, task?.task_id);
      if (!validation.result?.valid) throw new PlatformApiError(validation.result?.issues?.[0]?.message || "奖励配置未通过校验", { code: "REWARD_CONFIG_INVALID", details: validation.result?.issues });
      const result = await platformApi.createReward(config, rewardVersion?.version_id, robot?.robot_id, task?.task_id);
      setRewardVersion(result.item);
    } catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };

  const createAndSubmitRun = async () => {
    if (!selectedProjectId || !motionVersionId || !rewardVersion) return;
    setBusy(true); setLocalError(null);
    try {
      if (!robot?.robot_id || !task?.task_id || !task?.scene_id) throw new PlatformApiError("请先选择有效的机器人和任务", { code: "TRAINING_SCOPE_REQUIRED" });
      const control = robot?.actuation;
      if (!control || !Number.isFinite(Number(control.policy_dt)) || !Number.isFinite(Number(control.control_dt)) || !Number.isFinite(Number(control.action_scale))) {
        throw new PlatformApiError("机器人适配器未提供完整控制参数", { code: "ROBOT_CONFIG_INCOMPLETE" });
      }
      const actionMode = robot.action_modes?.[0];
      if (!actionMode) throw new PlatformApiError("机器人适配器未声明动作模式", { code: "ROBOT_ACTION_MODE_MISSING" });
      const trainingConfig = { schema_version: "training_config.v1", task_id: task.task_id, scene_id: task.scene_id, motion_asset_version_id: motionVersionId, observation: { history_length: 3, include_root_velocity: true, include_projected_gravity: true, include_reference: true, clip_value: 100 }, action: { mode: actionMode, scale: Number(control.action_scale), clip: 1 }, control: { decimation: Math.max(1, Math.round(Number(control.policy_dt) / Number(control.control_dt))), kp_profile: control.kp_profile, kd_profile: control.kd_profile }, ppo: { algorithm: "rsl_rl_ppo", max_iterations: Number(iterations) }, resources: { gpu_memory_gb: Number(gpuMemory), cpu_cores: 8, shared_memory_gb: 8, exclusive_gpu: false } };
      const validation = await platformApi.validateTraining(trainingConfig, robot?.robot_id);
      if (!validation.result?.valid) throw new PlatformApiError(validation.result?.issues?.[0]?.message || "训练配置未通过校验", { code: "TRAINING_CONFIG_INVALID", details: validation.result?.issues });
      const run = await platformApi.createRun({ project_id: selectedProjectId, robot: { robot_id: robot?.robot_id }, motion: { train_motion_asset_version_id: motionVersionId }, reward_config_sha256: rewardVersion.config_sha256, training_config_sha256: await hashJson(trainingConfig), execution: { mode: "async" } });
      onRunCreated(run.item);
      const submission = await platformApi.submitTraining(run.item.run_id, trainingConfig);
      onRunCreated({ ...run.item, status: "TRAINING_PREPARING", submission: submission.submission });
    } catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };

  const refreshSelected = async () => {
    if (!selectedRun?.run_id) return;
    try {
      onRunSelect(await platformApi.getRun(selectedRun.run_id));
    } catch (error) {
      if (error?.status === 404 || error?.code === "RUN_NOT_FOUND") {
        setSelectedRun(null);
        setLocalError({ message: "该 Run 已不存在，请从当前项目重新选择运行任务。" });
      } else setLocalError(error);
    }
  };

  const submitStage = async (operation) => {
    if (!selectedRun?.run_id) return;
    setBusy(true); setLocalError(null);
    try {
      const result = operation === "export" ? await platformApi.submitExport(selectedRun.run_id) : await platformApi.submitSim2sim(selectedRun.run_id);
      onRunSelect({ ...selectedRun, status: result.submission?.operation === "export" ? "EXPORTING" : "SIM2SIM_QUEUED", submission: result.submission });
    } catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };

  const updateRun = async (operation) => {
    if (!selectedRun?.run_id) return;
    setBusy(true); setLocalError(null);
    try {
      const result = await (operation === "cancel" ? platformApi.cancelRun(selectedRun.run_id) : platformApi.retryRun(selectedRun.run_id));
      onRunSelect(result.item);
      onRefresh();
    } catch (error) { setLocalError(error); }
    finally { setBusy(false); }
  };

  return <>
    <SectionHeader eyebrow="TRAINING & ACCEPTANCE" title="训练监控" description="RewardConfig、TrainingConfig 和 Run 均由后端版本化，长任务通过队列异步执行。" action={<button className="icon-button" title="刷新 Run" onClick={refreshSelected}><RefreshCw size={16} /></button>} />
    {localError && <ErrorBanner error={localError} onDismiss={() => setLocalError(null)} />}
    <div className="training-context"><div><span className="panel-kicker">ACTIVE PROJECT</span><strong>{projects.find((item) => item.project_id === selectedProjectId)?.name || "未选择项目"}</strong></div><div className="context-meta"><span><Box size={14} /> {robot?.model || robot?.robot_id || "Robot adapter pending"}</span><span><Timer size={14} /> {task?.task_id || "未选择任务"} · 异步队列 · SSE 实时事件</span></div></div>
    <div className="training-grid">
      <section className="platform-panel reward-panel"><SectionHeader eyebrow="REWARD BUILDER" title="奖励配置" action={<div className="section-actions">{rewardVersion && <Badge tone="green">v{rewardVersion.version}</Badge>}<button className="button button-light compact" onClick={() => setShowParameters((value) => !value)}>{showParameters ? "收起参数" : "展开参数"}</button></div>} />
        <div className="reward-list">{terms.length ? terms.map((term) => <div className={`reward-row ${term.enabled ? "enabled" : "disabled"}`} key={term.id}><label className="switch"><input type="checkbox" checked={term.enabled} onChange={(event) => updateTerm(term.id, { enabled: event.target.checked })} aria-label={`启用 ${term.id}`} /><span /></label><div className="reward-copy"><strong>{term.name || term.id}</strong><small>{term.description || "注册奖励项"}</small><em>{term.source || "registry"} · {term.unit || "normalized"}</em></div><label className="weight-wrap"><span>权重</span><input className="weight-input" type="number" value={term.weight} step="0.01" onChange={(event) => updateTerm(term.id, { weight: event.target.value })} disabled={!term.enabled} /></label>{showParameters && term.parameter_schema && Object.keys(term.parameter_schema).length > 0 && <div className="reward-params">{Object.entries(term.parameter_schema).map(([key, schema]) => <label key={key}><span>{key}</span><input type="number" value={term.params?.[key] ?? ""} min={schema.minimum} max={schema.maximum} step={schema.multiple_of || "any"} onChange={(event) => updateParam(term.id, key, event.target.value === "" ? "" : Number(event.target.value))} disabled={!term.enabled} /></label>)}</div>}</div>) : <div className="empty-state">正在加载奖励注册表</div>}</div>
        <div className="reward-footer"><label className="field-inline"><span>退火策略</span><select value={annealing} onChange={(event) => setAnnealing(event.target.value)}><option value="none">不退火</option><option value="linear">线性</option><option value="cosine">Cosine</option></select></label><span className="safe-term"><ShieldAlert size={14} /> 5 项安全终止由平台锁定</span></div>
        <button className="button button-dark wide" onClick={saveReward} disabled={busy || !terms.length}>{busy ? <><LoaderCircle size={14} className="spin" /> 保存中</> : <><ShieldCheck size={14} /> 校验并保存版本</>}</button>
      </section>
      <section className="platform-panel train-panel"><SectionHeader eyebrow="RUN BUILDER" title="创建训练 Run" />
        <div className="form-grid"><label>项目<select value={selectedProjectId || ""} disabled><option value={selectedProjectId}>{projects.find((item) => item.project_id === selectedProjectId)?.name || "选择项目"}</option></select></label><label>机器人<select value={robot?.robot_id || ""} onChange={(event) => onRobotSelect?.(event.target.value)}><option value="">选择机器人</option>{robots.map((item) => <option key={item.robot_id} value={item.robot_id}>{item.model || item.robot_id} · {item.dof} DoF</option>)}</select></label></div>
        <div className="form-grid"><label>任务<strong className="field-value">{task?.task_id || "未注册"}</strong></label><label>动作版本<select value={motionVersionId} onChange={(event) => setMotionVersionId(event.target.value)}><option value="">选择 TrainMotionNPZ</option>{motionAssets.map((asset) => <option key={asset.version.asset_version_id} value={asset.version.asset_version_id}>{asset.version.original_filename} · v{asset.version.version}</option>)}</select></label></div>
        <div className="form-grid"><label>最大迭代<input type="number" min="1" max="1000000" value={iterations} onChange={(event) => setIterations(event.target.value)} /></label><label>GPU 显存 GB<input type="number" min="1" max="48" value={gpuMemory} onChange={(event) => setGpuMemory(event.target.value)} /></label></div>
        <div className="run-readiness"><div><span><CheckCircle2 size={14} /> 机器人适配器</span><Badge tone={robot ? "green" : "red"}>{robot ? "READY" : "MISSING"}</Badge></div><div><span><CheckCircle2 size={14} /> 任务注册</span><Badge tone={task ? "green" : "red"}>{task ? "READY" : "MISSING"}</Badge></div><div><span><CheckCircle2 size={14} /> 奖励版本</span><Badge tone={rewardVersion ? "green" : "amber"}>{rewardVersion ? `v${rewardVersion.version}` : "需保存"}</Badge></div><div><span><CheckCircle2 size={14} /> 动作输入</span><Badge tone={motionVersionId ? "green" : "amber"}>{motionVersionId ? "SELECTED" : "需选择"}</Badge></div></div>
        <button className="button button-blue wide" onClick={createAndSubmitRun} disabled={busy || !selectedProjectId || !motionVersionId || !rewardVersion}><Play size={14} /> 提交异步训练</button>
        <div className="stage-actions"><button className="button button-light" onClick={() => submitStage("export")} disabled={busy || !selectedRun || selectedRun.status !== "TRAINING_SUCCEEDED"}><ArrowRight size={14} /> 导出策略</button><button className="button button-light" onClick={() => submitStage("sim2sim")} disabled={busy || !selectedRun || !["EXPORTED", "SIM2SIM_PASSED"].includes(selectedRun.status)}><ShieldCheck size={14} /> 三种子验收</button></div>
        {selectedRun && <div className="run-control-row"><span>Run {selectedRun.run_id.slice(0, 12)} · {RUN_STATUS_LABELS[selectedRun.status] || selectedRun.status}</span>{["TRAINING", "TRAINING_PREPARING", "SIM2SIM_RUNNING", "SIM2SIM_QUEUED", "EXPORTING"].includes(selectedRun.status) ? <button className="button button-danger" onClick={() => updateRun("cancel")} disabled={busy}><Square size={13} /> 取消</button> : ["FAILED", "CANCELLED"].includes(selectedRun.status) ? <button className="button button-light" onClick={() => updateRun("retry")} disabled={busy}><RotateCcw size={13} /> 重试新 attempt</button> : null}</div>}
        <small className="form-note">当前本地模式仅允许后端已注册的执行器；未注册真实 Isaac runner 时任务会被拒绝。</small>
      </section>
    </div>
    <div className="platform-columns training-bottom">
      <section className="platform-panel"><SectionHeader eyebrow="RUNS" title="运行列表" />{runs.length ? <div className="run-list">{runs.map((run) => <button className={`run-row ${selectedRun?.run_id === run.run_id ? "selected" : ""}`} key={run.run_id} onClick={() => onRunSelect(run)}><Badge tone={statusTone(run.status)}>{RUN_STATUS_LABELS[run.status] || run.status}</Badge><span><strong>{run.run_id.slice(0, 12)}</strong><small>{dateLabel(run.updated_at)}</small></span><ArrowRight size={14} /></button>)}</div> : <div className="empty-state">尚未创建 Run</div>}</section>
      <section className="platform-panel"><SectionHeader eyebrow="RUN MONITOR" title={selectedRun ? `Run ${selectedRun.run_id.slice(0, 12)}` : "选择一个 Run"} action={selectedRun && <Badge tone={statusTone(selectedRun.status)}>{RUN_STATUS_LABELS[selectedRun.status] || selectedRun.status}</Badge>} />{selectedRun ? <><div className="metric-grid"><MetricCard label="总回报" value={latestMetrics.total_reward?.toFixed?.(2) || "—"} hint="episode / total_reward" tone="blue" icon={LineChart} /><MetricCard label="跌倒率" value={latestMetrics.fall_rate != null ? `${(latestMetrics.fall_rate * 100).toFixed(1)}%` : "—"} hint="安全终止" tone="amber" icon={TriangleAlert} /><MetricCard label="显存" value={latestMetrics.gpu_memory_used_gb != null ? `${latestMetrics.gpu_memory_used_gb.toFixed(1)} GB` : "—"} hint="worker heartbeat" tone="green" icon={Cpu} /></div><div className="event-meta"><span><Clock3 size={14} /> 最后事件 #{lastSeq}</span><span><Cpu size={14} /> {selectedRun.manifest?.runtime?.isaac_sim_package || "Isaac Sim pending"}</span><span><Info size={14} /> 断线后按游标恢复</span></div><div className="event-log" aria-live="polite">{events.length ? events.map((event, index) => <div key={`${event.seq}-${index}`}><span>{event.seq || "—"}</span><Badge tone={event.level === "ERROR" ? "red" : event.level === "WARNING" ? "amber" : "silver"}>{event.event_type}</Badge><strong>{event.message || event.stage}</strong></div>) : <div className="empty-state">等待服务端事件</div>}</div></> : <div className="empty-state">从左侧选择 Run 查看实时指标和事件</div>}</section>
    </div>
  </>;
}

function Sim2SimView({ runs, robots, selectedRun, onRunSelect, onNavigate }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    if (!selectedRun?.run_id) { setReport(null); return undefined; }
    setLoading(true); setError(null);
    platformApi.getSim2simReport(selectedRun.run_id).then((result) => { if (!cancelled) setReport(result.item || result); }).catch((loadError) => { if (!cancelled) setError(loadError); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [selectedRun?.run_id]);

  const diagnosis = useMemo(() => {
    const failures = report?.hard_failures || [];
    if (!failures.length && report?.status === "PASSED") return { tone: "green", title: "验收通过", copy: "3 个随机种子均满足当前 adapter 的硬阈值。" };
    if (failures.some((item) => /JOINT|ROOT|ORIENTATION|FOOT|SATURATION/.test(item))) return { tone: "amber", title: "模型 / 控制映射需检查", copy: "优先核对 joint order、PD、control dt、action scale 与 MuJoCo adapter 版本。" };
    if (failures.some((item) => /SURVIVAL|PROCESS/.test(item))) return { tone: "red", title: "奖励 / 训练结果异常", copy: "检查训练回报、跌倒率、动作饱和率和最近一次 attempt 日志。" };
    return { tone: "amber", title: "动作质量需要复核", copy: "回到动作流水线检查根高度、四元数、限位和接触告警。" };
  }, [report]);

  const runRobotId = selectedRun?.manifest?.robot?.robot_id || selectedRun?.robot?.robot_id;
  const runRobot = robots.find((item) => item.robot_id === runRobotId);
  const robotLabel = runRobot?.model || runRobot?.robot_id || runRobotId || "目标机器人";
  const sim2simLabel = report?.adapter || "sim2sim pending";

  return <>
    <SectionHeader eyebrow="SIM2SIM ACCEPTANCE" title="验收报告" description="每个策略包固定运行 3 个随机种子，报告与阈值、版本和模型 hash 一起归档。" action={<button className="button button-light" onClick={() => onNavigate("training")}><Activity size={14} /> 返回训练监控</button>} />
    <div className="acceptance-layout">
      <section className="platform-panel run-picker"><SectionHeader eyebrow="RUNS" title="选择策略 Run" />{runs.length ? <div className="run-list">{runs.map((run) => <button className={`run-row ${selectedRun?.run_id === run.run_id ? "selected" : ""}`} key={run.run_id} onClick={() => onRunSelect(run)}><Badge tone={statusTone(run.status)}>{RUN_STATUS_LABELS[run.status] || run.status}</Badge><span><strong>{run.run_id.slice(0, 12)}</strong><small>{dateLabel(run.updated_at)}</small></span><ArrowRight size={14} /></button>)}</div> : <div className="empty-state">完成训练并导出策略后，这里会出现验收 Run。</div>}</section>
      <section className="platform-panel acceptance-report">{!selectedRun ? <div className="empty-panel"><ListChecks size={28} /><strong>选择一个 Run 查看验收</strong><span>报告会展示每个 seed 的结果、硬阈值和失败归因。</span></div> : loading ? <div className="empty-panel"><LoaderCircle size={22} className="spin" /><span>正在加载验收报告</span></div> : error ? <div className="empty-panel error-state"><TriangleAlert size={22} /><strong>报告暂不可用</strong><span>{error.message || `${robotLabel} 尚未生成 sim2sim 报告。`}</span><button className="button button-light" onClick={() => onNavigate("training")}>回到训练监控</button></div> : <><div className="report-header"><div><span className="panel-kicker">{robotLabel} · {sim2simLabel} · {report?.backend || "backend pending"}</span><h3>{diagnosis.title}</h3><p>{diagnosis.copy}</p></div><Badge tone={diagnosis.tone}>{report?.status || "PENDING"}</Badge></div><div className="seed-grid">{(report?.evaluations || []).map((evaluation) => <div className={`seed-card ${evaluation.status === "PASSED" ? "passed" : "failed"}`} key={evaluation.seed}><div><strong>Seed {evaluation.seed}</strong><Badge tone={evaluation.status === "PASSED" ? "green" : "red"}>{evaluation.status}</Badge></div><dl><div><dt>存活率</dt><dd>{evaluation.metrics?.survival_rate != null ? `${(evaluation.metrics.survival_rate * 100).toFixed(1)}%` : "—"}</dd></div><div><dt>关节 RMSE</dt><dd>{evaluation.metrics?.joint_rmse_rad != null ? `${evaluation.metrics.joint_rmse_rad.toFixed(3)} rad` : "—"}</dd></div><div><dt>根位置</dt><dd>{evaluation.metrics?.root_position_rmse_m != null ? `${evaluation.metrics.root_position_rmse_m.toFixed(3)} m` : "—"}</dd></div><div><dt>饱和率</dt><dd>{evaluation.metrics?.saturation_ratio != null ? `${(evaluation.metrics.saturation_ratio * 100).toFixed(1)}%` : "—"}</dd></div></dl>{evaluation.failure_code && <small className="seed-failure">{evaluation.failure_code}</small>}</div>)}</div><div className="threshold-block"><div className="threshold-head"><span><ShieldCheck size={15} /> 通过阈值</span><small>Sim2SimPolicy.v1 · 平台硬约束</small></div><div className="threshold-grid">{Object.entries(report?.thresholds || {}).filter(([key]) => key !== "schema_version").map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><strong>{typeof value === "number" && value < 1 ? `${(value * 100).toFixed(0)}%` : value}</strong></div>)}</div></div>{report?.hard_failures?.length > 0 && <div className="failure-list"><strong><TriangleAlert size={15} /> 硬失败原因</strong>{report.hard_failures.map((failure) => <span key={failure}>{failure}</span>)}</div>}<div className="report-actions"><button className="button button-light" onClick={() => onNavigate("motion")}><ArrowRight size={14} /> 回到动作流水线</button><button className="button button-light" onClick={() => onNavigate("training")}><Activity size={14} /> 调整奖励并重训</button><button className="button button-dark" onClick={() => onNavigate("artifacts")}><FileArchive size={14} /> 查看策略包</button></div></>}</section>
    </div>
  </>;
}

function ArtifactsView({ runs, selectedRun, onRunSelect, onNavigate }) {
  const [artifacts, setArtifacts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  useEffect(() => {
    let cancelled = false;
    if (!selectedRun?.run_id) { setArtifacts([]); return undefined; }
    setLoading(true); setError(null);
    platformApi.listArtifacts(selectedRun.run_id).then((result) => { if (!cancelled) setArtifacts(result.items || []); }).catch((loadError) => { if (!cancelled) setError(loadError); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [selectedRun?.run_id]);
  const download = async (artifact) => {
    try { const result = await platformApi.getArtifact(artifact.artifact_id); if (result.download_url) window.open(result.download_url, "_blank", "noopener,noreferrer"); }
    catch (loadError) { setError(loadError); }
  };
  return <>
    <SectionHeader eyebrow="POLICY BUNDLE" title="策略包" description="导出产物、sim2sim 报告和校验和都来自同一个不可变 attempt。下载前请完成人工复核。" action={<button className="button button-light" onClick={() => onNavigate("sim2sim")}><ListChecks size={14} /> 查看验收报告</button>} />
    <div className="artifact-layout"><section className="platform-panel run-picker"><SectionHeader eyebrow="RUNS" title="选择 Run" />{runs.length ? <div className="run-list">{runs.map((run) => <button className={`run-row ${selectedRun?.run_id === run.run_id ? "selected" : ""}`} key={run.run_id} onClick={() => onRunSelect(run)}><Badge tone={statusTone(run.status)}>{RUN_STATUS_LABELS[run.status] || run.status}</Badge><span><strong>{run.run_id.slice(0, 12)}</strong><small>{dateLabel(run.updated_at)}</small></span><ArrowRight size={14} /></button>)}</div> : <div className="empty-state">尚未有可导出的策略 Run</div>}</section><section className="platform-panel artifact-panel">{!selectedRun ? <div className="empty-panel"><FileArchive size={28} /><strong>选择一个 Run 查看产物</strong><span>策略包会在 sim2sim 通过后进入可下载状态。</span></div> : <><div className="artifact-summary"><div><span className="panel-kicker">RUN {selectedRun.run_id.slice(0, 12)}</span><h3>{selectedRun.status === "READY_TO_DOWNLOAD" ? "策略包已就绪" : "产物清单"}</h3></div><Badge tone={selectedRun.status === "READY_TO_DOWNLOAD" ? "green" : "amber"}>{RUN_STATUS_LABELS[selectedRun.status] || selectedRun.status}</Badge></div><div className="safety-callout"><ShieldAlert size={18} /><div><strong>安全提示</strong><span>策略包不代表真实机器人安全可用。部署前必须完成适配器、控制周期和人工复核。</span></div></div>{loading ? <div className="empty-panel"><LoaderCircle size={22} className="spin" /><span>正在读取产物</span></div> : error ? <div className="empty-panel error-state"><TriangleAlert size={22} /><span>{error.message}</span></div> : <div className="artifact-list">{artifacts.length ? artifacts.map((artifact) => <div className="artifact-row" key={artifact.artifact_id}><span className="artifact-icon"><FileArchive size={16} /></span><div><strong>{artifact.kind}</strong><small>{artifact.object_key}</small></div><code>{artifact.sha256?.slice(0, 12)}…</code><span className="artifact-size">{formatBytes(artifact.size_bytes)}</span><button className="icon-button small" title={`下载 ${artifact.kind}`} onClick={() => download(artifact)}><Download size={14} /></button></div>) : <div className="empty-state">该 Run 暂无产物，先在训练监控中提交导出。</div>}</div>}<div className="artifact-footer"><span><CheckCircle2 size={14} /> checksums.sha256 随包生成</span><span><Info size={14} /> attempt 不可覆盖</span></div></>}</section></div>
  </>;
}

async function hashJson(value) {
  const canonicalize = (item) => {
    if (Array.isArray(item)) return item.map(canonicalize);
    if (item && typeof item === "object") return Object.fromEntries(Object.keys(item).sort().map((key) => [key, canonicalize(item[key])]));
    return item;
  };
  const bytes = new TextEncoder().encode(JSON.stringify(canonicalize(value)));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((item) => item.toString(16).padStart(2, "0")).join("");
}

export default function PlatformWorkbench({ onOpenEditor }) {
  const [view, setView] = useState("overview");
  const [health, setHealth] = useState(null);
  const [infrastructure, setInfrastructure] = useState(null);
  const [projects, setProjects] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [robots, setRobots] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [selectedRobotId, setSelectedRobotId] = useState("");
  const [robotCheck, setRobotCheck] = useState(null);
  const [rewardTemplates, setRewardTemplates] = useState([]);
  const [assets, setAssets] = useState([]);
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [detection, setDetection] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = async () => {
    setLoading(true); setError(null);
    try {
      const [nextHealth, nextInfra, nextProjects, nextRobots] = await Promise.all([platformApi.health(), platformApi.infrastructureHealth(), platformApi.listProjects(), platformApi.listRobots()]);
      setHealth(nextHealth); setInfrastructure(nextInfra); setProjects(nextProjects.items || []); setRobots(nextRobots.items || []);
      const activeRobotId = nextRobots.items?.find((item) => item.robot_id === selectedRobotId)?.robot_id || nextRobots.items?.[0]?.robot_id || "";
      setSelectedRobotId(activeRobotId);
      let activeTasks = [];
      try { activeTasks = (await platformApi.listTasks(activeRobotId)).items || []; setTasks(activeTasks); } catch { setTasks([]); }
      try { setRewardTemplates((await platformApi.rewardTemplates(activeRobotId, activeTasks[0]?.task_id)).items || []); } catch { setRewardTemplates([]); }
      if (activeRobotId) {
        try { setRobotCheck((await platformApi.robotSelfCheck(activeRobotId)).result); } catch { setRobotCheck(null); }
      } else setRobotCheck(null);
      setSelectedProjectId((current) => current || nextProjects.items?.[0]?.project_id || "");
    } catch (loadError) { setError(loadError); }
    finally { setLoading(false); }
  };

  useEffect(() => { refresh(); }, [selectedRobotId]);

  useEffect(() => {
    if (!selectedProjectId) { setAssets([]); setRuns([]); return undefined; }
    let cancelled = false;
    Promise.all([platformApi.listProjectAssets(selectedProjectId), platformApi.listProjectRuns(selectedProjectId)])
      .then(([assetResult, runResult]) => {
        if (cancelled) return;
        const flattened = (assetResult.items || []).map(({ asset, versions }) => ({ ...asset, version: versions?.[versions.length - 1] || null }));
        setAssets(flattened);
        setRuns(runResult.items || []);
      })
      .catch((loadError) => { if (!cancelled) setError(loadError); });
    return () => { cancelled = true; };
  }, [selectedProjectId]);

  const refreshProject = async () => {
    if (!selectedProjectId) return;
    try {
      const [assetResult, runResult] = await Promise.all([platformApi.listProjectAssets(selectedProjectId), platformApi.listProjectRuns(selectedProjectId)]);
      const flattened = (assetResult.items || []).map(({ asset, versions }) => ({ ...asset, version: versions?.[versions.length - 1] || null }));
      setAssets(flattened);
      setRuns(runResult.items || []);
    } catch (loadError) { setError(loadError); }
  };

  useEffect(() => {
    if (!selectedProjectId || !assets.some((asset) => ["UPLOADING", "VALIDATING"].includes(asset.version?.status))) return undefined;
    const timer = window.setInterval(refreshProject, 2000);
    return () => window.clearInterval(timer);
  }, [selectedProjectId, assets]);

  const createProject = (project) => { setProjects((current) => [project, ...current.filter((item) => item.project_id !== project.project_id)]); setSelectedProjectId(project.project_id); };
  const addAsset = (asset) => {
    const normalized = asset.item ? { ...asset.item, version: asset.version, completed: asset.completed, sha256: asset.sha256 } : asset;
    setAssets((current) => [normalized, ...current.filter((item) => item.asset_id !== normalized.asset_id)]);
  };
  const deleteAsset = (assetVersionId) => {
    setAssets((current) => current.filter((item) => item.version?.asset_version_id !== assetVersionId));
  };
  const deleteProject = (projectId) => {
    setProjects((current) => current.filter((item) => item.project_id !== projectId));
    if (selectedProjectId === projectId) {
      setSelectedProjectId("");
      setAssets([]);
      setRuns([]);
      setSelectedRun(null);
    }
  };
  const addRun = (run) => { setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]); setSelectedRun(run); setView("training"); };
  const selectRun = async (run) => {
    if (run?.item) { setSelectedRun(run.item); return; }
    setSelectedRun(run);
    try {
      setSelectedRun((await platformApi.getRun(run.run_id)).item);
    } catch (loadError) {
      if (loadError?.status === 404 || loadError?.code === "RUN_NOT_FOUND") {
        setSelectedRun(null);
        setError({ message: "该 Run 已不存在，请从当前项目重新选择运行任务。" });
      } else setError(loadError);
    }
  };

  const content = useMemo(() => {
    if (view === "assets") return <Assets projects={projects} selectedProjectId={selectedProjectId} onSelectProject={setSelectedProjectId} onProjectCreated={createProject} onAssetCreated={addAsset} onAssetDeleted={deleteAsset} onProjectDeleted={deleteProject} assets={assets} onNavigate={(target) => setView(target)} />;
    if (view === "motion") return <MotionPipeline assets={assets} robots={robots} selectedRobotId={selectedRobotId} onRobotSelect={setSelectedRobotId} detection={detection} onDetect={setDetection} onOpenEditor={onOpenEditor} onRefresh={refreshProject} />;
    if (view === "training") return <Training projects={projects} selectedProjectId={selectedProjectId} assets={assets} robots={robots} tasks={tasks} selectedRobotId={selectedRobotId} onRobotSelect={setSelectedRobotId} rewardTemplates={rewardTemplates} runs={runs} selectedRun={selectedRun} onRunCreated={addRun} onRunSelect={selectRun} onRefresh={refresh} onProjectRefresh={refreshProject} />;
    if (view === "sim2sim") return <Sim2SimView runs={runs} robots={robots} selectedRun={selectedRun} onRunSelect={selectRun} onNavigate={setView} />;
    if (view === "artifacts") return <ArtifactsView runs={runs} selectedRun={selectedRun} onRunSelect={selectRun} onNavigate={setView} />;
    return <Overview health={health} infrastructure={infrastructure} projects={projects} robots={robots} robotCheck={robotCheck} runs={runs} onNavigate={(target, id) => { setView(target); if (id) selectRun({ run_id: id }); }} onRefresh={refresh} />;
  }, [assets, detection, health, infrastructure, loading, onOpenEditor, projects, refreshProject, rewardTemplates, robotCheck, robots, runs, selectedProjectId, selectedRobotId, selectedRun, tasks, view]);

  return <div className="app-shell platform-shell"><aside className="rail"><div className="brand-mark">RL</div><div className="rail-line" />{NAV_ITEMS.map(({ id, icon: Icon, label }) => <button key={id} className={`rail-button ${view === id ? "active" : ""}`} title={label} aria-label={label} onClick={() => setView(id)}><Icon size={18} /></button>)}<div className="rail-spacer" /><button className="rail-button" title="MuJoCo 编辑器" aria-label="MuJoCo 编辑器" onClick={onOpenEditor}><Play size={18} /></button><button className="rail-button" title="设置" aria-label="设置"><Settings2 size={18} /></button><div className="rail-user" aria-label="当前用户">HY</div></aside><main className="main-shell"><header className="topbar"><div className="crumb"><span className="crumb-muted">AllRobotRLLLab</span><span className="crumb-divider">/</span><strong>{NAV_ITEMS.find((item) => item.id === view)?.label || "总览"}</strong></div><div className="top-actions"><div className={`service-state ${health?.status === "ok" ? "online" : "offline"}`}><span className="live-dot" /> {loading ? "连接中" : health?.status === "ok" ? "Platform API" : "服务离线"}</div><button className="button button-light" onClick={onOpenEditor}><Play size={14} /> MuJoCo 预览</button></div></header><div className="platform-content"><ErrorBanner error={error} onDismiss={() => setError(null)} /><WorkflowStrip view={view} selectedRun={selectedRun} onNavigate={setView} /><NextAction view={view} selectedRun={selectedRun} projects={projects} selectedProjectId={selectedProjectId} assets={assets} onNavigate={setView} />{content}</div><footer className="statusbar"><div><span className="status-label">API</span><strong>{platformApi.baseUrl}</strong></div><div><span className="status-label">PROJECT</span><strong>{projects.find((item) => item.project_id === selectedProjectId)?.name || "未选择"}</strong></div><div className="status-spacer" /><div className="status-note"><CircleHelp size={14} /> Local File Mode · 后端事实源</div></footer></main></div>;
}
