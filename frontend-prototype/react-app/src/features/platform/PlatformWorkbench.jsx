import { useEffect, useMemo, useState } from "react";
import { CircleHelp, Play, Settings2 } from "lucide-react";
import { platformApi } from "../../api/platformClient";
import { useAssetStatusSocket } from "./hooks/useStatusSockets";
import { NAV_ITEMS, ErrorBanner, WorkflowStrip, NextAction, Overview, Assets, MotionPipeline, Training, Sim2SimView, ArtifactsView } from "./PlatformViews";

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

  useAssetStatusSocket(selectedProjectId, {
    onSnapshot: (items) => setAssets((items || []).map(({ asset, versions }) => ({ ...asset, version: versions?.[versions.length - 1] || null }))),
    onFallback: refreshProject
  });

  const createProject = (project) => { setProjects((current) => [project, ...current.filter((item) => item.project_id !== project.project_id)]); setSelectedProjectId(project.project_id); };
  const addAsset = (asset) => {
    // upload-complete returns the authoritative server version (normally
    // VALIDATING). Prefer it over the creation response's stale UPLOADING
    // version so the polling effect can advance the row to READY/REJECTED.
    const normalized = asset.item ? { ...asset.item, version: asset.completed?.item || asset.version, completed: asset.completed, sha256: asset.sha256 } : asset;
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
