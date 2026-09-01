const PLATFORM_API = import.meta.env.VITE_PLATFORM_API_BASE || "/api/v1";

export class PlatformApiError extends Error {
  constructor(message, { status = 0, code = "API_ERROR", requestId = "", details = {} } = {}) {
    super(message);
    this.name = "PlatformApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}

function normalizeError(body, status) {
  const error = body?.error || {};
  return new PlatformApiError(error.message || body?.detail || `请求失败 (${status})`, {
    status,
    code: error.code || "API_ERROR",
    requestId: body?.request_id || "",
    details: error.details || {}
  });
}

async function request(path, options = {}) {
  const headers = {
    Accept: "application/json",
    ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
    "X-User-Id": options.userId || "local-user",
    ...(options.headers || {})
  };
  const response = await fetch(`${PLATFORM_API}${path}`, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw normalizeError(body, response.status);
  return body;
}

async function uploadFile(uploadUrl, file, contentType) {
  const response = await fetch(uploadUrl, {
    method: "PUT",
    headers: { "Content-Type": contentType || file.type || "application/octet-stream" },
    body: file
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw normalizeError(body, response.status);
  }
  return response.json().catch(() => ({}));
}

async function sha256(file) {
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export const platformApi = {
  baseUrl: PLATFORM_API,
  health: () => request("/health"),
  infrastructureHealth: () => request("/health/infrastructure"),
  listProjects: () => request("/projects"),
  listProjectAssets: (projectId) => request(`/projects/${encodeURIComponent(projectId)}/assets`),
  listProjectRuns: (projectId) => request(`/projects/${encodeURIComponent(projectId)}/runs`),
  createProject: (name) => request("/projects", { method: "POST", body: JSON.stringify({ name }) }),
  listRobots: () => request("/robots"),
  robotSelfCheck: (robotId) => request(`/robots/${encodeURIComponent(robotId)}/self-check`),
  rewardTemplates: () => request("/reward-templates"),
  trainingSchema: () => request("/training-config/schema"),
  validateTraining: (config) => request("/training-config/validate", { method: "POST", body: JSON.stringify(config) }),
  createAsset: ({ projectId, kind, displayName, originalFilename, contentType, license }) => request(`/projects/${encodeURIComponent(projectId)}/assets`, {
    method: "POST",
    body: JSON.stringify({ kind, display_name: displayName, original_filename: originalFilename, content_type: contentType || null, license })
  }),
  uploadAsset: async ({ projectId, file, kind = "motion", displayName = file.name, license = { status: "declared", source: "user" }, onProgress }) => {
    const created = await platformApi.createAsset({ projectId, kind, displayName, originalFilename: file.name, contentType: file.type, license });
    const uploadUrl = created.upload?.upload_url;
    if (!uploadUrl) throw new PlatformApiError("服务未返回上传地址", { code: "UPLOAD_URL_MISSING" });
    onProgress?.(15);
    const resolvedUrl = uploadUrl.startsWith("http") ? uploadUrl : uploadUrl;
    await uploadFile(resolvedUrl, file, file.type);
    onProgress?.(75);
    const digest = await sha256(file);
    const completed = await request(`/assets/${created.version.asset_version_id}/upload-complete`, {
      method: "POST",
      body: JSON.stringify({ sha256: digest, size_bytes: file.size })
    });
    onProgress?.(100);
    return { ...created, completed, sha256: digest };
  },
  listAssetVersions: (assetId) => request(`/assets/${encodeURIComponent(assetId)}/versions`),
  detectMotion: ({ path, assetVersionId }) => request("/motions/detect", { method: "POST", body: JSON.stringify({ path: path || null, asset_version_id: assetVersionId || null }) }),
  processMotion: (assetVersionId, editConfig = null, executionMode = "async") => request(`/motions/${encodeURIComponent(assetVersionId)}/process`, { method: "POST", headers: { "X-Execution-Mode": executionMode }, body: JSON.stringify({ edit_config: editConfig }) }),
  getMotionPipeline: (assetVersionId) => request(`/motions/${encodeURIComponent(assetVersionId)}/pipeline`),
  getMotionPipelineById: (pipelineId) => request(`/motion-pipelines/${encodeURIComponent(pipelineId)}`),
  validateMotionEdit: (config) => request("/motion-edits", { method: "POST", body: JSON.stringify(config) }),
  compileMotionEdit: (versionId, arrays) => request(`/motion-edits/${encodeURIComponent(versionId)}/compile`, { method: "POST", body: JSON.stringify(arrays) }),
  validateReward: (config) => request("/reward-configs/validate", { method: "POST", body: JSON.stringify(config) }),
  createReward: (config, parentVersionId) => request(`/reward-configs${parentVersionId ? `?parent_version_id=${encodeURIComponent(parentVersionId)}` : ""}`, { method: "POST", body: JSON.stringify(config) }),
  createRun: (payload, idempotencyKey) => request("/runs", { method: "POST", headers: { "Idempotency-Key": idempotencyKey || crypto.randomUUID() }, body: JSON.stringify(payload) }),
  getRun: (runId) => request(`/runs/${encodeURIComponent(runId)}`),
  submitTraining: (runId, config) => request(`/runs/${encodeURIComponent(runId)}/train`, { method: "POST", headers: { "X-Execution-Mode": "async" }, body: JSON.stringify(config) }),
  submitExport: (runId) => request(`/runs/${encodeURIComponent(runId)}/export`, { method: "POST", headers: { "X-Execution-Mode": "async" } }),
  submitSim2sim: (runId, seeds = [20260101, 20260102, 20260103]) => request(`/runs/${encodeURIComponent(runId)}/sim2sim`, { method: "POST", headers: { "X-Execution-Mode": "async" }, body: JSON.stringify({ seeds }) }),
  getSim2simReport: (runId) => request(`/runs/${encodeURIComponent(runId)}/sim2sim`),
  cancelRun: (runId) => request(`/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }),
  retryRun: (runId) => request(`/runs/${encodeURIComponent(runId)}/retry`, { method: "POST" }),
  listArtifacts: (runId) => request(`/runs/${encodeURIComponent(runId)}/artifacts`),
  getArtifact: (artifactId) => request(`/artifacts/${encodeURIComponent(artifactId)}`),
  subscribeRun: (runId, { afterSeq = 0, onEvent, onError, signal } = {}) => {
    const controller = new AbortController();
    signal?.addEventListener("abort", () => controller.abort(), { once: true });
    const url = `${PLATFORM_API}/runs/${encodeURIComponent(runId)}/events?after_seq=${afterSeq}`;
    fetch(url, { headers: { Accept: "text/event-stream", "X-User-Id": "local-user" }, signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw normalizeError(await response.json().catch(() => ({})), response.status);
        if (!response.body) return;
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (!controller.signal.aborted) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const chunks = buffer.split("\n\n");
          buffer = chunks.pop() || "";
          chunks.forEach((chunk) => {
            const data = chunk.split("\n").find((line) => line.startsWith("data:"));
            if (data) {
              try { onEvent?.(JSON.parse(data.slice(5).trim())); } catch { /* ignore malformed keep-alive */ }
            }
          });
        }
      })
      .catch((error) => { if (!controller.signal.aborted) onError?.(error); });
    return () => controller.abort();
  }
};

export { PLATFORM_API };
