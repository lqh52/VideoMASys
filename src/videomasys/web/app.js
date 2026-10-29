const state = {
  videos: [],
  currentVideo: null,
  currentTask: null,
  videoPoll: null,
  taskPoll: null,
  choicesEnabled: false,
  webgpuReady: false,
  browserWorker: null,
  browserModelReady: null,
  browserBusy: false,
  browserLoadMs: null,
  browserDemo: false,
  demoExamples: [],
  demoVideos: [],
  currentDemoVideoKey: null,
  activeFlowStage: "planner",
  browserRequests: new Map(),
  browserLoadPromise: null,
  browserLoadResolve: null,
  browserLoadReject: null,
  pipelineRunning: false,
  localVideoFile: null,
  localVideoUrl: null,
  localScenes: null,
  confirmedLargeModels: new Set(),
  libraryLoading: false,
  libraryError: null,
  libraryRequest: null,
  browserCheckpoint: null,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  healthPill: $("#healthPill"), healthText: $("#healthText"),
  videoList: $("#videoList"), intro: $("#introSection"), analysis: $("#analysisSection"),
  input: $("#videoInput"), dropzone: $("#dropzone"), uploadProgress: $("#uploadProgress"),
  uploadLabel: $("#uploadLabel"), uploadPercent: $("#uploadPercent"), uploadBar: $("#uploadBar"),
  title: $("#videoTitle"), videoState: $("#videoState"), player: $("#videoPlayer"),
  videoEmpty: $("#videoEmpty"), duration: $("#metaDuration"), resolution: $("#metaResolution"),
  nodes: $("#metaNodes"), hash: $("#metaHash"), durationLabel: $("#durationLabel"),
  timelineEnd: $("#timelineEnd"), timeline: $("#timeline"), question: $("#questionInput"),
  choices: $("#choices"), toggleChoices: $("#toggleChoicesButton"), addChoice: $("#addChoiceButton"),
  run: $("#runButton"), cancel: $("#cancelButton"), pipelineCard: $("#pipelineCard"),
  pipeline: $("#pipeline"), pipelineMessage: $("#pipelineMessage"),
  pipelineDetail: $("#pipelineDetail"), results: $("#resultsSection"),
  answer: $("#answerText"), confidence: $("#answerConfidence"), answerMeta: $("#answerMeta"),
  limitations: $("#limitations"), evidenceList: $("#evidenceList"), evidenceCount: $("#evidenceCount"),
  error: $("#errorBanner"), errorText: $("#errorText"), toast: $("#toast"),
  browserStatus: $("#browserStatus"), browserModel: $("#browserModel"),
  browserModelDetail: $("#browserModelDetail"), loadBrowserModel: $("#loadBrowserModel"),
  runBrowserModel: $("#runBrowserModel"), browserProgress: $("#browserProgress"),
  browserProgressLabel: $("#browserProgressLabel"), browserProgressValue: $("#browserProgressValue"),
  browserProgressDetail: $("#browserProgressDetail"),
  browserProgressBar: $("#browserProgressBar"), localComparison: $("#localComparison"),
  localModelName: $("#localModelName"), localAnswer: $("#localAnswer"),
  localMetrics: $("#localMetrics"), browserDetails: $("#browserDetails"),
  browserContactSheet: $("#browserContactSheet"), localObservation: $("#localObservation"),
  browserLab: $("#browserLab"), demoExamplePicker: $("#demoExamplePicker"),
  demoExampleSelect: $("#demoExampleSelect"), demoExampleDataset: $("#demoExampleDataset"),
  library: $("#libraryDialog"), librarySearch: $("#librarySearch"), libraryNotice: $("#libraryNotice"),
};

const terminalTaskStates = new Set(["succeeded", "failed", "cancelled"]);
const pipelineDefinitions = [
  ["planner", "Planner", "Question → evidence needs", "plan"],
  ["tool_sequence", "Tool agent", "Needs → typed calls", "tool_sequence"],
  ["retrieve", "Video cache", "Calls → intervals", "retrieve"],
  ["inspect", "Evidence VLM", "Frames → observations", "inspect_evidence"],
  ["answer", "Answer", "Evidence → response", "compose_answer"],
  ["verify", "Verifier", "Draft → grounded result", "verify_answer"],
];

const browserModels = {
  "lfm-1.6b": {
    name: "LFM2.5-VL-1.6B",
    detail: "FP16 vision · Q4 decoder · 2.34 GB download; runtime memory is higher.",
    highMemory: true,
    loadWarning: "LFM2.5-VL-1.6B downloads 2.34 GB of weights and needs additional working memory. "
      + "It may overwhelm a low-memory computer. Load this model?",
  },
  "lfm-3b": {
    name: "LFM2.5-VL-3B",
    detail: "High-memory opt-in · model weights plus WebGPU working buffers can exceed 3 GB.",
    highMemory: true,
  },
  "qwen-4b": {
    name: "Qwen3.5-4B",
    detail: "Very-high-memory opt-in · intended for desktop GPUs with substantial free memory.",
    highMemory: true,
  },
  "lfm-450m": {
    name: "LFM2.5-VL-450M",
    detail: "Lower-memory option · faster startup, lower reasoning quality.",
    highMemory: false,
  },
};

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function formatTime(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "—";
  const total = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}` : `${minutes}:${String(secs).padStart(2, "0")}`;
}

function statusLabel(value) {
  return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* response is not JSON */ }
    throw new Error(message);
  }
  return response.json();
}

async function checkHealth() {
  try {
    const health = await api("/healthz");
    const healthy = health.status === "ok";
    elements.healthPill.className = `health-pill ${healthy ? "healthy" : "degraded"}`;
    elements.healthText.textContent = healthy ? "System ready" : "Database unavailable";
  } catch (_) {
    elements.healthPill.className = "health-pill degraded";
    elements.healthText.textContent = "API unavailable";
  }
}

async function loadVideos() {
  state.browserDemo = false;
  elements.demoExamplePicker.classList.add("hidden");
  elements.question.readOnly = false;
  elements.toggleChoices.classList.remove("hidden");
  elements.run.querySelector("span").textContent = "Run analysis";
  try {
    state.videos = await api("/v1/videos?limit=30");
    renderVideoList();
    const remembered = localStorage.getItem("videomasys.videoId");
    if (!state.currentVideo && remembered) {
      const match = state.videos.find((video) => video.id === remembered);
      if (match) selectVideo(match);
    }
    const demoRequested = new URLSearchParams(window.location.search).get("demo") === "1";
    if (!state.currentVideo && demoRequested) {
      const demoVideo = state.videos.find((video) => video.status === "ready");
      if (demoVideo) selectVideo(demoVideo);
    }
  } catch (error) {
    elements.videoList.innerHTML = `<div class="list-empty">Video history is unavailable.<br>${escapeHtml(error.message)}</div>`;
  }
}

function renderDemoExampleOptions(videoKey, preferredId = null) {
  const examples = state.demoExamples.filter((item) => item.video === videoKey);
  elements.demoExampleSelect.innerHTML = examples.map((item) =>
    `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`
  ).join("");
  const selected = examples.some((item) => item.id === preferredId) ? preferredId : examples[0]?.id;
  if (selected) elements.demoExampleSelect.value = selected;
  return selected;
}

async function loadBrowserDemoCatalog() {
  try {
    state.browserDemo = true;
    await refreshLibrary();
    elements.intro.classList.remove("hidden");
    elements.analysis.classList.add("hidden");
    elements.demoExamplePicker.classList.add("hidden");
    elements.healthPill.className = `health-pill ${state.libraryError ? "degraded" : "healthy"}`;
    elements.healthText.textContent = state.libraryError ? "Library unavailable" : "Choose a video to begin";
    const requested = new URLSearchParams(window.location.search).get("example");
    if (state.demoExamples.some((item) => item.id === requested)) {
      await loadBrowserDemo(requested);
    }
  } catch (error) {
    showError(`Could not load the video library: ${error.message}`);
  }
}

async function loadBrowserDemo(exampleId = null) {
  try {
    state.browserDemo = true;
    if (!state.demoExamples.length) {
      const catalog = await api("/v1/browser-demo/examples");
      state.demoExamples = catalog.examples;
      state.demoVideos = catalog.videos;
      if (!exampleId) {
        const requested = new URLSearchParams(window.location.search).get("example");
        exampleId = state.demoExamples.some((item) => item.id === requested) ? requested : catalog.default;
      }
      const initial = state.demoExamples.find((item) => item.id === exampleId) || state.demoExamples[0];
      exampleId = renderDemoExampleOptions(initial.video, initial.id);
    }
    const selectedId = exampleId || elements.demoExampleSelect.value || state.demoExamples[0]?.id;
    const demo = await api(`/v1/browser-demo?example=${encodeURIComponent(selectedId)}`);
    const selected = state.demoExamples.find((item) => item.id === demo.example_id);
    state.currentDemoVideoKey = selected.video;
    renderDemoExampleOptions(selected.video, demo.example_id);
    elements.demoExampleDataset.textContent = selected?.dataset || "Evaluation";
    state.videos = [demo.video];
    state.currentVideo = demo.video;
    state.currentTask = demo.task;
    state.localVideoFile = null;
    state.localScenes = null;
    resetBrowserResult();
    elements.intro.classList.add("hidden");
    elements.analysis.classList.remove("hidden");
    elements.pipelineCard.classList.add("hidden");
    elements.results.classList.add("hidden");
    elements.demoExamplePicker.classList.remove("hidden");
    elements.question.value = demo.task.request.question;
    elements.question.readOnly = true;
    elements.choices.innerHTML = "";
    enableChoices();
    elements.choices.innerHTML = "";
    demo.task.request.options.forEach(addChoice);
    elements.choices.querySelectorAll("input, button").forEach((control) => { control.disabled = true; });
    elements.toggleChoices.classList.add("hidden");
    elements.addChoice.classList.add("hidden");
    elements.run.querySelector("span").textContent = "Run live analysis";
    renderVideo(demo.video);
    renderVideoList();
    elements.healthPill.className = "health-pill healthy";
    elements.healthText.textContent = "Live browser pipeline ready";
  } catch (error) {
    showError(`Could not load browser demo: ${error.message}`);
  }
}

function readLocalVideoMetadata(url) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.onloadedmetadata = () => resolve({
      duration: video.duration,
      width: video.videoWidth,
      height: video.videoHeight,
    });
    video.onerror = () => reject(new Error("The browser could not decode this video."));
    video.src = url;
  });
}

async function loadLocalBrowserVideo(file) {
  state.browserCheckpoint = null;
  if (state.localVideoUrl) URL.revokeObjectURL(state.localVideoUrl);
  const url = URL.createObjectURL(file);
  const metadata = await readLocalVideoMetadata(url);
  state.localVideoFile = file;
  state.localVideoUrl = url;
  state.localScenes = null;
  state.currentDemoVideoKey = "local";
  state.currentVideo = {
    id: `local-${file.name}-${file.size}-${file.lastModified}`,
    filename: file.name,
    content_hash: `local-${file.size}-${file.lastModified}`,
    status: "ready",
    duration_seconds: metadata.duration,
    width: metadata.width,
    height: metadata.height,
    node_count: Math.min(24, Math.ceil(metadata.duration / Math.max(20, metadata.duration / 24))),
    content_url: url,
    local: true,
  };
  state.currentTask = null;
  elements.intro.classList.add("hidden");
  elements.analysis.classList.remove("hidden");
  elements.pipelineCard.classList.add("hidden");
  elements.results.classList.add("hidden");
  elements.demoExamplePicker.classList.add("hidden");
  elements.question.readOnly = false;
  elements.question.value = "";
  elements.choices.innerHTML = "";
  disableChoices();
  elements.toggleChoices.classList.remove("hidden");
  elements.run.querySelector("span").textContent = "Run live analysis";
  renderVideo(state.currentVideo);
  renderVideoList();
  elements.question.focus();
  toast("Video stays local — enter a question, then run");
}

function renderVideoList() {
  const search = elements.librarySearch.value.trim().toLowerCase();
  const cached = state.browserDemo ? state.demoVideos.length : state.videos.length;
  elements.libraryNotice.textContent = state.libraryLoading ? "Loading videos…"
    : state.libraryError ? (cached ? "Couldn't refresh. Showing previously loaded videos." : "Couldn't load the library. Try Refresh library.") : "";
  elements.libraryNotice.classList.toggle("hidden", !elements.libraryNotice.textContent);
  $("#refreshVideosButton").disabled = state.libraryLoading;
  if (!cached && (state.libraryLoading || state.libraryError)) {
    elements.videoList.innerHTML = "";
    return;
  }
  if (state.browserDemo && state.demoVideos.length) {
    const localVideo = state.currentVideo?.local
      ? [{ key: "local", label: state.currentVideo.filename, dataset: "Local", question_count: 0 }]
      : [];
    elements.videoList.innerHTML = [...localVideo, ...state.demoVideos]
      .filter((video) => `${video.label} ${video.dataset}`.toLowerCase().includes(search)).map((video) => `
      <button class="video-item ${state.currentDemoVideoKey === video.key ? "active" : ""}" data-demo-video-key="${escapeHtml(video.key)}" type="button">
        <span class="video-thumb">${video.key === "local"
          ? '<svg viewBox="0 0 24 24"><path d="m10 8 6 4-6 4V8Z"/></svg>'
          : `<img src="/v1/browser-demo/assets/${encodeURIComponent(video.key)}/frames/0/0" alt="" loading="lazy">`}</span>
        <span class="video-copy"><strong>${escapeHtml(video.label)}</strong><span>${escapeHtml(video.dataset)}${video.key === "local" ? " · Your video" : ` · ${video.question_count} question${video.question_count === 1 ? "" : "s"}`}</span></span>
      </button>`).join("");
    if (!elements.videoList.children.length) elements.videoList.innerHTML = '<div class="list-empty">No matching videos.</div>';
    elements.videoList.querySelectorAll("[data-demo-video-key]").forEach((button) => {
      button.addEventListener("click", () => {
        elements.library.close();
        if (button.dataset.demoVideoKey === "local") { elements.question.focus(); return; }
        const exampleId = renderDemoExampleOptions(button.dataset.demoVideoKey);
        if (exampleId) loadBrowserDemo(exampleId).then(() => elements.demoExampleSelect.focus());
      });
    });
    return;
  }
  if (!state.videos.length) {
    elements.videoList.innerHTML = '<div class="list-empty">No videos available yet. Add your own video to begin.</div>';
    return;
  }
  elements.videoList.innerHTML = state.videos.filter((video) => video.filename.toLowerCase().includes(search)).map((video) => `
    <button class="video-item ${state.currentVideo?.id === video.id ? "active" : ""}" data-video-id="${video.id}" type="button">
      <span class="video-thumb"><svg viewBox="0 0 24 24"><path d="m10 8 6 4-6 4V8Z"/><rect x="3" y="5" width="18" height="14" rx="3"/></svg></span>
      <span class="video-copy"><strong>${escapeHtml(video.filename)}</strong><span>${formatTime(video.duration_seconds)} · ${statusLabel(video.status)}</span></span>
    </button>`).join("");
  elements.videoList.querySelectorAll("[data-video-id]").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.browserDemo) return;
      elements.library.close();
      selectVideo(state.videos.find((item) => item.id === button.dataset.videoId));
    });
  });
}

function resetWorkspace() {
  if (state.pipelineRunning || state.browserBusy) { toast("Stop the current run before starting a new analysis."); return; }
  state.browserCheckpoint = null;
  elements.player.pause();
  if (state.browserDemo) {
    state.currentVideo = null;
    state.currentTask = null;
    state.currentDemoVideoKey = null;
    state.localScenes = null;
    elements.intro.classList.remove("hidden");
    elements.analysis.classList.add("hidden");
    elements.results.classList.add("hidden");
    elements.pipelineCard.classList.add("hidden");
    elements.demoExamplePicker.classList.add("hidden");
    renderVideoList();
    return;
  }
  clearInterval(state.videoPoll);
  clearInterval(state.taskPoll);
  state.currentVideo = null;
  state.currentTask = null;
  localStorage.removeItem("videomasys.videoId");
  elements.intro.classList.remove("hidden");
  elements.analysis.classList.add("hidden");
  elements.results.classList.add("hidden");
  elements.pipelineCard.classList.add("hidden");
  elements.question.value = "";
  resetBrowserResult();
  renderVideoList();
}

async function selectVideo(video) {
  if (!video) return;
  clearInterval(state.videoPoll);
  clearInterval(state.taskPoll);
  state.currentVideo = video;
  state.currentTask = null;
  localStorage.setItem("videomasys.videoId", video.id);
  elements.intro.classList.add("hidden");
  elements.analysis.classList.remove("hidden");
  elements.results.classList.add("hidden");
  elements.pipelineCard.classList.add("hidden");
  resetBrowserResult();
  renderVideo(video);
  renderVideoList();
  loadLatestTask(video.id);
  if (!["ready", "failed"].includes(video.status)) {
    state.videoPoll = setInterval(() => refreshCurrentVideo(), 2000);
  }
}

function renderVideo(video) {
  elements.title.textContent = video.filename.split(" · ").at(-1);
  elements.title.title = video.filename;
  elements.videoState.className = `video-state ${video.status}`;
  elements.videoState.innerHTML = `<span></span><b>${statusLabel(video.status)}</b>`;
  elements.duration.textContent = formatTime(video.duration_seconds);
  elements.resolution.textContent = video.width && video.height ? `${video.width} × ${video.height}` : "Pending";
  elements.nodes.textContent = video.node_count ?? "—";
  elements.hash.textContent = video.content_hash ? video.content_hash.slice(0, 10) : "—";
  elements.durationLabel.textContent = video.duration_seconds ? formatTime(video.duration_seconds) : "Indexing";
  elements.timelineEnd.textContent = formatTime(video.duration_seconds);
  elements.run.disabled = video.status !== "ready" || elements.question.value.trim().length < 2;
  if (elements.player.dataset.videoId !== video.id) {
    elements.videoEmpty.querySelector("span").textContent = "Preparing preview…";
    elements.videoEmpty.classList.remove("hidden");
    elements.player.src = video.content_url || `/v1/videos/${video.id}/content`;
    elements.player.dataset.videoId = video.id;
  }
  if (video.error) showError(video.error);
}

async function refreshCurrentVideo() {
  if (!state.currentVideo) return;
  try {
    const video = await api(`/v1/videos/${state.currentVideo.id}`);
    state.currentVideo = video;
    state.videos = state.videos.map((item) => item.id === video.id ? video : item);
    renderVideo(video);
    renderVideoList();
    if (["ready", "failed"].includes(video.status)) {
      clearInterval(state.videoPoll);
      toast(video.status === "ready" ? "Video cache is ready" : "Video indexing failed");
    }
  } catch (error) { showError(error.message); }
}

function uploadVideo(file) {
  if (!file || !file.type.startsWith("video/")) {
    showError("Choose a supported video file.");
    return;
  }
  if (state.browserDemo) {
    loadLocalBrowserVideo(file).catch((error) => showError(error.message));
    return;
  }
  const body = new FormData();
  body.append("file", file);
  const request = new XMLHttpRequest();
  elements.uploadProgress.classList.remove("hidden");
  elements.uploadLabel.textContent = `Uploading ${file.name}`;
  request.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round((event.loaded / event.total) * 100);
    elements.uploadPercent.textContent = `${percent}%`;
    elements.uploadBar.style.width = `${percent}%`;
  });
  request.addEventListener("load", async () => {
    if (request.status < 200 || request.status >= 300) {
      let message = "Upload failed.";
      try { message = JSON.parse(request.responseText).detail || message; } catch (_) { /* not JSON */ }
      showError(message);
      return;
    }
    const video = JSON.parse(request.responseText);
    elements.uploadLabel.textContent = "Upload complete · indexing queued";
    state.videos = [video, ...state.videos.filter((item) => item.id !== video.id)];
    await selectVideo(video);
    setTimeout(() => elements.uploadProgress.classList.add("hidden"), 800);
  });
  request.addEventListener("error", () => showError("The upload connection was interrupted."));
  request.open("POST", "/v1/videos");
  request.send(body);
}

function enableChoices() {
  state.choicesEnabled = true;
  elements.choices.classList.remove("hidden");
  elements.addChoice.classList.remove("hidden");
  elements.toggleChoices.textContent = "Remove choices";
  if (!elements.choices.children.length) {
    addChoice(); addChoice(); addChoice(); addChoice();
  }
}

function disableChoices() {
  state.choicesEnabled = false;
  elements.choices.classList.add("hidden");
  elements.addChoice.classList.add("hidden");
  elements.toggleChoices.textContent = "+ Add choices";
}

function addChoice(value = "") {
  if (elements.choices.children.length >= 10) return;
  const row = document.createElement("div");
  row.className = "choice-row";
  row.innerHTML = `<span class="choice-index"></span><input class="choice-input" maxlength="500" placeholder="Answer choice" value="${escapeHtml(value)}"><button class="remove-choice" type="button" aria-label="Remove choice">×</button>`;
  row.querySelector("button").addEventListener("click", () => { row.remove(); renumberChoices(); });
  elements.choices.appendChild(row);
  renumberChoices();
}

function renumberChoices() {
  elements.choices.querySelectorAll(".choice-index").forEach((label, index) => { label.textContent = String(index); });
}

function getChoices() {
  if (!state.choicesEnabled) return null;
  return [...elements.choices.querySelectorAll("input")].map((input) => input.value.trim()).filter(Boolean);
}

async function createTask() {
  if (state.browserDemo) {
    await runLiveBrowserPipeline();
    return;
  }
  const question = elements.question.value.trim();
  const options = getChoices();
  if (question.length < 2 || state.currentVideo?.status !== "ready") return;
  if (state.choicesEnabled && (!options || options.length < 2)) {
    showError("Multiple-choice questions need at least two non-empty choices.");
    return;
  }
  clearInterval(state.taskPoll);
  elements.run.disabled = true;
  elements.cancel.classList.remove("hidden");
  elements.results.classList.add("hidden");
  elements.pipelineCard.classList.remove("hidden");
  renderPipeline({ status: "queued", steps: [] });
  try {
    const task = await api("/v1/qa", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: state.currentVideo.id, question, options }),
    });
    state.currentTask = task;
    localStorage.setItem("videomasys.taskId", task.id);
    renderPipeline(task);
    state.taskPoll = setInterval(refreshTask, 1500);
  } catch (error) {
    elements.run.disabled = false;
    elements.cancel.classList.add("hidden");
    showError(error.message);
  }
}

async function runLiveStep(task, name, flowStage, operation) {
  if (task.status === "cancelled") throw new Error("Run cancelled by user.");
  if (!task._stageValues) Object.defineProperty(task, "_stageValues", { value: new Map() });
  if (task._stageValues.has(name)) return task._stageValues.get(name);
  const previous = task.steps.find((item) => item.name === name);
  const step = previous || {
    name,
    status: "running",
    started_at: new Date().toISOString(),
    finished_at: null,
    latency_ms: null,
    details: {},
  };
  if (!previous) task.steps.push(step);
  step.status = "running";
  step.started_at = new Date().toISOString();
  step.finished_at = null;
  delete step.details.error;
  state.activeFlowStage = flowStage;
  const labels = {
    plan: "Planning the search",
    tool_sequence: "Preparing the search",
    retrieve: "Finding relevant moments",
    inspect_evidence: "Looking at the evidence",
    compose_answer: "Writing the answer",
    verify_answer: "Checking the answer",
  };
  task.status_message = labels[name];
  setBrowserProgress(labels[name], "", `Step ${task.steps.length} of 6`);
  renderPipeline(task);
  const started = performance.now();
  try {
    const output = await operation();
    if (task.status === "cancelled") throw new Error("Run cancelled by user.");
    step.status = "succeeded";
    step.finished_at = new Date().toISOString();
    step.latency_ms = Math.round(performance.now() - started);
    step.details = { ...step.details, ...output.details };
    task._stageValues.set(name, output.value);
    renderPipeline(task);
    return output.value;
  } catch (error) {
    step.status = "failed";
    step.finished_at = new Date().toISOString();
    step.latency_ms = Math.round(performance.now() - started);
    step.details = { ...step.details, error: error.message };
    throw error;
  }
}

async function runLiveBrowserPipeline() {
  const live = window.VideoMASysLive;
  const question = elements.question.value.trim();
  const options = getChoices();
  if (!live) {
    showError("The live browser pipeline did not load. Refresh the page and try again.");
    return;
  }
  if (question.length < 2 || state.currentVideo?.status !== "ready") return;
  if (state.choicesEnabled && (!options || options.length < 2)) {
    showError("Multiple-choice questions need at least two non-empty choices.");
    return;
  }
  if (!state.webgpuReady) {
    showError("Live browser analysis requires WebGPU from localhost or HTTPS.");
    return;
  }

  elements.run.disabled = true;
  elements.error.classList.add("hidden");
  elements.cancel.classList.remove("hidden");
  elements.results.classList.add("hidden");
  elements.pipelineCard.classList.remove("hidden");
  state.pipelineRunning = true;
  const signature = JSON.stringify([state.currentVideo.id, question, options, elements.browserModel.value]);
  const resume = state.browserCheckpoint?.signature === signature
    && state.browserCheckpoint.task.status === "failed";
  const task = resume ? state.browserCheckpoint.task : {
    id: crypto.randomUUID(),
    video_id: state.currentVideo.id,
    task_type: "qa",
    status: "processing",
    request: { question, options },
    plan: null,
    result: null,
    evidence: [],
    steps: [],
    provenance: { mode: "live_browser", model: elements.browserModel.value },
  };
  task.status = "processing";
  task.error = null;
  state.browserCheckpoint = null;
  state.currentTask = task;
  state.activeFlowStage = "planner";
  renderPipeline(task);

  try {
    task.status_message = "Loading model";
    renderPipeline(task);
    await ensureBrowserModelLoaded();
    state.browserBusy = true;
    setBrowserStatus("busy", "Live agents running");

    const reasoningPlan = await runLiveStep(task, "plan", "planner", async () => {
      const response = await inferJsonAgent(
        live.plannerPrompt(question, options, state.currentVideo.duration_seconds),
        null,
        700,
        (raw) => live.validatePlan(raw, state.currentVideo.duration_seconds),
      );
      const value = response.data;
      task.plan = { reasoning_plan: value, tool_program: null };
      return {
        value,
        details: {
          provenance: "generated_live_by_selected_browser_model",
          model_inference_ms: response.inferenceMs,
          raw_output: response.raw,
          generated_plan: value,
          repaired: Boolean(response.repaired),
        },
      };
    });

    const toolProgram = await runLiveStep(task, "tool_sequence", "tool_sequence", async () => {
      const response = await inferJsonAgent(
        live.toolProgramPrompt(question, options, reasoningPlan),
        null,
        800,
        (raw) => live.validateProgram(raw, reasoningPlan),
      );
      const value = response.data;
      task.plan.tool_program = value;
      return {
        value,
        details: {
          provenance: "generated_live_by_selected_browser_model",
          model_inference_ms: response.inferenceMs,
          raw_output: response.raw,
          generated_program: value,
          repaired: Boolean(response.repaired),
        },
      };
    });

    const retrieval = await runLiveStep(task, "retrieve", "retrieve", async () => {
      const rawScenes = await getBrowserScenes();
      const captioned = await captionScenes(rawScenes);
      setBrowserProgress("Searching the video", "", "Step 3 of 6");
      let embedding = null;
      let embeddingError = null;
      try {
        const texts = [
          ...captioned.scenes.map((scene) => scene.caption),
          ...toolProgram.locate.map((call) => call.query),
        ];
        const response = await embedBrowserTexts(texts);
        embedding = {
          sceneVectors: response.vectors.slice(0, captioned.scenes.length),
          queryVectors: response.vectors.slice(captioned.scenes.length),
          dimensions: response.dimensions,
          modelId: response.modelId,
          inferenceMs: response.inferenceMs,
        };
      } catch (error) {
        embeddingError = error.message;
      }
      const value = live.executeRetrieval(toolProgram, captioned.scenes, embedding);
      value.scenes = captioned.scenes;
      return {
        value,
        details: {
          provenance: "computed_live_in_browser",
          caption_cache_hit: captioned.cacheHit,
          caption_model_ms: captioned.modelMs,
          vector_index: value.index,
          embedding_inference_ms: embedding?.inferenceMs || null,
          embedding_fallback_reason: embeddingError,
          tool_trace: value.trace,
          scene_captions: captioned.scenes.map((scene) => ({ index: scene.index, caption: scene.caption })),
          selected_scene_indices: value.selected.map((scene) => scene.index),
        },
      };
    });
    if (!retrieval.selected.length) throw new Error("The typed retrieval program selected no evidence intervals.");

    const evidence = await runLiveStep(task, "inspect_evidence", "inspect", async () => {
      const frames = framesForScenes(retrieval.selected, 2);
      const sheet = await buildEvidenceContactSheet(task, frames);
      const selectedWithLabels = retrieval.selected.map((scene) => ({
        ...scene,
        labels: frames.map((frame, index) => frame.sceneIndex === scene.index ? sheet.labels[index].split("  ")[0] : null).filter(Boolean),
      }));
      const response = await inferJsonAgent(
        live.inspectorPrompt(question, options, selectedWithLabels, sheet.labels),
        sheet.imageUrl,
        750,
        (raw) => {
          if (!Array.isArray(raw.evidence)) throw new Error("Evidence inspector returned no evidence list.");
          const assessments = new Map(raw.evidence.map((item) => [Number(item.scene_index), item]));
          for (const scene of retrieval.selected) {
            if (!String(assessments.get(scene.index)?.observation || "").trim()) {
              throw new Error(`Evidence inspector omitted retrieved scene ${scene.index}.`);
            }
          }
          return raw;
        },
      );
      if (!Array.isArray(response.data.evidence)) throw new Error("Evidence inspector returned no evidence list.");
      const assessments = new Map(response.data.evidence.map((item) => [Number(item.scene_index), item]));
      const value = retrieval.selected.map((scene) => {
        const assessment = assessments.get(scene.index);
        if (!assessment || !String(assessment.observation || "").trim()) {
          throw new Error(`Evidence inspector omitted retrieved scene ${scene.index}.`);
        }
        return {
          node_id: `browser-${task.id}-scene-${scene.index}`,
          scene_index: scene.index,
          start_seconds: scene.start_seconds,
          end_seconds: scene.end_seconds,
          observation: String(assessment.observation).trim(),
          confidence: Math.max(0, Math.min(1, Number(assessment.confidence) || 0)),
          relevant: assessment.relevant !== false,
          frame_paths: scene.frame_urls,
          frame_urls: scene.frame_urls,
        };
      }).filter((item) => item.relevant);
      task.evidence = value;
      return {
        value,
        details: {
          provenance: "generated_live_from_retrieved_pixels",
          model_inference_ms: response.inferenceMs,
          inspected_scene_indices: retrieval.selected.map((scene) => scene.index),
          retained_scene_indices: value.map((item) => item.scene_index),
          raw_output: response.raw,
          generated_evidence: value.map((item) => ({
            scene_index: item.scene_index,
            observation: item.observation,
            confidence: item.confidence,
          })),
        },
      };
    });

    const draft = await runLiveStep(task, "compose_answer", "answer", async () => {
      const response = await inferJsonAgent(
        live.answerPrompt(question, options, evidence),
        null,
        650,
        (raw) => {
          if (!String(raw.answer || "").trim()) throw new Error("Answer composer returned an empty answer.");
          const selectedOption = raw.selected_option == null ? null : Number(raw.selected_option);
          if (options && (!Number.isInteger(selectedOption) || selectedOption < 0 || selectedOption >= options.length)) {
            throw new Error("Answer composer did not return a valid zero-based option index.");
          }
          return raw;
        },
      );
      if (!String(response.data.answer || "").trim()) throw new Error("Answer composer returned an empty answer.");
      const parsedOption = response.data.selected_option == null
        ? null : Number(response.data.selected_option);
      const value = {
        answer: String(response.data.answer).trim(),
        selected_option: Number.isInteger(parsedOption) ? parsedOption : null,
        confidence: Math.max(0, Math.min(1, Number(response.data.confidence) || 0)),
        supporting_scene_indices: Array.isArray(response.data.supporting_scene_indices)
          ? response.data.supporting_scene_indices.map(Number) : [],
        limitations: Array.isArray(response.data.limitations) ? response.data.limitations.map(String) : [],
      };
      if (options && (value.selected_option == null || value.selected_option >= options.length)) {
        throw new Error("Answer composer did not return a valid zero-based option index.");
      }
      return {
        value,
        details: {
          provenance: "generated_live_by_selected_browser_model",
          model_inference_ms: response.inferenceMs,
          raw_output: response.raw,
          generated_answer: value,
        },
      };
    });

    const verification = await runLiveStep(task, "verify_answer", "verify", async () => {
      const response = await inferJsonAgent(
        live.verifierPrompt(question, options, evidence, draft),
        null,
        550,
        (raw) => {
          if (typeof raw.accepted !== "boolean") throw new Error("Verifier returned no accepted decision.");
          return raw;
        },
      );
      if (typeof response.data.accepted !== "boolean") throw new Error("Verifier returned no accepted decision.");
      return {
        value: response.data,
        details: {
          provenance: "generated_live_by_selected_browser_model",
          model_inference_ms: response.inferenceMs,
          raw_output: response.raw,
          generated_verification: response.data,
        },
      };
    });

    const corrected = !verification.accepted && String(verification.corrected_answer || "").trim();
    const correctedOption = verification.corrected_selected_option == null
      ? null : Number(verification.corrected_selected_option);
    const selectedOption = corrected && Number.isInteger(correctedOption)
      ? correctedOption : draft.selected_option;
    const rawFinalAnswer = corrected ? String(verification.corrected_answer) : draft.answer;
    const finalAnswer = options && Number.isInteger(selectedOption) && selectedOption < options.length
      ? `[${selectedOption}] ${options[selectedOption]}` : rawFinalAnswer;
    task.result = {
      answer: finalAnswer,
      selected_option: selectedOption,
      confidence: Math.min(draft.confidence, Math.max(0, Math.min(1, Number(verification.confidence) || 0))),
      citations: evidence.filter((item) => draft.supporting_scene_indices.includes(item.scene_index)).map((item) => ({
        node_id: item.node_id,
        start_seconds: item.start_seconds,
        end_seconds: item.end_seconds,
      })),
      limitations: draft.limitations,
      verified: verification.accepted,
      corrected_by_verifier: Boolean(corrected),
      verification_issues: Array.isArray(verification.issues) ? verification.issues.map(String) : [],
    };
    task.status = "succeeded";
    task._stageValues.clear();
    task._captionedScenes?.clear();
    elements.run.querySelector("span").textContent = "Run live analysis";
    task.status_message = null;
    setBrowserProgress("Analysis complete", "", "6 of 6", false);
    setBrowserStatus("ready", "Live run complete");
    renderPipeline(task);
    renderResults(task);
  } catch (error) {
    if (task.status !== "cancelled") {
      task.status = "failed";
      task.status_message = null;
      task.error = error.message;
      state.browserCheckpoint = { signature, task };
      elements.run.querySelector("span").textContent = "Retry failed stage";
      renderPipeline(task);
      setBrowserProgress("Analysis paused", "Completed steps are saved. Retry to continue.", "", false);
      setBrowserStatus("blocked", "Live run failed");
      const failedStage = pipelineDefinitions.find((definition) =>
        definition[3] === task.steps.find((step) => step.status === "failed")?.name
      )?.[1] || "Model";
      showError(`${failedStage}: ${error.message} Completed steps are saved; use Retry failed stage to continue.`);
    }
  } finally {
    state.pipelineRunning = false;
    state.browserBusy = false;
    elements.run.disabled = false;
    elements.cancel.classList.add("hidden");
    updateBrowserButtons();
  }
}

async function loadLatestTask(videoId) {
  try {
    const tasks = await api(`/v1/tasks?video_id=${encodeURIComponent(videoId)}&limit=1`);
    if (!tasks.length) return;
    state.currentTask = tasks[0];
    elements.question.value = state.currentTask.request?.question || "";
    const options = state.currentTask.request?.options;
    if (options?.length) {
      elements.choices.innerHTML = "";
      enableChoices();
      elements.choices.innerHTML = "";
      options.forEach(addChoice);
    } else { disableChoices(); }
    elements.pipelineCard.classList.remove("hidden");
    renderPipeline(state.currentTask);
    if (state.currentTask.status === "succeeded") renderResults(state.currentTask);
    else if (!terminalTaskStates.has(state.currentTask.status)) state.taskPoll = setInterval(refreshTask, 1500);
  } catch (_) { /* task history is optional */ }
}

async function refreshTask() {
  if (!state.currentTask) return;
  try {
    state.currentTask = await api(`/v1/tasks/${state.currentTask.id}`);
    renderPipeline(state.currentTask);
    if (terminalTaskStates.has(state.currentTask.status)) {
      clearInterval(state.taskPoll);
      elements.cancel.classList.add("hidden");
      elements.run.disabled = state.currentVideo?.status !== "ready" || elements.question.value.trim().length < 2;
      if (state.currentTask.status === "succeeded") renderResults(state.currentTask);
      else showError(state.currentTask.error || `Task ${state.currentTask.status}.`);
    }
  } catch (error) { clearInterval(state.taskPoll); showError(error.message); }
}

function renderPipeline(task) {
  const stepMap = new Map((task.steps || []).map((step) => [step.name, step]));
  const activeIndex = pipelineDefinitions.findIndex(([, , , stepName]) => stepMap.get(stepName)?.status === "running");
  elements.pipeline.innerHTML = pipelineDefinitions.map(([key, label, detail, stepName], index) => {
    const step = stepMap.get(stepName);
    const done = step?.status === "succeeded";
    const working = step?.status === "running"
      || (!step && activeIndex < 0 && task.status === "processing" && index === stepMap.size);
    const latency = step?.latency_ms != null ? `${(step.latency_ms / 1000).toFixed(1)}s` : detail;
    const stage = `<button type="button" class="pipeline-step ${done ? "done" : ""} ${working ? "working" : ""} ${state.activeFlowStage === key ? "selected" : ""}" data-flow-stage="${key}" aria-pressed="${state.activeFlowStage === key}">
      <span class="pipeline-stage-top"><i class="pipeline-node">${index + 1}</i><i class="pipeline-state"></i></span>
      <b>${label}</b><span>${escapeHtml(latency)}</span>
    </button>`;
    return index < pipelineDefinitions.length - 1
      ? `${stage}<span class="flow-connector" aria-hidden="true">→</span>`
      : stage;
  }).join("");
  elements.pipeline.querySelectorAll("[data-flow-stage]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeFlowStage = button.dataset.flowStage;
      renderPipeline(task);
    });
  });
  const messages = {
    queued: "Waiting for an available worker…", processing: "Agents are working over retrieved evidence…",
    succeeded: "Analysis completed with a grounded evidence trace.", failed: "The pipeline stopped with an error.",
    cancel_requested: "Cancellation requested…", cancelled: "Analysis cancelled.",
  };
  elements.pipelineMessage.textContent = task.status_message || messages[task.status] || statusLabel(task.status);
  renderFlowDetail(task, state.activeFlowStage);
}

function renderFlowDetail(task, stageKey) {
  const plan = task.plan || {};
  const reasoning = plan.reasoning_plan || plan;
  const program = plan.tool_program || {};
  const evidence = task.evidence || [];
  const result = task.result || {};
  const needs = reasoning.evidence_needs || [];
  const observations = evidence.map((item) => item.observation).filter(Boolean);
  const stepDetails = new Map((task.steps || []).map((step) => [step.name, step.details || {}]));
  const retrievalDetails = stepDetails.get("retrieve") || {};
  const selectedSceneIndices = retrievalDetails.selected_scene_indices || [];
  const traceText = (retrievalDetails.tool_trace || []).map((entry) =>
    `${entry.call}: ${(entry.hits || []).map((hit) =>
      `scene ${hit.scene}${hit.score == null ? "" : ` (${Number(hit.score).toFixed(3)})`}`
    ).join(", ") || "no hits"}`
  ).join(" → ");
  const details = {
    planner: {
      title: "Reasoning planner",
      input: task.request?.question || "Question unavailable",
      action: reasoning.reasoning || "Waiting for live model output",
      output: needs.map((need) =>
        `${need.description} [${need.temporal_role}${need.depends_on?.length ? `; depends on ${need.depends_on.join(", ")}` : ""}]`
      ).join(" → ") || "Waiting for live model output",
      source: stepDetails.get("plan")?.provenance,
    },
    tool_sequence: {
      title: "Tool-sequence agent",
      input: needs.map((need) => need.description).join(" → ") || "Waiting for planner output",
      action: program.rationale || "Waiting for live model output",
      output: [
        ...(program.locate || []).map((call) => `${call.id}=locate(“${call.query}”)`),
        ...(program.relate || []).map((call) => `${call.id}=relate(${call.source}, ${call.relation})`),
        ...(program.inspect ? [`inspect(${program.inspect.sources.join(", ")})`] : []),
      ].join(" → ") || "Waiting for live model output",
      source: stepDetails.get("tool_sequence")?.provenance,
    },
    retrieve: {
      title: "Vector video cache",
      input: retrievalDetails.vector_index
        ? `${retrievalDetails.vector_index.scene_count} generated scene captions · ${retrievalDetails.vector_index.kind} · ${retrievalDetails.vector_index.dimensions} terms`
        : (program.locate || []).map((call) => call.query).join(" · ") || "Typed retrieval calls",
      action: traceText || "Waiting for live vector scores and temporal traversal",
      output: selectedSceneIndices.length
        ? `Selected scene indices ${selectedSceneIndices.join(", ")} under the model-generated budget`
        : "Waiting for computed retrieval results",
      source: retrievalDetails.provenance,
    },
    inspect: {
      title: "Bounded evidence inspector",
      input: selectedSceneIndices.length ? `Retrieved scene indices ${selectedSceneIndices.join(", ")}` : "Waiting for retrieval",
      action: stepDetails.get("inspect_evidence")?.raw_output || "Waiting for live pixel inspection",
      output: observations.join(" · ") || "No observations retained",
      source: stepDetails.get("inspect_evidence")?.provenance,
    },
    answer: {
      title: "Answer composer",
      input: `${observations.length} grounded observation${observations.length === 1 ? "" : "s"}`,
      action: stepDetails.get("compose_answer")?.raw_output || "Waiting for live model output",
      output: result.answer || "Waiting for an answer",
      source: stepDetails.get("compose_answer")?.provenance,
    },
    verify: {
      title: "Grounding verifier",
      input: result.answer || "Draft answer pending",
      action: stepDetails.get("verify_answer")?.raw_output || "Waiting for live model output",
      output: result.verified
        ? `Verified · ${Math.round((result.confidence || 0) * 100)}% confidence`
        : (result.verification_issues || []).join(" · ") || "Verification pending",
      source: stepDetails.get("verify_answer")?.provenance,
    },
  };
  const detail = details[stageKey] || details.planner;
  const rawArtifacts = {
    planner: stepDetails.get("plan")?.generated_plan,
    tool_sequence: stepDetails.get("tool_sequence")?.generated_program,
    retrieve: retrievalDetails.tool_trace ? {
      vector_index: retrievalDetails.vector_index,
      caption_cache_hit: retrievalDetails.caption_cache_hit,
      tool_trace: retrievalDetails.tool_trace,
      selected_scene_indices: retrievalDetails.selected_scene_indices,
    } : null,
    inspect: stepDetails.get("inspect_evidence")?.generated_evidence,
    answer: stepDetails.get("compose_answer")?.generated_answer,
    verify: stepDetails.get("verify_answer")?.generated_verification,
  };
  const rawArtifact = rawArtifacts[stageKey];
  const stageName = pipelineDefinitions.find(([key]) => key === stageKey)?.[3];
  const responses = stepDetails.get(stageName)?.agent_responses || [];
  elements.pipelineDetail.innerHTML = `
    <div class="flow-detail-heading"><span>Selected stage</span><b>${escapeHtml(detail.title)}</b><em>${escapeHtml(detail.source ? detail.source.replaceAll("_", " ") : "not run")}</em></div>
    <div class="flow-io">
      <div><span>Input</span><p>${escapeHtml(detail.input)}</p></div>
      <i aria-hidden="true">→</i>
      <div><span>Operation</span><p>${escapeHtml(detail.action)}</p></div>
      <i aria-hidden="true">→</i>
      <div><span>Output passed forward</span><p>${escapeHtml(detail.output)}</p></div>
    </div>
    ${rawArtifact ? `<details class="raw-artifact"><summary>Inspect exact stage artifact</summary><pre>${escapeHtml(JSON.stringify(rawArtifact, null, 2))}</pre></details>` : ""}
    ${responses.length ? `<details class="raw-artifact"><summary>Model requests and responses (${responses.length})</summary><pre>${escapeHtml(JSON.stringify(responses, null, 2))}</pre></details>` : ""}`;
}

function renderResults(task) {
  const result = task.result || {};
  elements.results.classList.remove("hidden");
  elements.answer.textContent = result.answer || "No answer was produced.";
  elements.confidence.textContent = result.confidence != null ? `${Math.round(result.confidence * 100)}% confidence` : "Confidence unavailable";
  const cited = result.citations?.length || 0;
  const verificationState = result.corrected_by_verifier
    ? ["verified", "✓ Corrected by verifier"]
    : result.verified ? ["verified", "✓ Verified"] : ["unverified", "△ Verification caveat"];
  elements.answerMeta.innerHTML = `<span class="${verificationState[0]}">${verificationState[1]}</span><span>${cited} citation${cited === 1 ? "" : "s"}</span><span>${task.evidence?.length || 0} inspected moment${task.evidence?.length === 1 ? "" : "s"}</span>`;
  const draftIssues = (result.verification_issues || []).map((issue) => result.corrected_by_verifier ? `Draft issue: ${issue}` : issue);
  const notices = [...(result.limitations || []), ...draftIssues];
  elements.limitations.classList.toggle("hidden", !notices.length);
  elements.limitations.innerHTML = notices.map((item) => `• ${escapeHtml(item)}`).join("<br>");
  renderEvidence(task);
  updateBrowserButtons();
  elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderEvidence(task) {
  const evidence = [...(task.evidence || [])].sort((a, b) => a.start_seconds - b.start_seconds);
  elements.evidenceCount.textContent = `${evidence.length} moment${evidence.length === 1 ? "" : "s"}`;
  elements.timeline.querySelectorAll(".timeline-marker").forEach((item) => item.remove());
  const duration = state.currentVideo?.duration_seconds || 1;
  evidence.forEach((item) => {
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "timeline-marker";
    marker.style.left = `${Math.max(0, item.start_seconds / duration * 100)}%`;
    marker.style.width = `${Math.max(.5, (item.end_seconds - item.start_seconds) / duration * 100)}%`;
    marker.title = `${formatTime(item.start_seconds)}–${formatTime(item.end_seconds)}`;
    marker.addEventListener("click", () => seekTo(item.start_seconds));
    elements.timeline.appendChild(marker);
  });
  if (!evidence.length) {
    elements.evidenceList.innerHTML = '<div class="empty-evidence">No visual evidence was retained for this answer.</div>';
    return;
  }
  elements.evidenceList.innerHTML = evidence.map((item) => {
    const firstFrameUrl = item.frame_urls?.[0]
      || `/v1/tasks/${task.id}/evidence/${item.node_id}/frames/0`;
    const image = item.frame_paths?.length
      ? `<img class="evidence-image" loading="lazy" src="${escapeHtml(firstFrameUrl)}" alt="Evidence frame at ${formatTime(item.start_seconds)}">`
      : '<span class="evidence-placeholder"><svg viewBox="0 0 24 24"><path d="m4 16 4-4 4 4 3-3 5 5"/><rect x="3" y="4" width="18" height="16" rx="2"/></svg></span>';
    return `<article class="evidence-item" data-seconds="${item.start_seconds}" tabindex="0">${image}<div class="evidence-copy"><span class="evidence-time">${formatTime(item.start_seconds)}–${formatTime(item.end_seconds)}<span class="evidence-confidence">${Math.round(item.confidence * 100)}%</span></span><p>${escapeHtml(item.observation)}</p></div></article>`;
  }).join("");
  elements.evidenceList.querySelectorAll(".evidence-item").forEach((item) => {
    const activate = () => seekTo(Number(item.dataset.seconds));
    item.addEventListener("click", activate);
    item.addEventListener("keydown", (event) => { if (event.key === "Enter") activate(); });
  });
}


function seekTo(seconds) {
  elements.player.currentTime = seconds;
  elements.player.play().catch(() => {});
  elements.player.scrollIntoView({ behavior: "smooth", block: "center" });
}

function setBrowserStatus(kind, text) {
  elements.browserStatus.className = `browser-status ${kind}`;
  elements.browserStatus.innerHTML = `<i></i>${escapeHtml(text)}`;
}

function updateBrowserModelDetail() {
  const selected = browserModels[elements.browserModel.value];
  elements.browserModelDetail.textContent = selected?.detail || "";
  if (state.browserModelReady !== elements.browserModel.value && state.webgpuReady && !state.browserBusy) {
    setBrowserStatus("ready", "WebGPU ready · model not loaded");
  }
  updateBrowserButtons();
}

function changeBrowserModel() {
  state.browserCheckpoint = null;
  if (state.browserDemo) elements.run.querySelector("span").textContent = "Run live analysis";
  const selected = browserModels[elements.browserModel.value];
  elements.localModelName.textContent = selected?.name || elements.browserModel.value;
  elements.localAnswer.textContent = state.browserModelReady === elements.browserModel.value
    ? "Model loaded. Inspect the selected evidence when ready."
    : "Load this model to run the comparison.";
  elements.localMetrics.textContent = "";
  elements.localComparison.classList.remove("agrees", "disagrees");
  elements.browserDetails.classList.add("hidden");
  updateBrowserModelDetail();
}

function updateBrowserButtons() {
  const hasEvidence = Boolean(state.currentTask?.evidence?.some((item) => item.frame_paths?.length));
  const busy = state.browserBusy || state.pipelineRunning;
  elements.loadBrowserModel.disabled = !state.webgpuReady || busy;
  elements.runBrowserModel.disabled = !state.webgpuReady || busy
    || state.browserModelReady !== elements.browserModel.value || !hasEvidence;
  elements.browserModel.disabled = busy;
  elements.demoExampleSelect.disabled = busy;
}

function resetBrowserResult() {
  state.browserCheckpoint = null;
  elements.localModelName.textContent = state.browserModelReady
    ? browserModels[state.browserModelReady]?.name || state.browserModelReady
    : "Not loaded";
  elements.localAnswer.textContent = "Load a model, then inspect the retrieved evidence.";
  elements.localMetrics.textContent = "";
  elements.localComparison.classList.remove("agrees", "disagrees");
  elements.browserDetails.classList.add("hidden");
  elements.browserDetails.open = false;
  updateBrowserButtons();
}

async function checkBrowserVlmCapability() {
  updateBrowserModelDetail();
  if (!window.isSecureContext) {
    setBrowserStatus("blocked", "HTTPS or localhost required");
    elements.browserModelDetail.textContent += " WebGPU is unavailable from an insecure HTTP origin.";
    return;
  }
  if (!navigator.gpu) {
    setBrowserStatus("blocked", "WebGPU unavailable");
    elements.browserModelDetail.textContent += " Try a current Chromium browser with WebGPU enabled.";
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No compatible GPU adapter was found.");
    state.webgpuReady = true;
    const description = adapter.info?.description || adapter.info?.device || adapter.info?.vendor;
    setBrowserStatus("ready", description ? `WebGPU · ${description}` : "WebGPU ready");
  } catch (error) {
    setBrowserStatus("blocked", "WebGPU adapter unavailable");
    elements.browserModelDetail.textContent += ` ${error.message}`;
  }
  updateBrowserButtons();
}

function ensureBrowserWorker() {
  if (state.browserWorker) return state.browserWorker;
  const worker = new Worker("/static/browser-vlm-worker.js?v=20260905-21", { type: "module" });
  worker.addEventListener("message", handleBrowserWorkerMessage);
  worker.addEventListener("error", (event) => {
    for (const pending of state.browserRequests.values()) pending.reject(new Error(event.message || "Worker failed"));
    state.browserRequests.clear();
    state.browserLoadReject?.(new Error(event.message || "Worker failed"));
    state.browserLoadPromise = null;
    state.browserLoadResolve = null;
    state.browserLoadReject = null;
    worker.terminate();
    state.browserWorker = null;
    state.browserModelReady = null;
    finishBrowserOperation();
    setBrowserProgress("Model stopped", "The error details are shown below.", "", false);
    setBrowserStatus("blocked", "Worker failed");
    showError(`Browser VLM worker failed: ${event.message || "unknown error"}`);
  });
  state.browserWorker = worker;
  return worker;
}

function setBrowserProgress(label, detail = "", value = "", active = true) {
  elements.browserProgress.classList.remove("hidden");
  elements.browserProgress.classList.toggle("indeterminate", active);
  elements.browserProgress.setAttribute("aria-busy", String(active));
  elements.browserProgressLabel.textContent = label;
  elements.browserProgressDetail.textContent = detail;
  elements.browserProgressValue.textContent = value;
  elements.browserProgressBar.style.width = active ? "35%" : label === "Analysis complete" ? "100%" : "0%";
}

function startBrowserOperation(message) {
  state.browserBusy = true;
  setBrowserProgress(message);
  setBrowserStatus("busy", message);
  updateBrowserButtons();
}

function finishBrowserOperation() {
  state.browserBusy = false;
  updateBrowserButtons();
}

function handleBrowserWorkerMessage(event) {
  const message = event.data || {};
  if (message.type === "progress") {
    const bytes = Number(message.downloadedBytes) || 0;
    const size = bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB`
      : `${Math.floor(bytes / 1024 ** 2)} MB`;
    const label = message.scope === "embedding" ? "Preparing search" : "Loading model";
    const detail = bytes >= 1024 ** 2 ? `${size} downloaded`
      : "First load can take a few minutes. Later runs use the browser cache.";
    setBrowserProgress(label, detail, message.scope === "embedding" ? "Step 3 of 6" : "");
    return;
  }
  if (message.type === "ready") {
    state.browserModelReady = message.modelKey;
    if (message.loadMs) state.browserLoadMs = message.loadMs;
    finishBrowserOperation();
    setBrowserProgress("Model ready", "", "", false);
    elements.localModelName.textContent = browserModels[message.modelKey]?.name || message.modelId;
    setBrowserStatus("ready", "Model ready");
    state.browserLoadResolve?.(message);
    state.browserLoadPromise = null;
    state.browserLoadResolve = null;
    state.browserLoadReject = null;
    return;
  }
  if (message.type === "embeddings" && message.requestId && state.browserRequests.has(message.requestId)) {
    const pending = state.browserRequests.get(message.requestId);
    state.browserRequests.delete(message.requestId);
    pending.resolve(message);
    return;
  }
  if (message.type === "result") {
    if (message.requestId && state.browserRequests.has(message.requestId)) {
      const pending = state.browserRequests.get(message.requestId);
      state.browserRequests.delete(message.requestId);
      pending.resolve(message);
      return;
    }
    finishBrowserOperation();
    elements.browserProgress.classList.add("hidden");
    setBrowserStatus("ready", "Local inference complete");
    renderBrowserResult(message);
    return;
  }
  if (message.type === "error") {
    if (message.requestId && state.browserRequests.has(message.requestId)) {
      const pending = state.browserRequests.get(message.requestId);
      state.browserRequests.delete(message.requestId);
      pending.reject(new Error(message.message));
      return;
    }
    state.browserLoadReject?.(new Error(message.message));
    state.browserLoadPromise = null;
    state.browserLoadResolve = null;
    state.browserLoadReject = null;
    finishBrowserOperation();
    setBrowserProgress("Model stopped", "The error details are shown below.", "", false);
    setBrowserStatus("blocked", "Local inference failed");
    showError(`Browser VLM ${message.operation || "operation"} failed: ${message.message}`);
  }
}

function loadSelectedBrowserModel() {
  if (!state.webgpuReady || state.browserBusy) return;
  ensureBrowserModelLoaded().catch((error) => showError(error.message));
}

function ensureBrowserModelLoaded() {
  if (!state.webgpuReady) return Promise.reject(new Error("WebGPU requires localhost or HTTPS and a compatible browser."));
  if (state.browserModelReady === elements.browserModel.value) return Promise.resolve();
  if (state.browserLoadPromise) return state.browserLoadPromise;
  const selectedKey = elements.browserModel.value;
  const selected = browserModels[selectedKey];
  if (selected?.highMemory && !state.confirmedLargeModels.has(selectedKey)) {
    const accepted = window.confirm(
      selected.loadWarning || `${selected.name} is a high-memory model and may make the browser or computer unresponsive. `
      + "LFM2.5-VL-450M uses less memory. Load the large model anyway?",
    );
    if (!accepted) {
      elements.browserModel.value = "lfm-450m";
      updateBrowserModelDetail();
      return Promise.reject(new Error("Model load cancelled. The lower-memory 450M model is now selected."));
    }
    state.confirmedLargeModels.add(selectedKey);
  }
  startBrowserOperation("Loading model");
  state.browserLoadPromise = new Promise((resolve, reject) => {
    state.browserLoadResolve = resolve;
    state.browserLoadReject = reject;
  });
  ensureBrowserWorker().postMessage({ type: "load", modelKey: elements.browserModel.value });
  return state.browserLoadPromise;
}

function neutralAgentImage() {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const context = canvas.getContext("2d", { alpha: false });
  context.fillStyle = "#f5f6f8";
  context.fillRect(0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

function inferBrowserAgent(prompt, imageUrl = null, maxNewTokens = 500) {
  const requestId = crypto.randomUUID();
  const worker = ensureBrowserWorker();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      state.browserRequests.delete(requestId);
      reject(new Error("Browser agent inference timed out."));
    }, 10 * 60 * 1000);
    state.browserRequests.set(requestId, {
      resolve: (message) => { clearTimeout(timeout); resolve(message); },
      reject: (error) => { clearTimeout(timeout); reject(error); },
    });
    worker.postMessage({
      type: "infer",
      requestId,
      modelKey: elements.browserModel.value,
      imageUrl: imageUrl || neutralAgentImage(),
      prompt,
      maxNewTokens,
    });
  });
}

function embedBrowserTexts(texts) {
  const requestId = crypto.randomUUID();
  const worker = ensureBrowserWorker();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      state.browserRequests.delete(requestId);
      reject(new Error("Browser embedding inference timed out."));
    }, 10 * 60 * 1000);
    state.browserRequests.set(requestId, {
      resolve: (message) => { clearTimeout(timeout); resolve(message); },
      reject: (error) => { clearTimeout(timeout); reject(error); },
    });
    worker.postMessage({ type: "embed", requestId, texts });
  });
}

async function inferJsonAgent(prompt, imageUrl = null, maxNewTokens = 500, validator = null) {
  const step = state.currentTask?.steps?.find((item) => item.status === "running");
  const attempts = [];
  let lastError = null;
  let previousOutput = "";
  let inferenceMs = 0;
  const concise = "Return one complete JSON object only. Use all required keys. Keep each text value under 20 words. Escape any quotes inside strings. No markdown or commentary.";
  for (let attempt = 0; attempt < 3; attempt += 1) {
    if (state.currentTask?.status === "cancelled") throw new Error("Run cancelled by user.");
    let request = `${prompt}\n\n${concise}`;
    if (attempt === 1) {
      request += `\n\nYour previous response failed validation: ${lastError.message}\nRewrite it completely using the required schema. Preserve grounded facts and do not invent missing evidence.\nPrevious response:\n${previousOutput}`;
    } else if (attempt === 2) {
      request += `\n\nGenerate a fresh, shorter response. The previous attempt failed: ${lastError.message}. Use short phrases and empty optional arrays when appropriate.`;
    }
    if (attempt) elements.browserProgressDetail.textContent = `Reformatting model response · attempt ${attempt + 1} of 3`;
    const response = await inferBrowserAgent(request, imageUrl, maxNewTokens);
    inferenceMs += response.inferenceMs || 0;
    previousOutput = response.text;
    const record = {
      prompt: request,
      raw_output: response.text,
      model_inference_ms: response.inferenceMs,
      generated_tokens: response.generatedTokens ?? null,
      hit_token_limit: response.hitTokenLimit ?? null,
      syntax_repaired: false,
    };
    attempts.push(record);
    if (step) (step.details.agent_responses ||= []).push(record);
    try {
      const parsed = window.VideoMASysLive.parseModelJson(response.text, response.hitTokenLimit);
      record.syntax_repaired = parsed.repaired;
      if (parsed.repaired) record.repaired_json = parsed.normalized;
      const data = validator ? validator(parsed.data) : parsed.data;
      record.validation = "passed";
      return { data, inferenceMs, raw: response.text, repaired: attempt > 0 || parsed.repaired, attempts };
    } catch (error) {
      lastError = error;
      record.validation = "failed";
      record.error = error.message;
    }
  }
  throw new Error("The model could not produce a valid response after 3 attempts. Open this stage’s model responses for details.");
}

function selectBrowserFrames(task, limit = 6) {
  const evidence = [...(task.evidence || [])]
    .filter((item) => item.frame_paths?.length)
    .sort((a, b) => a.start_seconds - b.start_seconds);
  const selected = [];
  for (const fraction of [0.5, 0.75, 0.25]) {
    for (const item of evidence) {
      if (selected.length >= limit) return selected;
      const count = item.frame_paths.length;
      const index = Math.min(count - 1, Math.max(0, Math.round((count - 1) * fraction)));
      if (selected.some((frame) => frame.item.node_id === item.node_id && frame.index === index)) continue;
      const timestamp = item.start_seconds + ((index + 1) / (count + 1))
        * (item.end_seconds - item.start_seconds);
      selected.push({ item, index, timestamp });
    }
  }
  return selected;
}

function loadBrowserImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load evidence frame: ${url}`));
    image.src = url;
  });
}

async function buildEvidenceContactSheet(task, frames) {
  const columns = frames.length === 1 ? 1 : 2;
  const cellWidth = 320;
  const cellHeight = 210;
  const rows = Math.ceil(frames.length / columns);
  const canvas = document.createElement("canvas");
  canvas.width = cellWidth * columns;
  canvas.height = cellHeight * rows;
  const context = canvas.getContext("2d", { alpha: false });
  context.fillStyle = "#101828";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const labels = [];
  await Promise.all(frames.map(async (frame, frameIndex) => {
    const url = frame.item.frame_urls?.[frame.index]
      || `/v1/tasks/${task.id}/evidence/${frame.item.node_id}/frames/${frame.index}`;
    const image = await loadBrowserImage(url);
    const column = frameIndex % columns;
    const row = Math.floor(frameIndex / columns);
    const x = column * cellWidth;
    const y = row * cellHeight;
    const labelHeight = 26;
    const scale = Math.min(cellWidth / image.naturalWidth, (cellHeight - labelHeight) / image.naturalHeight);
    const drawWidth = image.naturalWidth * scale;
    const drawHeight = image.naturalHeight * scale;
    context.drawImage(
      image,
      x + (cellWidth - drawWidth) / 2,
      y + (cellHeight - labelHeight - drawHeight) / 2,
      drawWidth,
      drawHeight,
    );
    const label = `T${frameIndex + 1}  ${formatTime(frame.timestamp)}`;
    labels[frameIndex] = label;
    context.fillStyle = "rgba(16,24,40,.94)";
    context.fillRect(x, y + cellHeight - labelHeight, cellWidth, labelHeight);
    context.fillStyle = "#ffffff";
    context.font = "600 13px sans-serif";
    context.fillText(label, x + 10, y + cellHeight - 8);
  }));
  return { imageUrl: canvas.toDataURL("image/jpeg", 0.88), labels };
}

function seekLocalVideo(video, seconds) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Timed out while sampling the local video.")), 15000);
    video.onseeked = () => { clearTimeout(timeout); resolve(); };
    video.onerror = () => { clearTimeout(timeout); reject(new Error("Could not seek the local video.")); };
    video.currentTime = Math.max(0, Math.min(seconds, video.duration - 0.05));
  });
}

async function extractLocalScenes() {
  if (state.localScenes) return state.localScenes;
  const source = document.createElement("video");
  source.preload = "auto";
  source.muted = true;
  source.src = state.localVideoUrl;
  await new Promise((resolve, reject) => {
    source.onloadeddata = resolve;
    source.onerror = () => reject(new Error("The browser could not decode frames from this video."));
  });
  const duration = state.currentVideo.duration_seconds;
  const sceneSeconds = Math.max(20, duration / 24);
  const scenes = [];
  for (let index = 0, start = 0; start < duration && index < 24; index += 1, start += sceneSeconds) {
    const end = Math.min(duration, start + sceneSeconds);
    const frameUrls = [];
    for (const fraction of [0.25, 0.5, 0.75]) {
      const timestamp = start + (end - start) * fraction;
      await seekLocalVideo(source, timestamp);
      const scale = Math.min(1, 640 / source.videoWidth);
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(source.videoWidth * scale));
      canvas.height = Math.max(1, Math.round(source.videoHeight * scale));
      canvas.getContext("2d", { alpha: false }).drawImage(source, 0, 0, canvas.width, canvas.height);
      frameUrls.push(canvas.toDataURL("image/jpeg", 0.84));
    }
    scenes.push({ index, start_seconds: start, end_seconds: end, frame_urls: frameUrls });
  }
  state.localScenes = scenes;
  return scenes;
}

async function getBrowserScenes() {
  if (state.currentVideo?.local) return extractLocalScenes();
  const payload = await api(`/v1/browser-demo/assets/${encodeURIComponent(state.currentDemoVideoKey)}/scenes`);
  return payload.scenes;
}

function framesForScenes(scenes, framesPerScene = 2) {
  const frames = [];
  for (const scene of scenes) {
    const urls = scene.frame_urls || [];
    const selectedIndices = urls.length <= framesPerScene
      ? urls.map((_, index) => index)
      : Array.from({ length: framesPerScene }, (_, index) =>
        Math.round(index * (urls.length - 1) / Math.max(framesPerScene - 1, 1))
      );
    for (const index of selectedIndices) {
      frames.push({
        item: {
          node_id: `scene-${scene.index}`,
          start_seconds: scene.start_seconds,
          end_seconds: scene.end_seconds,
          frame_paths: urls,
          frame_urls: urls,
        },
        index,
        sceneIndex: scene.index,
        timestamp: scene.start_seconds + ((index + 1) / (urls.length + 1))
          * (scene.end_seconds - scene.start_seconds),
      });
    }
  }
  return frames;
}

async function captionScenes(rawScenes) {
  const cacheKey = `videomasys.live-captions.v1.${state.currentVideo.id}.${elements.browserModel.value}`;
  try {
    const cached = JSON.parse(localStorage.getItem(cacheKey));
    if (Array.isArray(cached) && cached.length === rawScenes.length && cached.every((item) => item.caption)) {
      return { scenes: rawScenes.map((scene, index) => ({ ...scene, ...cached[index] })), cacheHit: true, modelMs: 0 };
    }
  } catch (_) { /* regenerate a corrupt or unavailable cache */ }

  if (!state.currentTask._captionedScenes) {
    Object.defineProperty(state.currentTask, "_captionedScenes", { value: new Map() });
  }
  const analyses = state.currentTask._captionedScenes;
  let modelMs = 0;
  for (let offset = 0; offset < rawScenes.length; offset += 3) {
    const batch = rawScenes.slice(offset, offset + 3);
    if (batch.every((scene) => analyses.has(scene.index))) continue;
    const frames = framesForScenes(batch, 2);
    const sheet = await buildEvidenceContactSheet(state.currentTask, frames);
    const sceneMap = batch.map((scene) => ({
      index: scene.index,
      start: scene.start_seconds,
      end: scene.end_seconds,
      labels: frames.map((frame, index) => frame.sceneIndex === scene.index ? sheet.labels[index].split("  ")[0] : null).filter(Boolean),
    }));
    setBrowserProgress("Reading the video", `Scenes ${offset + 1}–${offset + batch.length} of ${rawScenes.length}`, "Step 3 of 6");
    const response = await inferJsonAgent(
      window.VideoMASysLive.sceneCaptionPrompt(sceneMap, sheet.labels),
      sheet.imageUrl,
      650,
      (raw) => {
        if (!Array.isArray(raw.scenes)) throw new Error("Scene analyst returned no scene list.");
        const returned = new Map(raw.scenes.map((item) => [Number(item.index), item]));
        for (const scene of batch) {
          if (!String(returned.get(scene.index)?.caption || "").trim()) {
            throw new Error(`Scene analyst omitted scene ${scene.index}.`);
          }
        }
        return raw;
      },
    );
    modelMs += response.inferenceMs;
    if (!Array.isArray(response.data.scenes)) throw new Error("Scene analyst returned no scene list.");
    for (const analysis of response.data.scenes) {
      const index = Number(analysis.index);
      if (!batch.some((scene) => scene.index === index) || !String(analysis.caption || "").trim()) {
        throw new Error("Scene analyst returned an invalid scene index or empty caption.");
      }
      analyses.set(index, {
        caption: String(analysis.caption).trim(),
        entities: Array.isArray(analysis.entities) ? analysis.entities.map(String) : [],
        actions: Array.isArray(analysis.actions) ? analysis.actions.map(String) : [],
      });
    }
  }
  if (analyses.size !== rawScenes.length) throw new Error("Scene analyst did not caption every scene.");
  const scenes = rawScenes.map((scene) => ({ ...scene, ...analyses.get(scene.index) }));
  try {
    localStorage.setItem(cacheKey, JSON.stringify(scenes.map(({ caption, entities, actions }) => ({ caption, entities, actions }))));
  } catch (_) { /* caching is an optimization, not a correctness requirement */ }
  return { scenes, cacheHit: false, modelMs };
}

function browserPrompt(task, labels) {
  const options = task.request?.options || [];
  const optionText = options.length
    ? `\nAnswer choices (zero-based):\n${options.map((option, index) => `${index}: ${option}`).join("\n")}`
    : "";
  return `You are the visual evidence inspector in a retrieval-first video QA system.
The contact sheet contains chronological frames selected from a much longer video.
Panels are ordered left-to-right, top-to-bottom and labelled ${labels.join(", ")}.
Use only visible evidence. Compare actions and state changes across panels. Do not invent events outside them.

Question: ${task.request?.question || ""}${optionText}

Return exactly one compact JSON object with this schema:
{"answer":"direct answer","selected_option":null,"observation":"chronological visual evidence","confidence":0.0}
For multiple choice, selected_option must be the zero-based integer. For open questions, use null.`;
}

async function runSelectedBrowserModel() {
  if (!state.currentTask || state.browserModelReady !== elements.browserModel.value || state.browserBusy) return;
  try {
    const frames = selectBrowserFrames(state.currentTask);
    if (!frames.length) throw new Error("This task has no cached evidence frames to inspect.");
    startBrowserOperation("Preparing the evidence");
    const sheet = await buildEvidenceContactSheet(state.currentTask, frames);
    elements.browserContactSheet.src = sheet.imageUrl;
    elements.browserContactSheet.dataset.frameCount = String(frames.length);
    ensureBrowserWorker().postMessage({
      type: "infer",
      modelKey: elements.browserModel.value,
      imageUrl: sheet.imageUrl,
      prompt: browserPrompt(state.currentTask, sheet.labels),
    });
    setBrowserProgress("Looking at the evidence", `${frames.length} frames`);
  } catch (error) {
    finishBrowserOperation();
    setBrowserStatus("blocked", "Evidence preparation failed");
    showError(error.message);
  }
}

function parseBrowserOutput(text) {
  const withoutThinking = String(text || "").replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const fenced = withoutThinking.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1];
  const candidate = fenced || withoutThinking.match(/\{[\s\S]*\}/)?.[0];
  if (!candidate) return { answer: withoutThinking, observation: withoutThinking, confidence: null, selected_option: null };
  try {
    const parsed = JSON.parse(candidate);
    return {
      answer: String(parsed.answer || withoutThinking),
      observation: String(parsed.observation || parsed.answer || withoutThinking),
      confidence: Number.isFinite(Number(parsed.confidence)) ? Math.max(0, Math.min(1, Number(parsed.confidence))) : null,
      selected_option: Number.isInteger(parsed.selected_option) ? parsed.selected_option : null,
    };
  } catch (_) {
    return { answer: withoutThinking, observation: withoutThinking, confidence: null, selected_option: null };
  }
}

function renderBrowserResult(message) {
  const parsed = parseBrowserOutput(message.text);
  const model = browserModels[message.modelKey];
  const baselineOption = state.currentTask?.result?.selected_option;
  const comparable = Number.isInteger(baselineOption) && Number.isInteger(parsed.selected_option);
  const agrees = comparable && baselineOption === parsed.selected_option;
  elements.localModelName.textContent = model?.name || message.modelId;
  elements.localAnswer.textContent = parsed.answer || "The browser model returned an empty answer.";
  elements.localObservation.textContent = parsed.observation || "No observation returned.";
  elements.localComparison.classList.toggle("agrees", agrees);
  elements.localComparison.classList.toggle("disagrees", comparable && !agrees);
  const confidence = parsed.confidence == null ? "confidence unavailable" : `${Math.round(parsed.confidence * 100)}% confidence`;
  const agreement = comparable ? (agrees ? "agrees with the live pipeline" : "disagrees with the live pipeline") : "open-answer comparison";
  const frameCount = elements.browserContactSheet.dataset.frameCount || "0";
  const loadMetric = state.browserLoadMs ? ` · load ${(state.browserLoadMs / 1000).toFixed(1)}s` : "";
  elements.localMetrics.textContent = `${(message.inferenceMs / 1000).toFixed(1)}s inference${loadMetric} · ${frameCount} frames · ${confidence} · ${agreement}`;
  elements.browserDetails.classList.remove("hidden");
}

async function cancelTask() {
  if (!state.currentTask) return;
  if (state.browserDemo && state.pipelineRunning) {
    state.browserWorker?.terminate();
    state.browserWorker = null;
    for (const pending of state.browserRequests.values()) pending.reject(new Error("Run cancelled by user."));
    state.browserRequests.clear();
    state.pipelineRunning = false;
    state.browserBusy = false;
    state.browserModelReady = null;
    state.browserLoadReject?.(new Error("Run cancelled by user."));
    state.browserLoadPromise = null;
    state.browserLoadResolve = null;
    state.browserLoadReject = null;
    state.currentTask.status = "cancelled";
    renderPipeline(state.currentTask);
    elements.cancel.classList.add("hidden");
    elements.run.disabled = false;
    setBrowserProgress("Analysis cancelled", "", "", false);
    setBrowserStatus("ready", "Run cancelled · reload model to retry");
    return;
  }
  try {
    state.currentTask = await api(`/v1/tasks/${state.currentTask.id}/cancel`, { method: "POST" });
    renderPipeline(state.currentTask);
    toast("Cancellation requested");
  } catch (error) { showError(error.message); }
}

function showError(message) {
  elements.errorText.textContent = message;
  elements.error.classList.remove("hidden");
}

let toastTimeout;
function toast(message) {
  clearTimeout(toastTimeout);
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  toastTimeout = setTimeout(() => elements.toast.classList.add("hidden"), 2600);
}

elements.input.addEventListener("change", () => uploadVideo(elements.input.files[0]));
["dragenter", "dragover"].forEach((eventName) => elements.dropzone.addEventListener(eventName, (event) => { event.preventDefault(); elements.dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((eventName) => elements.dropzone.addEventListener(eventName, (event) => { event.preventDefault(); elements.dropzone.classList.remove("dragging"); }));
elements.dropzone.addEventListener("drop", (event) => uploadVideo(event.dataTransfer.files[0]));
elements.player.addEventListener("loadedmetadata", () => elements.videoEmpty.classList.add("hidden"));
elements.player.addEventListener("error", () => {
  elements.videoEmpty.querySelector("span").textContent = "Preview unavailable";
  elements.videoEmpty.classList.remove("hidden");
});
elements.question.addEventListener("input", () => { elements.run.disabled = state.currentVideo?.status !== "ready" || elements.question.value.trim().length < 2; });
elements.toggleChoices.addEventListener("click", () => state.choicesEnabled ? disableChoices() : enableChoices());
elements.addChoice.addEventListener("click", () => addChoice());
elements.run.addEventListener("click", createTask);
elements.demoExampleSelect.addEventListener("change", () => loadBrowserDemo(elements.demoExampleSelect.value));
elements.cancel.addEventListener("click", cancelTask);
elements.browserModel.addEventListener("change", changeBrowserModel);
elements.loadBrowserModel.addEventListener("click", loadSelectedBrowserModel);
elements.runBrowserModel.addEventListener("click", runSelectedBrowserModel);
$("#newSessionButton").addEventListener("click", resetWorkspace);
function refreshLibrary() {
  if (state.libraryRequest) return state.libraryRequest;
  state.libraryLoading = true;
  state.libraryError = null;
  renderVideoList();
  state.libraryRequest = (async () => {
    try {
      if (state.browserDemo) {
        const catalog = await api("/v1/browser-demo/examples");
        state.demoExamples = catalog.examples;
        state.demoVideos = catalog.videos;
      } else {
        state.videos = await api("/v1/videos?limit=30");
      }
    } catch (error) {
      state.libraryError = error.message;
    } finally {
      state.libraryLoading = false;
      state.libraryRequest = null;
      renderVideoList();
    }
  })();
  return state.libraryRequest;
}

function openLibrary() {
  if (state.pipelineRunning || state.browserBusy) { toast("Stop the current run before changing videos."); return; }
  elements.librarySearch.value = "";
  renderVideoList();
  elements.library.showModal();
  elements.librarySearch.focus();
}

$("#refreshVideosButton").addEventListener("click", refreshLibrary);
$("#libraryFocusButton").addEventListener("click", openLibrary);
$("#libraryButton").addEventListener("click", openLibrary);
$("#closeLibraryButton").addEventListener("click", () => elements.library.close());
elements.librarySearch.addEventListener("input", renderVideoList);
elements.librarySearch.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    elements.videoList.querySelector("button")?.click();
  }
});
elements.library.addEventListener("click", (event) => {
  const bounds = elements.library.getBoundingClientRect();
  if (event.target === elements.library && (event.clientX < bounds.left || event.clientX > bounds.right
      || event.clientY < bounds.top || event.clientY > bounds.bottom)) elements.library.close();
});
$("#libraryUploadButton").addEventListener("click", () => {
  elements.library.close();
  elements.input.value = "";
  elements.input.click();
});
$("#dismissError").addEventListener("click", () => elements.error.classList.add("hidden"));

const launchParams = new URLSearchParams(window.location.search);
const browserDemoRequested = launchParams.get("mode") !== "server" && launchParams.get("browser_demo") !== "0";
// Keep root navigation and bookmarks in the browser prototype. Server indexing
// requires an explicit mode because it depends on PostgreSQL and hosted workers.
if (browserDemoRequested) loadBrowserDemoCatalog();
else {
  checkHealth();
  loadVideos();
}
checkBrowserVlmCapability();
if (!browserDemoRequested) setInterval(checkHealth, 15000);
