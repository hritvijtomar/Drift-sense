/* Drift-Sense frontend
   ---------------------
   No build step, no framework -- talks to the Flask API in app.py via
   fetch() and Server-Sent Events (EventSource). Every number rendered on
   screen comes from a backend event payload; nothing here is fabricated.
*/

const STAGES = [
  { key: "load", label: "LOAD" },
  { key: "template_extraction", label: "TEMPLATE" },
  { key: "candidate_generation", label: "GENERATE" },
  { key: "verification", label: "VERIFY" },
  { key: "ranking", label: "RANK" },
  { key: "decision", label: "DECIDE" },
  { key: "result", label: "RESULT" },
];

const state = {
  mode: "example",
  examples: [],
  currentExample: null,
  config: { reference_scale_range: [0.07, 0.15], recommended_reference_upscale_factor: 9.1, ambiguous_gap_threshold: 0.08, gt_match_tolerance_px: 15 },
  upload: { refToken: null, searchToken: null, selfGT: null },
  refImg: null,
  searchImg: null,
  candidates: [],
  ranked: [],
  selected: null,
  groundTruth: null,
  gtSource: null,
  diagnosisTrueRank: null,
  view: { scale: 1, offsetX: 0, offsetY: 0, baseScale: 1, dragging: false, lastX: 0, lastY: 0 },
  es: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ============================================================ config */
async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    state.config = await res.json();
  } catch (e) { /* keep defaults */ }
}

/* ============================================================ tracker */
function initTracker() {
  const el = $("#tracker");
  el.innerHTML = "";
  STAGES.forEach((s) => {
    const step = document.createElement("div");
    step.className = "tracker-step";
    step.id = `step-${s.key}`;
    step.innerHTML = `<div class="row1"><span class="dot"></span><span class="label">${s.label}</span></div><div class="sub" id="sub-${s.key}"></div>`;
    el.appendChild(step);
  });
}

function setStage(stageKey, status, subText) {
  const step = $(`#step-${stageKey}`);
  if (!step) return;
  step.classList.remove("done", "running", "failed");
  if (status === "complete") step.classList.add("done");
  else if (status === "running") step.classList.add("running");
  else if (status === "failed") step.classList.add("failed");
  if (subText !== undefined) {
    const sub = $(`#sub-${stageKey}`);
    if (sub) sub.textContent = subText;
  }
}

function trackerSubText(stage, status, data) {
  switch (stage) {
    case "load":
      if (status === "complete") return `${data.search_shape.w}×${data.search_shape.h}px`;
      return "reading files\u2026";
    case "template_extraction":
      return data.template_w ? `${data.template_w}×${data.template_h}px` : "";
    case "candidate_generation":
      if (status === "complete") return `${data.candidate_count} candidates`;
      return "NCC sweep\u2026";
    case "verification":
      if (data && data.candidate_index !== undefined) return `${data.candidate_index + 1} / ${state.candidates.length} verified`;
      if (status === "complete") return "done";
      return "ORB + RANSAC\u2026";
    case "ranking":
      if (data && data.ranking && data.ranking.length) return `top score ${data.ranking[0].final_score.toFixed(3)}`;
      return "";
    case "decision":
      if (data && data.selected) return `(${data.selected.x.toFixed(0)}, ${data.selected.y.toFixed(0)})`;
      return "";
    case "result":
      if (data && data.result) return `${data.result.runtime_sec.toFixed(2)}s`;
      return "";
    default: return "";
  }
}

/* ============================================================ feed / explanations */
function explain(stage, status, data) {
  switch (stage) {
    case "load":
      if (status === "complete") return "Both images decoded successfully and are ready for matching.";
      return null;
    case "template_extraction":
      return "The entire reference image is used as the match template -- Drift-Sense does not internally crop it.";
    case "candidate_generation":
      if (status === "complete") {
        return `The multi-scale / multi-rotation NCC search produced ${data.candidate_count} candidate location(s) before verification and ranking. This is a normalized cross-correlation score, not a probability.`;
      }
      return `Sweeping scales ${state.config.reference_scale_range[0]}\u2013${state.config.reference_scale_range[1]}\u00d7 and rotations \u00b14\u00b0 of the reference across the full search image.`;
    case "verification":
      if (data && data.candidate) {
        const c = data.candidate;
        return `${c.orb_inliers} of ${c.orb_total_matches} ORB feature matches remained geometrically consistent under RANSAC affine estimation -- this is what separates a true match from a periodic-pattern alias.`;
      }
      return "Cropping around each NCC candidate and matching ORB keypoints against the template, then estimating an affine transform with RANSAC.";
    case "ranking":
      return "Candidates ranked by a 50/50 combination of NCC score and ORB-inlier fraction (capped at 15 inliers).";
    case "decision":
      return "Highest combined score wins; if multiple candidates are within 0.005 of the top score, the one closest to the search-image center is selected (official tie-break rule).";
    case "result":
      return null;
    default:
      return null;
  }
}

function appendFeed(stage, status, message, data) {
  const feed = $("#feed");
  const ph = feed.querySelector(".placeholder");
  if (ph) ph.remove();
  const item = document.createElement("div");
  item.className = "feed-item" + (status === "failed" ? " failed" : "");
  const ex = explain(stage, status, data);
  item.innerHTML = `<div class="fi-stage">${stage.replace(/_/g, " ")}</div>
    <div class="fi-msg">${message}</div>
    ${ex ? `<div class="fi-explain">${ex}</div>` : ""}`;
  feed.appendChild(item);
  feed.scrollTop = feed.scrollHeight;
}

/* ============================================================ canvases */
function setupCanvas(canvas, wrap) {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
}

function drawReference() {
  const canvas = $("#refCanvas");
  setupCanvas(canvas, $("#refWrap"));
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.refImg) return;
  const img = state.refImg;
  const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
  const ox = (canvas.width - w) / 2, oy = (canvas.height - h) / 2;
  ctx.drawImage(img, ox, oy, w, h);
  $("#refDims").textContent = `${img.naturalWidth}\u00d7${img.naturalHeight}px`;
}

function computeBaseScale(canvas, img) {
  if (!img) return 1;
  return Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
}

const MARKER_COLORS = {
  candidate: "#8e8e93",
  verified: "#ffd60a",
  selected: "#0a84ff",
  gt: "#30d158",
  error: "#ff9f0a",
};

function drawSearch() {
  const canvas = $("#searchCanvas");
  setupCanvas(canvas, $("#searchWrap"));
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.searchImg) return;

  const img = state.searchImg;
  state.view.baseScale = computeBaseScale(canvas, img);
  const total = state.view.baseScale * state.view.scale;

  ctx.save();
  ctx.translate(state.view.offsetX, state.view.offsetY);
  ctx.scale(total, total);
  ctx.drawImage(img, 0, 0);

  const inv = 1 / total; // keeps stroke widths / text visually constant regardless of zoom

  function haloRect(x, y, w, h, color, lw, dashed) {
    ctx.save();
    if (dashed) ctx.setLineDash([5 * inv, 4 * inv]);
    ctx.strokeStyle = "rgba(0,0,0,0.65)";
    ctx.lineWidth = (lw + 2) * inv;
    ctx.strokeRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = lw * inv;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }

  function labelChip(x, y, text, color) {
    ctx.save();
    ctx.font = `600 ${11 * inv}px -apple-system, sans-serif`;
    const pad = 3 * inv;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.fillRect(x - pad, y - 11 * inv - pad, tw + pad * 2, 13 * inv + pad);
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  // raw NCC candidates (before verification data attaches)
  if (state.ranked.length === 0 && state.candidates.length) {
    state.candidates.forEach((c) => {
      haloRect(c.tl_x, c.tl_y, c.w, c.h, MARKER_COLORS.candidate, 1.4, true);
    });
  }

  // ranked / verified candidates
  state.ranked.forEach((c) => {
    const isSelected = state.selected && c.x === state.selected.x && c.y === state.selected.y;
    const color = isSelected ? MARKER_COLORS.selected : MARKER_COLORS.verified;
    haloRect(c.tl_x, c.tl_y, c.w, c.h, color, isSelected ? 3 : 1.8, false);
    labelChip(c.tl_x + 4 * inv, c.tl_y + 15 * inv, `#${c.rank}`, color);
  });

  // ground truth marker
  if (state.groundTruth) {
    const g = state.groundTruth;
    const r = 13 * inv;
    ctx.save();
    ctx.strokeStyle = "rgba(0,0,0,0.65)";
    ctx.lineWidth = 4.2 * inv;
    ctx.beginPath();
    ctx.moveTo(g.x - r, g.y); ctx.lineTo(g.x + r, g.y);
    ctx.moveTo(g.x, g.y - r); ctx.lineTo(g.x, g.y + r);
    ctx.stroke();
    ctx.strokeStyle = MARKER_COLORS.gt;
    ctx.lineWidth = 2.4 * inv;
    ctx.beginPath();
    ctx.moveTo(g.x - r, g.y); ctx.lineTo(g.x + r, g.y);
    ctx.moveTo(g.x, g.y - r); ctx.lineTo(g.x, g.y + r);
    ctx.stroke();
    ctx.restore();
    labelChip(g.x + r + 3 * inv, g.y - r, "GROUND TRUTH", MARKER_COLORS.gt);
  }

  // predicted final marker + error vector
  if (state.selected) {
    const s = state.selected;
    ctx.save();
    ctx.strokeStyle = "rgba(0,0,0,0.65)";
    ctx.lineWidth = 4.5 * inv;
    ctx.beginPath(); ctx.arc(s.x, s.y, 9 * inv, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.6 * inv;
    ctx.beginPath(); ctx.arc(s.x, s.y, 9 * inv, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();

    if (state.groundTruth) {
      ctx.save();
      ctx.strokeStyle = "rgba(0,0,0,0.65)";
      ctx.lineWidth = 3.4 * inv;
      ctx.setLineDash([6 * inv, 4 * inv]);
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(state.groundTruth.x, state.groundTruth.y); ctx.stroke();
      ctx.strokeStyle = MARKER_COLORS.error;
      ctx.lineWidth = 1.8 * inv;
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(state.groundTruth.x, state.groundTruth.y); ctx.stroke();
      ctx.restore();
    }
  }

  ctx.restore();
  $("#searchDims").textContent = `${img.naturalWidth}\u00d7${img.naturalHeight}px  \u00b7  zoom ${state.view.scale.toFixed(1)}\u00d7`;
}

function attachZoomPan() {
  const canvas = $("#searchCanvas");
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const newScale = Math.min(8, Math.max(1, state.view.scale * factor));
    const total = state.view.baseScale * state.view.scale;
    const imgX = (mx - state.view.offsetX) / total;
    const imgY = (my - state.view.offsetY) / total;
    state.view.scale = newScale;
    const newTotal = state.view.baseScale * state.view.scale;
    state.view.offsetX = mx - imgX * newTotal;
    state.view.offsetY = my - imgY * newTotal;
    drawSearch();
  }, { passive: false });

  canvas.addEventListener("mousedown", (e) => {
    state.view.dragging = true;
    state.view.lastX = e.clientX;
    state.view.lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => { state.view.dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!state.view.dragging) return;
    state.view.offsetX += e.clientX - state.view.lastX;
    state.view.offsetY += e.clientY - state.view.lastY;
    state.view.lastX = e.clientX;
    state.view.lastY = e.clientY;
    drawSearch();
  });

  $("#btnZoomReset").addEventListener("click", () => {
    state.view.scale = 1; state.view.offsetX = 0; state.view.offsetY = 0;
    drawSearch();
  });
}

window.addEventListener("resize", () => {
  drawReference(); drawSearch();
  if (pendingCropImage) setupCropCanvas(true);
});

/* ============================================================ examples */
async function loadExamples() {
  const res = await fetch("/api/examples");
  const data = await res.json();
  state.examples = data.examples;
  const rail = $("#exampleRail");
  rail.innerHTML = "";
  state.examples.forEach((ex) => {
    const chip = document.createElement("button");
    chip.className = "scenario-chip";
    chip.dataset.id = ex.id;
    const dotClass = ex.expected_category === "success" ? "dot-success"
      : ex.expected_category === "candidate_generation_miss" ? "dot-genmiss"
      : "dot-rankfail";
    chip.innerHTML = `<span class="dot ${dotClass}"></span>${ex.label}`;
    chip.addEventListener("click", () => selectExample(ex));
    rail.appendChild(chip);
  });
  if (state.examples.length) selectExample(state.examples[0]);
}

function selectExample(ex) {
  resetPipeline();
  state.mode = "example";
  state.currentExample = ex;
  $$(".scenario-chip").forEach((c) => c.classList.toggle("selected", c.dataset.id === ex.id));
  $("#scenarioDesc").innerHTML = `<b>${ex.style.toUpperCase()}${ex.hard_case ? " \u00b7 hard case" : ""}${ex.heavier_noise ? " \u00b7 heavier noise" : ""}</b> \u2014 ${ex.note}`;

  state.refImg = new Image();
  state.refImg.onload = drawReference;
  state.refImg.src = ex.reference_url;

  state.searchImg = new Image();
  state.searchImg.onload = drawSearch;
  state.searchImg.src = ex.search_url;

  state.groundTruth = null; // revealed only after the real result confirms it
  state.gtSource = null;
  $("#btnRun").disabled = false;
  $("#legendLine").innerHTML =
    `<span style="color:${MARKER_COLORS.candidate}">\u25a2</span> raw NCC candidate &nbsp; ` +
    `<span style="color:${MARKER_COLORS.verified}">\u25a2</span> verified &amp; ranked &nbsp; ` +
    `<span style="color:${MARKER_COLORS.selected}">\u25a2</span> selected &nbsp; ` +
    `<span style="color:${MARKER_COLORS.gt}">+</span> ground truth &nbsp; ` +
    `<span style="color:${MARKER_COLORS.error}">- -</span> error vector`;
}

/* ============================================================ pipeline reset */
function resetPipeline() {
  if (state.es) { state.es.close(); state.es = null; }
  state.candidates = [];
  state.ranked = [];
  state.selected = null;
  state.groundTruth = null;
  state.diagnosisTrueRank = null;
  state.view.scale = 1; state.view.offsetX = 0; state.view.offsetY = 0;
  initTracker();
  $("#feed").innerHTML = '<div class="placeholder">Select a scenario and click Run Drift-Sense to watch the real backend execute.</div>';
  $("#rankTable").style.display = "none";
  $("#rankBody").innerHTML = "";
  $("#rankPlaceholder").style.display = "block";
  $("#resultPanel").style.display = "none";
  $("#scanline").classList.remove("active");
  drawReference();
  drawSearch();
}

/* ============================================================ ranking table */
function renderRankTable() {
  const body = $("#rankBody");
  body.innerHTML = "";
  $("#rankTable").style.display = state.ranked.length ? "table" : "none";
  $("#rankPlaceholder").style.display = state.ranked.length ? "none" : "block";

  state.ranked.forEach((c) => {
    const tr = document.createElement("tr");
    const isSelected = state.selected && c.x === state.selected.x && c.y === state.selected.y;
    const isGT = state.diagnosisTrueRank !== null && state.diagnosisTrueRank !== undefined && c.rank === state.diagnosisTrueRank;
    if (isSelected) tr.classList.add("selected");
    if (isGT) tr.classList.add("gt-row");
    tr.innerHTML = `
      <td><span class="rank-badge">${c.rank}</span></td>
      <td>(${c.x.toFixed(0)}, ${c.y.toFixed(0)})</td>
      <td>${c.ncc_score.toFixed(3)}</td>
      <td>${c.orb_inliers}</td>
      <td>${c.final_score.toFixed(3)}${isSelected ? '<span class="tag-pill tag-selected">SELECTED</span>' : ""}${isGT ? '<span class="tag-pill tag-gt">GROUND TRUTH</span>' : ""}</td>`;
    body.appendChild(tr);
  });
}

/* ============================================================ result panel */
function renderResult(resultData, diagnosis) {
  const panel = $("#resultPanel");
  panel.style.display = "block";
  const grid = $("#resultGrid");

  const ambTip = `Top two candidates' final scores were within ${state.config.ambiguous_gap_threshold.toFixed(2)} of each other (the backend's own ambiguity rule). This is independent of whether the result is actually correct.`;

  const stats = [
    { k: "Predicted X", v: resultData.x.toFixed(1) + "px" },
    { k: "Predicted Y", v: resultData.y.toFixed(1) + "px" },
    { k: "Confidence (final score)", v: resultData.confidence.toFixed(3), cls: "accent" },
    { k: "Runtime", v: resultData.runtime_sec.toFixed(2) + "s" },
    { k: "Candidates Considered", v: resultData.candidates_considered },
    { k: "Backend Ambiguous Flag", v: resultData.ambiguous ? "TRUE" : "FALSE", cls: resultData.ambiguous ? "warn" : "good", tip: ambTip },
  ];
  if (diagnosis) {
    const srcNote = diagnosis.gt_source === "self_crop" ? " (from your target selection)" : " (evaluation annotation)";
    stats.push({ k: "Error vs Ground Truth", v: diagnosis.error_px.toFixed(1) + "px", cls: diagnosis.category === "success" ? "good" : "bad", tip: "Straight-line distance between the predicted center and ground truth" + srcNote + "." });
  }
  grid.innerHTML = stats.map(s => `<div class="stat ${s.cls || ""}">
      <div class="k">${s.k}${s.tip ? `<span class="info-dot">i</span>` : ""}</div>
      <div class="v">${s.v}</div>
      ${s.tip ? `<div class="tip">${s.tip}</div>` : ""}
    </div>`).join("");

  const banner = $("#diagnosisBanner");
  if (!diagnosis) {
    banner.innerHTML = "";
    $("#resultExplain").textContent = "No ground truth is available for this input, so accuracy cannot be evaluated -- only the predicted location above is real backend output.";
    return;
  }

  const titleMap = {
    success: "LOCALIZATION SUCCESSFUL",
    candidate_generation_miss: "CANDIDATE-GENERATION MISS",
    ranking_selection_failure: "RANKING / SELECTION FAILURE",
    true_ambiguity: "GENUINE AMBIGUITY DETECTED",
  };
  const classMap = {
    success: "diag-success",
    candidate_generation_miss: "diag-genmiss",
    ranking_selection_failure: "diag-rankfail",
    true_ambiguity: "diag-tie",
  };
  banner.innerHTML = `<div class="diagnosis-banner ${classMap[diagnosis.category]}">
    <span class="dg-title">${titleMap[diagnosis.category]}</span>
    ${diagnosis.message}
  </div>`;
  $("#resultExplain").textContent = "";

  state.diagnosisTrueRank = diagnosis.true_rank;
  renderRankTable();
}

/* ============================================================ SSE run */
function runStream(url) {
  resetPipeline();
  $("#btnRun").disabled = true;

  const es = new EventSource(url);
  state.es = es;
  let streamEnded = false; // guards against onerror firing after we've already handled completion

  es.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    const { stage, status, message, data } = ev;

    setStage(stage, status, trackerSubText(stage, status, data));
    appendFeed(stage, status, message, data);

    if (stage === "candidate_generation") {
      $("#scanline").classList.toggle("active", status === "running");
      if (status === "complete" && data.candidates) {
        state.candidates = data.candidates;
        drawSearch();
      }
    }

    if (stage === "verification" && data && data.candidate) {
      drawSearch();
    }

    if (stage === "ranking" && data && data.ranking) {
      state.ranked = data.ranking;
      renderRankTable();
      drawSearch();
    }

    if (stage === "decision" && data && data.selected) {
      state.selected = data.selected;
      drawSearch();
      renderRankTable();
    }

    if (stage === "result") {
      streamEnded = true;
      $("#scanline").classList.remove("active");
      if (status === "complete") {
        const result = data.result;
        const diagnosis = data.diagnosis;
        if (diagnosis) {
          state.groundTruth = diagnosis.ground_truth;
          state.gtSource = diagnosis.gt_source;
        }
        renderResult(result, diagnosis);
        drawSearch();
      } else {
        $("#resultPanel").style.display = "block";
        $("#diagnosisBanner").innerHTML = `<div class="diagnosis-banner diag-genmiss">
          <span class="dg-title">LOCALIZATION COULD NOT BE CONFIRMED</span>
          ${message}</div>`;
        $("#resultGrid").innerHTML = "";
        $("#resultExplain").textContent = "";
      }
      es.close();
      $("#btnRun").disabled = false;
    }
  };

  es.onerror = () => {
    if (streamEnded) return; // stream already completed normally; a trailing close event is not an error
    streamEnded = true;

    // A true EventSource-level error (as opposed to a normal in-stream
    // "result: failed" event, handled above) means the connection itself
    // dropped mid-run -- e.g. the server process was killed or the proxy
    // returned a 502. Mark every stage that never reached "done" as
    // failed instead of leaving it stuck mid-animation, and surface a
    // specific, actionable message rather than a generic one.
    STAGES.forEach((s) => {
      const step = $(`#step-${s.key}`);
      if (step && !step.classList.contains("done")) {
        setStage(s.key, "failed");
      }
    });
    $("#scanline").classList.remove("active");

    const msg = "The connection to the server was lost while this was running. " +
      "This usually means the request took too long or the server ran out of memory " +
      "processing the image. Try a smaller image, or click Run again.";
    appendFeed("connection", "failed", msg, {});
    $("#resultPanel").style.display = "block";
    $("#diagnosisBanner").innerHTML = `<div class="diagnosis-banner diag-genmiss">
      <span class="dg-title">CONNECTION LOST</span>
      ${msg}</div>`;
    $("#resultGrid").innerHTML = "";
    $("#resultExplain").textContent = "";

    es.close();
    $("#btnRun").disabled = false;
  };
}

$("#btnRun").addEventListener("click", () => {
  if (state.mode === "example" && state.currentExample) {
    runStream(`/api/run/stream?mode=example&example_id=${encodeURIComponent(state.currentExample.id)}`);
  } else if (state.mode === "upload" && state.upload.refToken && state.upload.searchToken) {
    const g = state.upload.selfGT;
    runStream(`/api/run/stream?mode=upload&reference_token=${encodeURIComponent(state.upload.refToken)}&search_token=${encodeURIComponent(state.upload.searchToken)}&self_gt_x=${g.x}&self_gt_y=${g.y}`);
  }
});
$("#btnReset").addEventListener("click", resetPipeline);
$("#btnAnother").addEventListener("click", () => {
  if (state.mode !== "example" || !state.examples.length) return;
  const idx = state.examples.findIndex(e => e.id === (state.currentExample && state.currentExample.id));
  const next = state.examples[(idx + 1) % state.examples.length];
  selectExample(next);
});

/* ============================================================ mode toggle */
$("#modeToggle").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;
  $$("#modeToggle button").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  state.mode = btn.dataset.mode;
  $("#examplePanel").style.display = state.mode === "example" ? "block" : "none";
  $("#uploadPanel").style.display = state.mode === "upload" ? "block" : "none";
  resetPipeline();
  if (state.mode === "upload") {
    state.refImg = null; state.searchImg = null;
    drawReference(); drawSearch();
    $("#legendLine").innerHTML = "";
    $("#btnRun").disabled = true;
  } else if (state.currentExample) {
    selectExample(state.currentExample);
  }
});

/* ============================================================ upload mode: single image -> crop template */
let pendingCropImage = null; // full-resolution Image object of the uploaded search image
let cropRect = null, cropDragging = false, cropStart = null;

function wireDropzone(dzId, fileInputId, onFile) {
  const dz = $(dzId), input = $(fileInputId);
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) onFile(input.files[0]); });
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Upload failed.");
  return data;
}

wireDropzone("#dzSearch", "#fileSearch", async (file) => {
  try {
    $("#uploadError").innerHTML = "";
    const data = await uploadFile(file);
    state.upload.searchToken = data.token;

    pendingCropImage = new Image();
    pendingCropImage.onload = () => {
      $("#uploadStep1").style.display = "none";
      $("#uploadStep2").style.display = "block";
      setupCropCanvas(true);
    };
    pendingCropImage.src = data.url;
  } catch (err) {
    $("#uploadError").innerHTML = `<div class="error-banner">${err.message}</div>`;
  }
});

function cropFit() {
  const canvas = $("#cropCanvas");
  const img = pendingCropImage;
  const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
  return { scale, ox: (canvas.width - w) / 2, oy: (canvas.height - h) / 2, w, h };
}

function setupCropCanvas(keepRect) {
  const canvas = $("#cropCanvas");
  const wrap = canvas.parentElement;
  canvas.width = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  if (!keepRect) cropRect = null;
  drawCropCanvas();

  canvas.onmousedown = (e) => {
    const r = canvas.getBoundingClientRect();
    cropStart = { x: e.clientX - r.left, y: e.clientY - r.top };
    cropDragging = true;
  };
  canvas.onmousemove = (e) => {
    if (!cropDragging) return;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    cropRect = {
      x: Math.min(cropStart.x, x), y: Math.min(cropStart.y, y),
      w: Math.abs(x - cropStart.x), h: Math.abs(y - cropStart.y),
    };
    drawCropCanvas();
  };
  window.addEventListener("mouseup", () => { cropDragging = false; });
}

function drawCropCanvas() {
  const canvas = $("#cropCanvas");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!pendingCropImage) return;
  const fit = cropFit();
  ctx.drawImage(pendingCropImage, fit.ox, fit.oy, fit.w, fit.h);

  if (cropRect && cropRect.w > 2 && cropRect.h > 2) {
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(fit.ox, fit.oy, fit.w, cropRect.y - fit.oy);
    ctx.fillRect(fit.ox, cropRect.y + cropRect.h, fit.w, fit.oy + fit.h - (cropRect.y + cropRect.h));
    ctx.fillRect(fit.ox, cropRect.y, cropRect.x - fit.ox, cropRect.h);
    ctx.fillRect(cropRect.x + cropRect.w, cropRect.y, fit.ox + fit.w - (cropRect.x + cropRect.w), cropRect.h);
    ctx.strokeStyle = "#0a84ff";
    ctx.lineWidth = 2;
    ctx.strokeRect(cropRect.x, cropRect.y, cropRect.w, cropRect.h);
    ctx.restore();
  }
}

$("#btnCropConfirm").addEventListener("click", async () => {
  if (!cropRect || cropRect.w < 6 || cropRect.h < 6) {
    $("#uploadError").innerHTML = `<div class="error-banner">Drag a box around a pattern first.</div>`;
    return;
  }
  $("#uploadError").innerHTML = "";
  const fit = cropFit();
  const nx = (cropRect.x - fit.ox) / fit.scale;
  const ny = (cropRect.y - fit.oy) / fit.scale;
  const nw = cropRect.w / fit.scale;
  const nh = cropRect.h / fit.scale;

  // Ground truth by construction: the crop's center in the ORIGINAL
  // (search) image's pixel space really is the target location -- not an
  // estimate, since we cropped it directly out of that same image.
  const selfGT = { x: nx + nw / 2, y: ny + nh / 2 };

  // The backend assumes the reference is a ~1/scale zoomed-in depiction of
  // the target (see /api/config -> reference_scale_range, ~0.07-0.15x).
  // A same-resolution crop of the search image is at the WRONG scale for
  // that convention and will not be found. So we upscale the crop by the
  // backend's own recommended factor before sending it as the reference --
  // this makes the crop's pixel content match what the pipeline expects a
  // "reference chip" to look like, without fabricating any new content.
  const factor = state.config.recommended_reference_upscale_factor || 9.1;
  const off = document.createElement("canvas");
  off.width = Math.max(8, Math.round(nw * factor));
  off.height = Math.max(8, Math.round(nh * factor));
  const octx = off.getContext("2d");
  octx.imageSmoothingEnabled = true;
  octx.imageSmoothingQuality = "high";
  octx.drawImage(pendingCropImage, nx, ny, nw, nh, 0, 0, off.width, off.height);

  off.toBlob(async (blob) => {
    try {
      const file = new File([blob], "target_reference.png", { type: "image/png" });
      const data = await uploadFile(file);
      state.upload.refToken = data.token;
      state.upload.selfGT = selfGT;

      state.refImg = new Image();
      state.refImg.onload = drawReference;
      state.refImg.src = data.url;

      state.searchImg = pendingCropImage;
      drawSearch();

      state.groundTruth = null; // hidden until the real result is computed
      $("#legendLine").innerHTML =
        `<span style="color:${MARKER_COLORS.candidate}">\u25a2</span> raw NCC candidate &nbsp; ` +
        `<span style="color:${MARKER_COLORS.verified}">\u25a2</span> verified &amp; ranked &nbsp; ` +
        `<span style="color:${MARKER_COLORS.selected}">\u25a2</span> selected &nbsp; ` +
        `<span style="color:${MARKER_COLORS.gt}">+</span> your target (ground truth) &nbsp; ` +
        `<span style="color:${MARKER_COLORS.error}">- -</span> error vector`;

      $("#uploadPanel").style.display = "none";
      $("#btnRun").disabled = false;
    } catch (err) {
      $("#uploadError").innerHTML = `<div class="error-banner">${err.message}</div>`;
    }
  }, "image/png");
});

$("#btnCropRestart").addEventListener("click", () => {
  pendingCropImage = null;
  cropRect = null;
  state.upload = { refToken: null, searchToken: null, selfGT: null };
  $("#uploadStep2").style.display = "none";
  $("#uploadStep1").style.display = "block";
  $("#fileSearch").value = "";
});

/* ============================================================ boot */
initTracker();
attachZoomPan();
loadConfig();
loadExamples();
