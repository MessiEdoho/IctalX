"use strict";
/* IctalX interface logic. Talks to the API in ictalx/server.py (spec section 12). */

const REQUIRED_FS = 500;
const POLL_MS = 1000;

const S = {
  fmt: "EDF",
  file: null,
  edf: null,            // parsed EDF header
  csvColumns: [],
  xhr: null,            // upload in progress
  analysisId: null,
  pollTimer: null,
  result: null,         // /api/analyses/{id} body when done
  events: [],           // event rows, in time order
  selected: null,       // event_id
  draft: null,          // "confirmed" | "rejected" | null (unsaved)
  windows: {},          // event_id -> /api/events body
  downloaded: false,
  pendingTarget: null,  // event_id waiting behind the unsaved prompt
};

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

/* ---------------------------------------------------------------- helpers */
function humanSize(n) {
  return n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(1) + " KB"
    : n < 1073741824 ? (n / 1048576).toFixed(1) + " MB" : (n / 1073741824).toFixed(2) + " GB";
}
function hms(sec) {            // same format as the training code: 00h14m10.0
  const neg = sec < 0; sec = Math.abs(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec - h * 3600 - m * 60;
  return (neg ? "−" : "") + String(h).padStart(2, "0") + "h" + String(m).padStart(2, "0") + "m" + s.toFixed(1).padStart(4, "0");
}
function duration(sec) {
  if (sec >= 3600) return { v: (sec / 3600).toFixed(1), u: "h" };
  if (sec >= 60) return { v: (sec / 60).toFixed(1), u: "min" };
  return { v: sec.toFixed(1), u: "s" };
}
function statusLabel(s) { return s === "confirmed" ? "Confirmed" : s === "rejected" ? "Rejected" : "Pending"; }
let toastTimer;
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 4200);
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/* --------------------------------------------------------- hero animation */
function initScope() {
  const cv = $("#scopeCanvas"), ctx = cv.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  function resize() { const r = cv.getBoundingClientRect(); cv.width = r.width * dpr; cv.height = r.height * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
  resize(); window.addEventListener("resize", resize);
  const still = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  let t = still ? 2.2 : 0;
  function frame() {
    const w = cv.width / dpr, h = cv.height / dpr, mid = h / 2;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(155,152,242,.08)"; ctx.lineWidth = 1;
    for (let x = 0; x < w; x += w / 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
    const burst = w * 0.5 + Math.sin(t * 0.3) * w * 0.18;
    ctx.lineWidth = 2; ctx.beginPath();
    for (let x = 0; x <= w; x++) {
      const p = x / w, d = Math.abs(x - burst);
      let y = Math.sin(p * 40 + t) * 4 + Math.sin(p * 13 - t * 0.7) * 3;
      y += Math.exp(-(d * d) / (2 * Math.pow(w * 0.045, 2))) * Math.sin(x * 0.9 + t * 4) * 46 * (0.7 + 0.3 * Math.sin(t * 2));
      if (x === 0) ctx.moveTo(x, mid - y); else ctx.lineTo(x, mid - y);
    }
    const g = ctx.createLinearGradient(burst - w * 0.14, 0, burst + w * 0.14, 0);
    g.addColorStop(0, "#9B98F2"); g.addColorStop(0.5, "#F0A81E"); g.addColorStop(1, "#9B98F2");
    ctx.strokeStyle = g; ctx.stroke();
    if (!still) { t += 0.05; requestAnimationFrame(frame); }
  }
  frame();
}

/* ------------------------------------------------------- EDF header (V04/V05/V09 pre-checks) */
async function readEdfHeader(file) {
  const ascii = bytes => String.fromCharCode.apply(null, bytes).trim();
  const head = new Uint8Array(await file.slice(0, 256).arrayBuffer());
  const ns = parseInt(ascii(head.slice(252, 256)), 10);
  const recDur = parseFloat(ascii(head.slice(244, 252)));
  if (!(ns > 0) || !(recDur > 0)) throw new Error("bad header");
  const sig = new Uint8Array(await file.slice(256, 256 + 256 * ns).arrayBuffer());
  if (sig.length < 256 * ns) throw new Error("bad header");
  // field widths in the EDF signal header, in order
  const widths = [["label", 16], ["transducer", 80], ["unit", 8], ["pmin", 8], ["pmax", 8],
                  ["dmin", 8], ["dmax", 8], ["prefilter", 80], ["nsamp", 8], ["reserved", 32]];
  const fields = {}; let off = 0;
  for (const [name, w] of widths) {
    fields[name] = [];
    for (let i = 0; i < ns; i++) fields[name].push(ascii(sig.slice(off + i * w, off + (i + 1) * w)));
    off += w * ns;
  }
  const channels = [];
  for (let i = 0; i < ns; i++) {
    if (fields.label[i] === "EDF Annotations") continue;
    channels.push({ label: fields.label[i], unit: fields.unit[i], fs: parseInt(fields.nsamp[i], 10) / recDur });
  }
  const d = ascii(head.slice(168, 176)), t = ascii(head.slice(176, 184));
  let start = null;
  const dm = /^(\d\d)\.(\d\d)\.(\d\d)$/.exec(d), tm = /^(\d\d)\.(\d\d)\.(\d\d)$/.exec(t);
  if (dm && tm && d !== "00.00.00" && d !== "01.01.85") {
    const yy = parseInt(dm[3], 10), year = yy >= 85 ? 1900 + yy : 2000 + yy;
    start = `${year}-${dm[2]}-${dm[1]} ${tm[1]}:${tm[2]}:${tm[3]}`;
  }
  return { channels, start };
}

/* ------------------------------------------------------------ CSV header */
async function readCsvHeader(file) {
  const text = await file.slice(0, 65536).text();
  const line = text.split(/\r?\n/)[0] || "";
  const cols = []; let cur = "", q = false;
  for (const ch of line) {
    if (ch === '"') q = !q;
    else if (ch === "," && !q) { cols.push(cur.trim()); cur = ""; }
    else cur += ch;
  }
  cols.push(cur.trim());
  return cols.filter(c => c !== "");
}

/* -------------------------------------------------------------- upload card */
function setFormat(fmt) {
  if (busy()) return;
  S.fmt = fmt;
  $$(".seg button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.fmt === fmt)));
  $("#dzFmt").textContent = fmt;
  $("#fileInput").setAttribute("accept", fmt === "EDF" ? ".edf" : ".csv");
  clearFile();
}

function busy() { return !!(S.xhr || S.pollTimer); }

function clearFile() {
  S.file = null; S.edf = null; S.csvColumns = [];
  $("#fileInput").value = "";
  $("#filechip").hidden = true;
  $("#edfFields").hidden = true;
  $("#csvFields").hidden = true;
  hideError();
  updateDetectButton();
}

async function selectFile(file) {
  if (busy()) return;
  const ext = file.name.toLowerCase().split(".").pop();
  clearFile();
  S.file = file;
  $("#fileName").textContent = file.name;
  $("#fileSize").textContent = humanSize(file.size);
  $("#filechip").hidden = false;
  if (ext !== S.fmt.toLowerCase()) { showError("V01", "This file is not a valid EDF or CSV file.", true); return; }
  if (file.size > 2 * 1024 ** 3) { showError("V07", "Files larger than 2 GB are not supported.", true); return; }
  try {
    if (S.fmt === "EDF") {
      S.edf = await readEdfHeader(file);
      const sel = $("#edfChan"); sel.innerHTML = "";
      S.edf.channels.forEach(c => {
        const o = el("option", null, `${c.label} (${c.fs} Hz)`); o.value = c.label; sel.appendChild(o);
      });
      $("#edfChanHint").textContent = `${S.edf.channels.length} channel${S.edf.channels.length === 1 ? "" : "s"} in file`;
      $("#edfStart").textContent = S.edf.start || "missing";
      $("#edfFields").hidden = false;
      onChannelChange();
    } else {
      S.csvColumns = await readCsvHeader(file);
      const sig = $("#sigCol"), tc = $("#timeCol");
      sig.innerHTML = ""; tc.innerHTML = "";
      tc.appendChild(Object.assign(el("option", null, "None"), { value: "" }));
      S.csvColumns.forEach(c => {
        sig.appendChild(Object.assign(el("option", null, c), { value: c }));
        tc.appendChild(Object.assign(el("option", null, c), { value: c }));
      });
      const eeg = S.csvColumns.find(c => c.toLowerCase() === "eeg");
      if (eeg) sig.value = eeg;
      $("#csvFields").hidden = false;
    }
  } catch (e) {
    showError("V02", "The file appears to be damaged and could not be read.", true);
  }
  updateDetectButton();
}

function onChannelChange() {
  hideError();
  const c = S.edf && S.edf.channels.find(x => x.label === $("#edfChan").value);
  $("#edfRate").textContent = c ? `${c.fs} Hz` : "—";
  $("#edfUnit").textContent = c ? (c.unit || "—") : "—";
  if (c && c.fs !== REQUIRED_FS) showError("V05", `Sampling rate is ${c.fs} Hz. IctalX currently accepts ${REQUIRED_FS} Hz recordings only.`, true);
  else if (S.edf && !S.edf.start) showError("V09", "This EDF file has no recording start time.", true);
  updateDetectButton();
}

function updateDetectButton() {
  let ok = !!S.file && $("#errorBox").hidden && !busy();
  if (ok && S.fmt === "EDF") ok = !!(S.edf && S.edf.channels.length);
  if (ok && S.fmt === "CSV") ok = S.csvColumns.length > 0 && $("#sfreq").value !== "";
  $("#detectBtn").disabled = !ok;
}

function showError(code, message, local, title) {
  $("#errorBox .t").textContent = title || "File not accepted";
  $("#errorMsg").textContent = message;
  $("#errorCode").textContent = code ? "Check " + code : "";
  $("#errorBox .s").lastChild.textContent = (code ? " · " : "") + (local
    ? "Checked in your browser; nothing was uploaded."
    : "The file was not analysed and has not been kept.");
  $("#errorBox").hidden = false;
  updateDetectButton();
}
function hideError() { $("#errorBox").hidden = true; }

function initUpload() {
  $$(".seg button").forEach(b => b.addEventListener("click", () => setFormat(b.dataset.fmt)));
  const dz = $("#dropzone"), fi = $("#fileInput");
  dz.addEventListener("click", () => { if (!busy()) fi.click(); });
  dz.addEventListener("keydown", e => { if ((e.key === "Enter" || e.key === " ") && !busy()) { e.preventDefault(); fi.click(); } });
  fi.addEventListener("change", () => { if (fi.files[0]) selectFile(fi.files[0]); });
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", e => { if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]); });
  $("#fileRemove").addEventListener("click", () => { if (!busy()) clearFile(); });
  $("#edfChan").addEventListener("change", onChannelChange);
  ["#sfreq", "#sigCol", "#timeCol", "#unit", "#startT"].forEach(id => $(id).addEventListener("input", () => { hideError(); updateDetectButton(); }));
  $("#detectBtn").addEventListener("click", startDetection);
  $("#cancelBtn").addEventListener("click", cancelDetection);
}

/* ---------------------------------------------------------------- progress */
function setStep(step, extra) {
  const order = ["upload", "validate", "queue", "detect", "windows"];
  const k = order.indexOf(step);
  $$("#steps li[data-step]").forEach(li => {
    const i = order.indexOf(li.dataset.step);
    li.className = i < k ? "done" : i === k ? "now" : "";
  });
  const bar = $("#detectBar");
  bar.hidden = step !== "detect";
  if (!bar.hidden) $("#steps li[data-step='detect']").after(bar);
  if (extra) extra();
}

function lockCard(on) {
  $$(".seg button").forEach(b => b.disabled = on);
  $("#dropzone").setAttribute("aria-disabled", String(on));
  $$("#edfFields select, #csvFields input, #csvFields select").forEach(x => x.disabled = on);
  $("#fileRemove").hidden = on;
}

function startDetection() {
  if (S.result && !confirm("Start a new analysis? The current results will be cleared from this page.")) return;
  resetResults();
  hideError();
  const fd = new FormData();
  fd.append("file", S.file);
  fd.append("file_format", S.fmt);
  if (S.fmt === "EDF") fd.append("channel", $("#edfChan").value);
  else {
    fd.append("signal_column", $("#sigCol").value);
    fd.append("sampling_rate", $("#sfreq").value);
    fd.append("unit", $("#unit").value);
    if ($("#timeCol").value) fd.append("time_column", $("#timeCol").value);
    if ($("#startT").value.trim()) fd.append("start_time", $("#startT").value.trim());
  }

  $("#progId").textContent = "assigned after validation";
  $("#upPct").textContent = "0%"; $("#queuePos").textContent = ""; $("#segCount").textContent = "";
  $("#detectBar i").style.width = "0";
  $("#progress").hidden = false;
  $("#detectBtn").hidden = true;
  lockCard(true);
  setStep("upload");

  const xhr = new XMLHttpRequest();
  S.xhr = xhr;
  xhr.open("POST", "/api/detect");
  xhr.upload.onprogress = e => { if (e.lengthComputable) $("#upPct").textContent = Math.round(100 * e.loaded / e.total) + "%"; };
  xhr.upload.onload = () => { $("#upPct").textContent = "100%"; setStep("validate"); };
  xhr.onload = () => {
    S.xhr = null;
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
    if (xhr.status === 202) {
      S.analysisId = body.analysis_id;
      $("#progId").textContent = body.short_id;
      poll();
    } else if (xhr.status === 422) {
      endProgress();
      showError(body.code, body.message, false);
    } else {
      endProgress();
      showError("", "The server could not process this request. Please try again.", false, "Something went wrong");
    }
  };
  xhr.onerror = () => { S.xhr = null; endProgress(); toast("Upload failed: the server could not be reached."); };
  xhr.onabort = () => { S.xhr = null; endProgress(); toast("Upload cancelled."); };
  xhr.send(fd);
}

function endProgress() {
  clearTimeout(S.pollTimer); S.pollTimer = null;
  $("#progress").hidden = true;
  $("#detectBtn").hidden = false;
  lockCard(false);
  updateDetectButton();
}

async function poll() {
  S.pollTimer = null;
  let body;
  try {
    const r = await fetch(`/api/analyses/${S.analysisId}`);
    if (r.status === 404) { endProgress(); toast("This analysis has expired."); return; }
    body = await r.json();
  } catch (e) {
    S.pollTimer = setTimeout(poll, POLL_MS * 3);
    return;
  }
  if (body.status === "queued") {
    setStep("queue", () => { $("#queuePos").textContent = body.queue_position ? `position ${body.queue_position}` : ""; });
  } else if (body.status === "running" && body.stage === "detecting") {
    setStep("detect", () => {
      const pct = body.total_segments ? 100 * body.processed_segments / body.total_segments : 0;
      $("#segCount").textContent = body.total_segments
        ? `${body.processed_segments.toLocaleString("en-GB")} / ${body.total_segments.toLocaleString("en-GB")} segments` : "";
      $("#detectBar i").style.width = pct.toFixed(1) + "%";
    });
  } else if (body.status === "running") {
    setStep("windows");
  } else if (body.status === "done") {
    endProgress();
    showResults(body);
    return;
  } else if (body.status === "failed") {
    endProgress();
    showError("", body.error || "Detection failed.", false, "Detection failed");
    return;
  } else if (body.status === "cancelled") {
    endProgress();
    toast("Detection cancelled. The uploaded file has been deleted.");
    return;
  }
  S.pollTimer = setTimeout(poll, POLL_MS);
}

async function cancelDetection() {
  if (S.xhr) { S.xhr.abort(); return; }
  if (S.analysisId) {
    try { await fetch(`/api/analyses/${S.analysisId}/cancel`, { method: "POST" }); } catch (e) { /* poll will report */ }
    $("#cancelBtn").disabled = true;
    setTimeout(() => { $("#cancelBtn").disabled = false; }, 2000);
  }
}

/* ----------------------------------------------------------------- results */
function resetResults() {
  S.result = null; S.events = []; S.selected = null; S.draft = null; S.windows = {}; S.downloaded = false;
  $("#review").hidden = true;
  $("#sessionNotice").hidden = true;
  ["#stRec", "#stEvents", "#stIctal", "#stConf", "#stReviewed", "#stConfirmed", "#stRejected", "#stId"].forEach(id => $(id).textContent = "—");
}

function showResults(body) {
  S.result = body;
  S.events = body.events;
  const id = encodeURIComponent(body.analysis_id);
  $("#dlCsv").href = `/api/report?analysis_id=${id}&format=csv`;
  $("#dlZip").href = `/api/report?analysis_id=${id}&format=zip`;
  $("#review").hidden = false;
  $("#sessionNotice").hidden = sessionStorageGet("noticeDismissed") === "1";
  $("#noEvents").hidden = S.events.length > 0;
  $("#detail").hidden = S.events.length === 0;
  renderTable();
  updateStats();
  if (S.events.length) selectEvent(S.events[0].event_id, true);
  toast(`Detection complete: ${S.events.length} event${S.events.length === 1 ? "" : "s"}.`);
  $("#review").scrollIntoView({ behavior: "smooth", block: "start" });
}

function sessionStorageGet(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } }
function sessionStorageSet(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* ignore */ } }

function updateStats() {
  const s = S.result.summary, ev = S.events;
  const rec = duration(s.recording.duration_s);
  $("#stRec").innerHTML = ""; $("#stRec").append(rec.v, Object.assign(el("small", null, " " + rec.u)));
  $("#stEvents").textContent = ev.length;
  const ictal = duration(ev.reduce((a, e) => a + e.duration_s, 0));
  $("#stIctal").innerHTML = ""; $("#stIctal").append(ev.length ? ictal.v : "0", el("small", null, " " + (ev.length ? ictal.u : "s")));
  $("#stConf").innerHTML = "";
  if (ev.length) $("#stConf").append((100 * ev.reduce((a, e) => a + e.probability, 0) / ev.length).toFixed(0), el("small", null, "%"));
  else $("#stConf").textContent = "—";
  const reviewed = ev.filter(e => e.status !== "pending").length;
  $("#stReviewed").innerHTML = ""; $("#stReviewed").append(String(reviewed), el("small", null, "/" + ev.length));
  $("#stConfirmed").textContent = ev.filter(e => e.status === "confirmed").length;
  $("#stRejected").textContent = ev.filter(e => e.status === "rejected").length;
  $("#stId").textContent = S.result.short_id;
}

function visibleEvents() {
  const f = $("#filterSel").value, so = $("#sortSel").value;
  let list = S.events.filter(e => f === "all" || e.status === f);
  if (so === "prob") list = [...list].sort((a, b) => b.probability - a.probability);
  return list;
}

function renderTable() {
  const tb = $("#evBody"); tb.innerHTML = "";
  visibleEvents().forEach(e => {
    const tr = document.createElement("tr");
    tr.tabIndex = 0; tr.dataset.id = e.event_id;
    if (S.selected === e.event_id) { tr.className = "sel"; tr.setAttribute("aria-selected", "true"); }
    const num = parseInt(e.event_id.slice(3), 10);
    const cell = (main, sub, cls) => { const td = el("td", cls, main); if (sub) td.appendChild(el("span", "clk", sub)); return td; };
    tr.append(
      cell(String(num)),
      cell(e.onset_elapsed, e.onset_clock),
      cell(e.offset_elapsed, e.offset_clock, "c-off"),
      cell(e.duration_s.toFixed(1) + " s"),
      cell(e.probability.toFixed(2)),
    );
    const st = el("td"); st.appendChild(el("span", "pill " + e.status, statusLabel(e.status))); tr.appendChild(st);
    tr.addEventListener("click", () => requestSelect(e.event_id));
    tr.addEventListener("keydown", ev => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); requestSelect(e.event_id); }
    });
    tb.appendChild(tr);
  });
}

/* ------------------------------------------------------- selection + detail */
function requestSelect(id) {
  if (id === S.selected) return;
  if (S.draft) { openUnsaved(id); return; }
  selectEvent(id);
}

// Scroll only the table box (never the page), so the detail above stays where it is.
function keepRowInView(tr) {
  const box = $(".m-scroll"), head = box.querySelector("thead").offsetHeight;
  const top = tr.offsetTop - head, bottom = tr.offsetTop + tr.offsetHeight - box.clientHeight;
  if (box.scrollTop > top) box.scrollTop = top;
  else if (box.scrollTop < bottom) box.scrollTop = bottom;
}

function selectEvent(id, noFocus) {
  S.selected = id; S.draft = null;
  const e = S.events.find(x => x.event_id === id);
  $$("#evBody tr").forEach(tr => {
    const on = tr.dataset.id === id;
    tr.classList.toggle("sel", on); tr.setAttribute("aria-selected", String(on));
    if (on && !noFocus) tr.focus({ preventScroll: true });
    if (on) keepRowInView(tr);
  });
  $("#dEvId").textContent = parseInt(id.slice(3), 10);
  const withClock = (node, a, b) => { node.innerHTML = ""; node.append(a); if (b) node.appendChild(el("small", null, b)); };
  withClock($("#dOnset"), e.onset_elapsed, e.onset_clock);
  withClock($("#dOffset"), e.offset_elapsed, e.offset_clock);
  $("#dDur").textContent = e.duration_s.toFixed(1) + " s";
  $("#dProb").textContent = e.probability.toFixed(3);
  $("#dPeak").textContent = e.peak_uV.toFixed(1) + " µV";
  $("#dKurt").textContent = e.kurtosis.toFixed(2);
  $("#dSkew").textContent = e.skewness.toFixed(2);
  $("#dEnv").textContent = e.envelope_uV.toFixed(1) + " µV";
  showDecision(e.status, null);
  loadWindow(id);
}

function setPills(status) {
  ["#dPill", "#rvStatus"].forEach(sel => {
    const p = $(sel); p.className = "pill " + status;
    p.textContent = status === "pending" ? "Pending review" : statusLabel(status);
  });
}

function showDecision(saved, draft) {
  const shown = draft || saved;
  $("#tConf").setAttribute("aria-pressed", String(shown === "confirmed"));
  $("#tRej").setAttribute("aria-pressed", String(shown === "rejected"));
  $("#saveReview").disabled = !draft;
  $("#unsavedFlag").hidden = !draft;
  setPills(saved);
}

function setDraft(dec) {
  if (!S.selected) return;
  const e = S.events.find(x => x.event_id === S.selected);
  // Pressing the active draft clears it; choosing the saved status needs no draft.
  S.draft = S.draft === dec || e.status === dec ? null : dec;
  showDecision(e.status, S.draft);
}

async function saveReview() {
  if (!S.selected || !S.draft) return true;
  const e = S.events.find(x => x.event_id === S.selected);
  try {
    const r = await fetch("/api/review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ analysis_id: S.result.analysis_id, event_id: e.event_id, status: S.draft }),
    });
    if (!r.ok) throw new Error(r.status === 404 ? "This analysis has expired." : "The decision was not saved.");
  } catch (err) {
    toast(err.message || "The decision was not saved.");
    return false;
  }
  e.status = S.draft; S.draft = null; S.downloaded = false;
  showDecision(e.status, null);
  renderTable(); updateStats();
  $$("#evBody tr").forEach(tr => { if (tr.dataset.id === e.event_id) tr.classList.add("sel"); });
  toast(`Saved: event ${parseInt(e.event_id.slice(3), 10)} marked ${statusLabel(e.status)}.`);
  return true;
}

/* ------------------------------------------------------------- unsaved modal */
function openUnsaved(target) {
  S.pendingTarget = target;
  const n = parseInt(S.selected.slice(3), 10);
  $("#umText").textContent = `Event ${n} is marked ${statusLabel(S.draft)} but not saved. What would you like to do?`;
  $("#unsavedModal").hidden = false;
  $("#umSave").focus();
}
function closeUnsaved() {
  $("#unsavedModal").hidden = true;
  const row = $$("#evBody tr").find(tr => tr.dataset.id === S.selected);
  if (row) row.focus({ preventScroll: true });
}
function initModal() {
  $("#umStay").addEventListener("click", () => { S.pendingTarget = null; closeUnsaved(); });
  $("#umDiscard").addEventListener("click", () => { const t = S.pendingTarget; S.draft = null; $("#unsavedModal").hidden = true; selectEvent(t); });
  $("#umSave").addEventListener("click", async () => {
    const t = S.pendingTarget;
    if (await saveReview()) { $("#unsavedModal").hidden = true; selectEvent(t); }
  });
  $("#unsavedModal").addEventListener("keydown", e => {
    if (e.key === "Escape") { e.preventDefault(); S.pendingTarget = null; closeUnsaved(); }
    if (e.key === "Tab") {                                  // keep focus inside the dialog
      const b = [$("#umStay"), $("#umDiscard"), $("#umSave")];
      const i = b.indexOf(document.activeElement);
      e.preventDefault();
      b[(i + (e.shiftKey ? b.length - 1 : 1)) % b.length].focus();
    }
  });
}

/* ------------------------------------------------------------ keyboard */
function initKeys() {
  document.addEventListener("keydown", e => {
    if (!$("#unsavedModal").hidden || $("#review").hidden || !S.selected) return;
    const tag = (document.activeElement && document.activeElement.tagName) || "";
    if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "c") { e.preventDefault(); setDraft("confirmed"); }
    else if (k === "r") { e.preventDefault(); setDraft("rejected"); }
    else if (k === "s" || (k === "enter" && tag !== "TR" && tag !== "BUTTON" && tag !== "A")) { e.preventDefault(); saveReview(); }
    else if (k === "arrowdown" || k === "arrowup") {
      const list = visibleEvents(), i = list.findIndex(x => x.event_id === S.selected);
      const j = k === "arrowdown" ? i + 1 : i - 1;
      if (j >= 0 && j < list.length) { e.preventDefault(); requestSelect(list[j].event_id); }
    }
  });
}

/* ------------------------------------------------------------ signal views */
async function loadWindow(id) {
  const msg = $("#vizLoading");
  if (!S.windows[id]) {
    msg.textContent = "Loading signal…"; msg.hidden = false;
    try {
      const r = await fetch(`/api/events/${S.result.analysis_id}/${id}`);
      if (!r.ok) throw new Error();
      S.windows[id] = await r.json();
    } catch (e) {
      if (S.selected === id) msg.textContent = "The signal could not be loaded.";
      return;
    }
  }
  if (S.selected !== id) return;            // the user moved on while this was loading
  msg.hidden = true;
  drawWaveform(S.windows[id]);
  drawSpectrogram(S.windows[id]);
}

function sizeCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.getBoundingClientRect().width, h = cv.getBoundingClientRect().height;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function hatch(ctx, x0, x1, top, bottom) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x0, top, x1 - x0, bottom - top); ctx.clip();
  ctx.fillStyle = "rgba(255,255,255,.04)"; ctx.fillRect(x0, top, x1 - x0, bottom - top);
  ctx.strokeStyle = "rgba(155,152,242,.22)"; ctx.lineWidth = 1;
  for (let x = x0 - (bottom - top); x < x1; x += 7) { ctx.beginPath(); ctx.moveTo(x, bottom); ctx.lineTo(x + (bottom - top), top); ctx.stroke(); }
  ctx.restore();
}

function gapRuns(statuses) {             // runs of non-valid columns: [start, end, status]
  const runs = []; let i = 0;
  while (i < statuses.length) {
    if (statuses[i] === 2) { i++; continue; }
    let j = i, st = statuses[i];
    while (j < statuses.length && statuses[j] !== 2) { if (statuses[j] === 1) st = 1; j++; }
    runs.push([i, j, st]); i = j;
  }
  return runs;
}

function drawWaveform(d) {
  const { ctx, w, h } = sizeCanvas($("#wave"));
  const ws = d.window_start_s, we = d.window_end_s, tx = t => (t - ws) / (we - ws) * w;
  const top = 22, bottom = h - 20, mid = (top + bottom) / 2;
  const W = d.waveform, n = W.t.length, colW = w / n;

  ctx.strokeStyle = "rgba(155,152,242,.1)"; ctx.lineWidth = 1;
  for (let x = 0; x < w; x += w / 16) { ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); }
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();

  const sx = tx(d.onset_s), ex = tx(d.offset_s);
  ctx.fillStyle = "rgba(240,168,30,.16)"; ctx.fillRect(sx, top, Math.max(2, ex - sx), bottom - top);
  ctx.strokeStyle = "rgba(240,168,30,.7)"; ctx.lineWidth = 1.25;
  [sx, ex].forEach(x => { ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); });

  // Vertical scale: 110% of the largest swing inside the seizure, so brief artefacts
  // elsewhere in the window do not shrink it. Larger values are clipped at the edges.
  let amp = 0;
  for (let i = 0; i < n; i++) {
    const x = i * colW + colW / 2;
    if (W.status[i] === 2 && x >= sx && x <= ex) amp = Math.max(amp, Math.abs(W.lo[i]), Math.abs(W.hi[i]));
  }
  if (!amp) for (let i = 0; i < n; i++) if (W.status[i] === 2) amp = Math.max(amp, Math.abs(W.lo[i]), Math.abs(W.hi[i]));
  amp = (amp || 1) * 1.1;
  const ty = v => mid - Math.max(-1, Math.min(1, v / amp)) * (bottom - top) / 2;
  ctx.strokeStyle = "#B9B8F5"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    if (W.status[i] !== 2) continue;
    const x = i * colW + colW / 2;
    ctx.moveTo(x, ty(W.hi[i])); ctx.lineTo(x, ty(W.lo[i]) + 0.5);
  }
  ctx.stroke();
  // trace inside the seizure in amber
  ctx.strokeStyle = "#F0A81E"; ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = i * colW + colW / 2;
    if (W.status[i] !== 2 || x < sx || x > ex) continue;
    ctx.moveTo(x, ty(W.hi[i])); ctx.lineTo(x, ty(W.lo[i]) + 0.5);
  }
  ctx.stroke();

  ctx.font = "10px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textBaseline = "middle"; ctx.textAlign = "center";
  gapRuns(W.status).forEach(([a, b, st]) => {
    const x0 = a * colW, x1 = b * colW;
    hatch(ctx, x0, x1, top, bottom);
    if (x1 - x0 > 70) { ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.fillText(st === 1 ? "removed" : "no data", (x0 + x1) / 2, mid); }
  });

  ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  if (ws < 0) ctx.fillText(hms(0) + " · recording start", tx(0) + 4, h - 6);   // padding before the recording
  else ctx.fillText(hms(ws), 4, h - 6);
  ctx.textAlign = "right"; ctx.fillText(hms(we), w - 4, h - 6);
  ctx.textAlign = "left"; ctx.fillText("±" + amp.toFixed(1) + " z", 4, 13);
  ctx.fillStyle = "rgba(240,168,30,.95)"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText("◆ seizure " + hms(d.onset_s) + " – " + hms(d.offset_s), Math.min(Math.max((sx + ex) / 2, 110), w - 110), 5);
}

const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
function viridis(v) {
  const x = Math.max(0, Math.min(1, v)) * (VIRIDIS.length - 1), i = Math.floor(x), f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function drawSpectrogram(d) {
  const { ctx, w, h } = sizeCanvas($("#spectro"));
  const sp = d.spectrogram, nT = sp.t.length, nF = sp.f.length;
  const left = 0, right = w - 34, top = 4, bottom = h - 4;
  const ws = d.window_start_s, we = d.window_end_s, tx = t => left + (t - ws) / (we - ws) * (right - left);

  const vals = [];
  sp.db.forEach(col => { if (col) for (let k = 0; k < col.length; k += 3) vals.push(col[k]); });
  vals.sort((a, b) => a - b);
  const lo = vals.length ? vals[Math.floor(vals.length * 0.05)] : 0;
  const hi = vals.length ? vals[Math.floor(vals.length * 0.995)] : 1;

  const off = document.createElement("canvas"); off.width = nT; off.height = nF;
  const octx = off.getContext("2d"), img = octx.createImageData(nT, nF);
  for (let c = 0; c < nT; c++) {
    const col = sp.db[c];
    for (let k = 0; k < nF; k++) {
      const p = ((nF - 1 - k) * nT + c) * 4;
      if (!col) { img.data[p + 3] = 0; continue; }
      const [r, g, b] = viridis((col[k] - lo) / (hi - lo || 1));
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  const hop = nT > 1 ? sp.t[1] - sp.t[0] : 1;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, tx(sp.t[0] - hop / 2), top, tx(sp.t[nT - 1] + hop / 2) - tx(sp.t[0] - hop / 2), bottom - top);

  // hatch regions with no spectrogram (padding, removed segments, edges)
  const nulls = sp.db.map(c => (c ? 2 : 0));
  gapRuns(nulls).forEach(([a, b]) => hatch(ctx, tx(sp.t[a] - hop / 2), tx(sp.t[Math.min(b, nT) - 1] + hop / 2), top, bottom));
  if (tx(sp.t[0] - hop / 2) > left) hatch(ctx, left, tx(sp.t[0] - hop / 2), top, bottom);
  if (tx(sp.t[nT - 1] + hop / 2) < right) hatch(ctx, tx(sp.t[nT - 1] + hop / 2), right, top, bottom);

  ctx.strokeStyle = "#E03131"; ctx.lineWidth = 2;
  ctx.strokeRect(tx(d.onset_s), top + 1, tx(d.offset_s) - tx(d.onset_s), bottom - top - 2);

  ctx.font = "10px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.fillStyle = "rgba(255,255,255,.6)"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
  const fmax = sp.f[nF - 1];
  for (let f = 0; f <= fmax + 0.1; f += 50) {
    const y = bottom - f / fmax * (bottom - top);
    ctx.fillRect(right, y, 4, 1);
    ctx.fillText(String(f), right + 6, Math.min(Math.max(y, top + 6), bottom - 6));
  }
}

/* ------------------------------------------------------------ samples */
async function loadSamples() {
  const box = $("#sampleList");
  try {
    const r = await fetch("/api/samples");
    const { samples } = await r.json();
    box.innerHTML = "";
    if (!samples.length) { box.appendChild(el("div", "hint", "No sample files are available yet.")); return; }
    samples.forEach(s => {
      const row = el("div", "samplerow");
      row.append(el("span", "fi", s.name), el("span", "meta", s.name.split(".").pop().toUpperCase() + " · " + humanSize(s.bytes)));
      const a = el("a", null, "Download"); a.href = `/api/samples/${encodeURIComponent(s.name)}`; a.setAttribute("download", s.name);
      row.appendChild(a); box.appendChild(row);
    });
  } catch (e) {
    box.innerHTML = ""; box.appendChild(el("div", "hint", "Sample files could not be listed."));
  }
}

/* ------------------------------------------------------------ boot */
window.addEventListener("DOMContentLoaded", () => {
  initScope();
  initUpload();
  initModal();
  initKeys();
  setFormat("EDF");
  loadSamples();
  $("#tConf").addEventListener("click", () => setDraft("confirmed"));
  $("#tRej").addEventListener("click", () => setDraft("rejected"));
  $("#saveReview").addEventListener("click", saveReview);
  $("#filterSel").addEventListener("change", renderTable);
  $("#sortSel").addEventListener("change", renderTable);
  $("#noticeClose").addEventListener("click", () => { $("#sessionNotice").hidden = true; sessionStorageSet("noticeDismissed", "1"); });
  ["#dlCsv", "#dlZip"].forEach(id => $(id).addEventListener("click", () => {
    if (S.draft) toast("Note: the unsaved decision is not in the download. Save it first.");
    S.downloaded = true;
  }));
  window.addEventListener("resize", () => { if (S.selected && S.windows[S.selected]) { drawWaveform(S.windows[S.selected]); drawSpectrogram(S.windows[S.selected]); } });
  window.addEventListener("beforeunload", e => {
    if (S.draft || busy() || (S.result && S.events.length && !S.downloaded)) { e.preventDefault(); e.returnValue = ""; }
  });
});
