// Jobs page: job list, job result (tiles, laps and clips matrices), build log, compare and the new job panel.
window.JOBS_PAGE = true;
const { h } = U;
const S = { jobs: [], sel: null, checked: [], meta: { recipes: [], tiers: null, conditions: [], default_ref: "" }, detail: null, tab: "laps", clipCell: null, mode: null,
  timer: null, tier: "smoke", where: "local", paramsTouched: false, cells: {}, slice: {}, armed: null, armTimer: null, f: { q: "", tier: "", where: "" } };
const main = document.getElementById("main");
const ACTIVE = ["building", "running"];
const LINES = ["center", "line1", "line3"];
const CLASSES = ["offset_aligned", "misaligned_on_line", "oscillation", "recoverable_failure"];
const TRASH = '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5h6.6L12 4M6.8 6.5v4.5M9.2 6.5v4.5"/></svg>';

// ---------- list ----------
function rowStatus(j) {
  if (j.status === "running" || j.status === "building")
    return h("span", { class: "when eta", text: j.eta_s ? "ETA " + U.dur(j.eta_s) : j.status });
  if (j.status === "build_failed") return h("span", { class: "when bad", text: "build failed" });
  if (j.status === "failed") return h("span", { class: "when bad", text: "failed" });
  if (j.status === "queued") return h("span", { class: "when", text: "queued" });
  return h("span", { class: "when", text: U.rel(j.finished_at || j.submitted_at) });
}

function canCheck(j) { return !["build_failed", "queued", "failed"].includes(j.status); }

function visible(j) {
  const q = S.f.q.trim().toLowerCase();
  return (!q || (j.recipe + " " + j.user + " " + j.notes + " " + j.tier).toLowerCase().includes(q)) && (!S.f.tier || j.tier === S.f.tier) && (!S.f.where || j.where === S.f.where);
}

function renderList() {
  const sel = document.getElementById("fwhere");
  if (sel) { for (const w of new Set(S.jobs.map((j) => j.where))) if (![...sel.options].some((o) => o.value === w)) sel.append(h("option", { value: w, text: w.replace("-", " ") }));
  sel.value = S.f.where; } // absent while a server still caches the old jobs.html
  for (const b of document.querySelectorAll("#ftier button")) b.classList.toggle("on", b.dataset.v === S.f.tier);
  const rows = document.getElementById("rows");
  rows.replaceChildren();
  for (const j of S.jobs) {
    if (!visible(j)) continue;
    const on = S.checked.includes(j.id);
    const box = h("input", { type: "checkbox", "aria-label": "Select " + j.recipe, checked: on,
      disabled: !canCheck(j) || (!on && S.checked.length >= 5) });
    box.addEventListener("click", (e) => { e.stopPropagation(); toggleCheck(j.id); });
    const row = h("div", { class: "row" + (S.sel === j.id && S.mode !== "compare" ? " sel" : ""), "data-id": j.id,
      onclick: () => select(j.id) },
      h("span", { class: "name", text: j.recipe, title: j.recipe }),
      h("span", { class: "tag " + j.tier, text: j.tier }),
      h("label", { class: "chk" + (on ? " on" : ""), onclick: (e) => e.stopPropagation() }, box),
      h("span", { class: "sub", text: j.user }),
      rowStatus(j));
    if (ACTIVE.includes(j.status)) {
      const p = j.total ? Math.max(2, (100 * j.done) / j.total) : 2;
      row.append(h("div", { class: "bar" }, h("i", { style: `width:${p}%` })));
    }
    rows.append(row);
  }
  const n = S.checked.length;
  document.getElementById("filter").hidden = n >= 2;
  const cmp = document.getElementById("compare");
  cmp.hidden = n < 2;
  cmp.textContent = "Compare " + n;
  document.getElementById("clear").hidden = n < 2;
}

function toggleCheck(id) {
  const i = S.checked.indexOf(id);
  if (i >= 0) S.checked.splice(i, 1);
  else if (S.checked.length < 5) S.checked.push(id);
  if (S.mode === "compare") {
    if (S.checked.length >= 2) openCompare();
    else if (S.sel) select(S.sel);
  }
  renderList();
}

async function poll() {
  clearTimeout(S.timer);
  try {
    const r = await U.api("/api/jobs");
    if (r.ok) {
      const prev = Object.fromEntries(S.jobs.map((j) => [j.id, j.status + j.done]));
      S.jobs = r.body;
      renderList();
      const cur = S.jobs.find((j) => j.id === S.sel);
      if (cur && S.mode === "detail" && prev[cur.id] !== undefined && prev[cur.id] !== cur.status + cur.done) select(cur.id, true);
    }
  } finally {
    const busy = S.jobs.some((j) => ACTIVE.includes(j.status) || j.status === "queued");
    S.timer = setTimeout(poll, busy ? 2000 : 15000);
  }
}

// ---------- main panel ----------
function setUrl(params) {
  const u = new URL(location.href);
  for (const k of U.SLICE) if (S.slice[k]) params[k] = S.slice[k];
  if (u.searchParams.get("color")) params.color = u.searchParams.get("color");
  for (const k of ["q", "tier", "where"]) if (S.f[k]) params[k] = S.f[k];
  u.search = new URLSearchParams(params).toString();
  history.replaceState(null, "", u);
}

function emptyMain() {
  S.mode = "empty";
  main.replaceChildren(h("div", { class: "empty" }, h("button", { class: "btn gold", type: "button", text: "New job", onclick: openPanel })));
}

function headerFor(job, progress) {
  const ref = job.recipe.source.ref, commit = job.recipe.source.commit;
  const meta = h("span", { class: "meta" }, h("span", { text: !commit ? U.shortRef(ref) : commit.startsWith(ref) ? commit.slice(0, 7) : ref + " @ " + commit.slice(0, 7), title: commit ? ref + " @ " + commit : ref }), h("span", { text: job.user }),
    h("span", { text: U.date(job.started_at || job.submitted_at) }));
  if (job.finished_at && job.started_at) meta.append(h("span", { text: U.dur((new Date(job.finished_at) - new Date(job.started_at)) / 1000) }));
  if (ACTIVE.includes(job.status) && progress.total) {
    meta.append(h("span", { text: `${progress.done} / ${progress.total}` }));
    if (progress.eta_s) meta.append(h("span", { text: "ETA " + U.dur(progress.eta_s) }));
  }
  if (job.status === "build_failed") meta.append(h("span", { class: "status-bad", text: "build failed" }));
  if (job.status === "failed") meta.append(h("span", { class: "status-bad", text: "failed" }));
  if (job.status === "queued") meta.append(h("span", { text: "queued" }));
  if (job.errors > 0) meta.append(h("span", { class: "status-bad", text: job.errors + " err" }));
  const out = [h("div", { class: "head" }, h("h1", { text: job.recipe.name }), h("span", { class: "tag " + job.tier, text: job.tier }), meta, delButton(job.id))];
  if (job.notes) out.push(h("p", { class: "notes", text: job.notes }));
  if (ACTIVE.includes(job.status) && progress.total) out.push(h("div", { class: "prog" }, h("i", { style: `width:${(100 * progress.done) / progress.total}%` })));
  return out;
}

function delButton(id) {
  if (S.armed === id) return h("button", { class: "btn red del", type: "button", text: "Delete", onclick: () => removeJob(id) });
  const b = h("button", { class: "x del", type: "button", "aria-label": "Delete job", onclick: () => {
    S.armed = id;
    clearTimeout(S.armTimer);
    S.armTimer = setTimeout(() => { S.armed = null; const d = document.querySelector(".head .del"); if (d) d.replaceWith(delButton(id)); }, 4000);
    b.replaceWith(delButton(id));
  } });
  b.innerHTML = TRASH;
  return b;
}

async function removeJob(id) {
  const d = document.querySelector(".head .del");
  if (d) d.disabled = true;
  const r = await U.api("/api/jobs/" + encodeURIComponent(id), { method: "DELETE" });
  clearTimeout(S.armTimer);
  S.armed = null;
  if (!r.ok) { const e = document.querySelector(".head .del"); if (e) e.replaceWith(delButton(id)); return; }
  S.checked = S.checked.filter((x) => x !== id);
  if (S.sel === id) { S.sel = null; setUrl({}); emptyMain(); }
  await poll();
}

function tile(label, valueNodes, worst, cls) {
  const cap = [S.slice.kart === "worst" ? U.kart(worst) : "", U.sliceText(S.slice, S.meta.conditions)].filter(Boolean).join(" · ");
  return h("div", { class: "tile" + (cls ? " " + cls : "") }, h("div", { class: "l", text: label }), h("div", { class: "v" }, valueNodes),
    h("div", { class: "k", text: cap || " " }));
}

function renderFilters(opts) {
  return U.filters(opts, S.slice, S.meta.conditions, (k, v) => { S.slice = Object.assign({}, S.slice, { [k]: v }); if (S.mode === "compare") openCompare(); else select(S.sel); });
}

function renderTiles(hd) {
  const out = h("div", { class: "tiles" });
  const val = (m) => hd[m] && hd[m].value !== null && hd[m].value !== undefined;
  for (const m of ["cte_p50", "cte_p90", "cte_p99", "recovery_rate", "recovery_s_median", "safety", "osc", "laps_pct", "loc_rms"]) {
    if (m === "osc") {
      if (!val("osc_10")) continue;
      const parts = [];
      ["osc_10", "osc_20", "osc_30"].forEach((k, i) => {
        if (i) parts.push(h("span", { class: "sep", text: "/" }));
        parts.push(U.METRICS[k].fmt(hd[k].value));
      });
      parts.push(h("small", { text: "/min" }));
      out.append(tile("Osc > 10 / 20 / 30", parts, hd.osc_10.worst_kart, "osc"));
      continue;
    }
    if (!val(m)) continue;
    const d = U.METRICS[m];
    const el = tile(d.label, [d.fmt(hd[m].value), h("small", { text: d.unit })], hd[m].worst_kart);
    if (m === "safety") U.tip(el, U.unsafeText((k) => [hd[k] && hd[k].value]));
    out.append(el);
  }
  return out;
}

function scaleColor(v) {
  const t = Math.max(0, Math.min(1, v / 0.5));
  const mix = (a, b, f) => a.map((c, i) => Math.round(c + (b[i] - c) * f));
  const W = [255, 255, 255], G = [207, 185, 145], B = [0, 0, 0];
  const c = t < 0.5 ? mix(W, G, t / 0.5) : mix(G, B, (t - 0.5) / 0.5);
  return { bg: `rgb(${c.join(",")})`, fg: t > 0.62 ? "#FFFFFF" : "#1A1A1A" };
}

function cteText(v) {
  if (v === null || v === undefined) return "";
  return v < 0.01 ? "0.00" : v < 1 ? v.toPrecision(2) : v.toFixed(1);
}

function condInfo(name) {
  return (S.meta.conditions.find((c) => c.name === name)) || { name, group: U.human(name), level: "" };
}

function renderLaps(b, makeCell = lapCell) {
  const laps = b.laps;
  const conds = S.meta.conditions.map((c) => c.name).filter((c) => laps[c]).concat(Object.keys(laps).filter((c) => !S.meta.conditions.some((m) => m.name === c)));
  const karts = [...new Set(conds.flatMap((c) => Object.keys(laps[c])))].sort();
  const lines = LINES.filter((l) => conds.some((c) => karts.some((k) => laps[c][k] && laps[c][k][l])));
  const cols = ["auto", "auto"];
  karts.forEach((k, i) => { if (i) cols.push("12px"); lines.forEach(() => cols.push("44px")); });
  const g = h("div", { class: "lm", style: `grid-template-columns:${cols.join(" ")}` });
  g.append(h("div"), h("div"));
  karts.forEach((k, i) => { if (i) g.append(h("div")); g.append(h("div", { class: "kh", style: `grid-column:span ${lines.length}`, text: U.kart(k) })); });
  g.append(h("div"), h("div"));
  karts.forEach((k, i) => { if (i) g.append(h("div")); lines.forEach((l) => g.append(h("div", { class: "lh", text: l }))); });
  let lastGroup = null;
  for (const c of conds) {
    const info = condInfo(c);
    if (lastGroup !== null && info.group !== lastGroup) g.append(h("div", { class: "brk" }));
    g.append(h("div", { class: "g", style: "padding-right:8px", text: info.group === lastGroup ? "" : info.group }),
      h("div", { class: "lv", style: "padding-right:12px", text: info.level }));
    lastGroup = info.group;
    karts.forEach((k, i) => {
      if (i) g.append(h("div"));
      for (const l of lines) {
        const cell = laps[c][k] && laps[c][k][l];
        if (!cell) { g.append(h("div", { class: "cell none" })); continue; }
        const el = makeCell(cell, c, k, l);
        g.append(el);
      }
    });
  }
  return g;
}

function lapCell(cell) {
  let el;
  if (cell.status !== "ok") {
    const cause = cell.status === "error" ? "error" : cell.cause;
    el = h("div", { class: "cell fail", text: U.CAUSE[cause] || "fail" });
    U.tip(el, U.human(cause || "failed") + (cell.t !== null && cell.t !== undefined ? ` at ${cell.t.toFixed(1)} s` : ""));
  } else {
    const col = scaleColor(cell.cte_p90 || 0);
    el = h("div", { class: "cell", style: `background:${col.bg};color:${col.fg}`, text: cteText(cell.cte_p90) });
    U.tip(el, `${U.num(cell.cte_p90, 2)} m`);
  }
  el.dataset.key = cell.key;
  el.addEventListener("click", () => Viewer.open(cell.key));
  S.cells[cell.key] = el;
  return el;
}

function renderClips(b) {
  const wrap = h("div");
  const clips = b.clips;
  const classes = CLASSES.filter((c) => clips[c]).concat(Object.keys(clips).filter((c) => !CLASSES.includes(c)));
  const karts = [...new Set(classes.flatMap((c) => Object.keys(clips[c])))].sort();
  const g = h("div", { class: "cm" });
  g.append(h("div"));
  karts.forEach((k) => g.append(h("div", { class: "kh", text: U.kart(k) })));
  const list = h("div");
  for (const c of classes) {
    g.append(h("div", { class: "g", text: U.human(c) }));
    for (const k of karts) {
      const cell = clips[c][k];
      if (!cell || cell.recovery_rate === null) { g.append(h("div", { class: "ccell na", text: "–" })); continue; }
      const on = S.clipCell && S.clipCell[0] === c && S.clipCell[1] === k;
      const el = h("div", { class: "ccell" + (on ? " on" : ""), "data-cls": c, "data-kart": k },
        h("div", { class: "r", text: U.pct(cell.recovery_rate) + "%" }),
        h("div", { class: "m", text: cell.median_recovery_s !== null ? U.num(cell.median_recovery_s, 1) + " s" : " " }),
        h("i", { class: "fill", style: `width:${100 * cell.recovery_rate}%` }));
      el.addEventListener("click", () => {
        S.clipCell = on ? null : [c, k];
        wrap.replaceWith(renderClips(b));
        markCell(Viewer.current);
      });
      g.append(el);
    }
  }
  wrap.append(g);
  if (S.clipCell && clips[S.clipCell[0]] && clips[S.clipCell[0]][S.clipCell[1]]) {
    const eps = [...clips[S.clipCell[0]][S.clipCell[1]].episodes].sort((a, b) => a.clip_id.localeCompare(b.clip_id));
    const tb = h("tbody");
    for (const e of eps) {
      const inv = e.context_valid === false;
      const tr = h("tr", { class: inv ? "inv" : "", "data-key": e.key, onclick: () => Viewer.open(e.key) },
        h("td", { text: e.clip_id }),
        h("td", { class: inv || e.recovered ? "" : "no", text: inv ? "invalid" : e.recovered ? "yes" : "no" }),
        h("td", { class: "num", text: e.recovery_s !== null && e.recovery_s !== undefined ? U.num(e.recovery_s, 1) + " s" : "" }),
        h("td", { class: "num", text: e.peak_cte_after_m !== null && e.peak_cte_after_m !== undefined ? U.num(e.peak_cte_after_m, 2) + " m" : "" }));
      S.cells[e.key] = tr;
      tb.append(tr);
    }
    wrap.append(h("table", { class: "clist" }, h("thead", {}, h("tr", {}, h("th", { text: "Clip" }), h("th", { text: "Recovered" }),
      h("th", { class: "num", text: "Time" }), h("th", { class: "num", text: "Peak CTE" }))), tb));
  }
  markCell(Viewer.current);
  return wrap;
}

function markCell(key) {
  for (const el of document.querySelectorAll(".cell.on, .clist tr.on")) el.classList.remove("on");
  if (key && S.cells[key] && S.cells[key].isConnected) S.cells[key].classList.add("on");
}

function renderBreak() {
  const b = S.detail.breakdown;
  const box = document.getElementById("d-break");
  S.cells = {};
  const hasLaps = Object.keys(b.laps).length > 0, hasClips = Object.keys(b.clips).length > 0;
  if (!hasLaps && !hasClips) { box.replaceChildren(); return; }
  if (S.tab === "clips" && !hasClips) S.tab = "laps";
  if (S.tab === "laps" && !hasLaps) S.tab = "clips";
  const tabs = h("div", { class: "tabs", role: "tablist" });
  for (const [t, label, ok] of [["laps", "Laps", hasLaps], ["clips", "Clips", hasClips]]) {
    if (!ok) continue;
    tabs.append(h("button", { type: "button", role: "tab", class: S.tab === t ? "on" : "", "data-tab": t, text: label,
      onclick: () => { S.tab = t; renderBreak(); } }));
  }
  box.replaceChildren(tabs, S.tab === "laps" ? renderLaps(b) : renderClips(b));
  markCell(Viewer.current);
}

async function select(id, refresh) {
  const j = S.jobs.find((x) => x.id === id);
  if (!refresh) { S.sel = id; S.clipCell = null; }
  S.mode = "detail";
  setUrl({ job: id });
  renderList();
  const q = U.sliceQuery(S.slice);
  const r = await U.api("/api/jobs/" + encodeURIComponent(id) + "?" + q.slice(1));
  if (!r.ok || S.sel !== id || S.mode !== "detail" || q !== U.sliceQuery(S.slice)) return;
  S.detail = r.body;
  const job = r.body.job;
  if (job.status === "build_failed" || (job.status === "failed" && job.build_log)) {
    const log = await U.api(`/api/jobs/${encodeURIComponent(id)}/build_log`);
    main.replaceChildren(h("div", { class: "pad" }, headerFor(job, r.body.progress), h("pre", { class: "log", id: "build-log", text: log.ok ? log.body : "" })));
    return;
  }
  if (refresh && document.getElementById("d-head")) {
    document.getElementById("d-head").replaceChildren(...headerFor(job, r.body.progress));
    document.getElementById("d-filt").replaceChildren(renderFilters(r.body.options));
    document.getElementById("d-tiles").replaceChildren(renderTiles(r.body.headline));
    renderBreak();
    Viewer.update(r.body.breakdown);
    return;
  }
  const pad = h("div", { class: "pad" }, h("div", { id: "d-head" }, headerFor(job, r.body.progress)),
    h("div", { id: "d-filt" }, renderFilters(r.body.options)), h("div", { id: "d-tiles" }, renderTiles(r.body.headline)), h("div", { id: "d-break" }), h("div", { id: "d-viewer" }));
  main.replaceChildren(pad);
  main.scrollTop = 0;
  Viewer.mount(document.getElementById("d-viewer"), [{ id, color: U.COLORS[0] }], r.body.breakdown, S.meta.conditions, markCell);
  renderBreak();
  Viewer.openDefault();
}

function openCompare() {
  S.mode = "compare";
  setUrl({ compare: S.checked.join(",") });
  renderList();
  Compare.render(main, [...S.checked], S.meta.conditions, S.slice, renderFilters);
}

// ---------- new job panel ----------
const panel = document.getElementById("panel");
const F = (id) => document.getElementById(id);

function fieldError(name, msg) {
  const f = panel.querySelector(`.field[data-f="${name}"]`);
  if (!f) return;
  f.classList.toggle("err", !!msg);
  let e = f.querySelector(".e");
  if (msg) { if (!e) { e = h("div", { class: "e" }); f.append(e); } e.textContent = msg; }
  else if (e) e.remove();
}

function checkParams() {
  const t = F("f-params").value.trim() || "{}";
  try {
    const v = JSON.parse(t);
    if (typeof v !== "object" || v === null || Array.isArray(v)) throw new SyntaxError("Must be a JSON object");
    fieldError("params", null);
    return v;
  } catch (e) {
    fieldError("params", e.message.replace(/^JSON\.parse: /, ""));
    return undefined;
  }
}

function setSeg(id, v) {
  for (const b of F(id).querySelectorAll("button")) b.classList.toggle("on", b.dataset.v === v);
}

function updateEst() {
  const modal = F("f-where").querySelector('[value="modal"]');
  modal.disabled = S.tier !== "smoke";
  if (S.tier !== "smoke" && S.where === "modal") S.where = "local";
  const mgpu = F("f-where").querySelector('[value="modal-gpu"]');
  mgpu.disabled = S.tier === "large";
  if (S.tier === "large" && S.where === "modal-gpu") S.where = "local";
  setSeg("f-tier", S.tier);
  F("f-where").value = S.where;
  const t = S.meta.tiers && S.meta.tiers[S.tier];
  if (!t || t.episodes === null) { F("est-n").textContent = "\u2013 episodes"; F("est-c").textContent = ""; return; }
  F("est-n").textContent = t.episodes + " episodes";
  F("est-c").textContent = S.where === "modal" ? "$" + Math.max(t.cost_usd, 0.01).toFixed(2) : "$0";
}

function openPanel() {
  panel.hidden = false;
  if (!F("f-ref").value) F("f-ref").value = S.meta.default_ref;
  F("f-ref").scrollLeft = F("f-ref").scrollWidth;
  try { if (!F("f-user").value) F("f-user").value = localStorage.getItem("evc.user") || ""; } catch (e) { /* storage blocked */ }
  updateEst();
  F("f-recipe").focus();
}
function closePanel() { panel.hidden = true; }

async function submit() {
  const params = checkParams();
  const body = { recipe_name: F("f-recipe").value, ref: F("f-ref").value, params: params === undefined ? F("f-params").value : params,
    tier: S.tier, where: S.where, user: F("f-user").value, notes: F("f-notes").value };
  const rec = S.meta.recipes.find((x) => x.name === body.recipe_name);
  if (rec && rec.sim) body.sim = rec.sim;   // sim options (CUDA, canaries) come only from a config recipe
  for (const f of ["recipe_name", "ref", "user", "tier", "where"]) fieldError(f, null);
  const r = await U.api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    for (const [f, m] of Object.entries((r.body && r.body.errors) || {})) fieldError(f, m);
    return;
  }
  try { localStorage.setItem("evc.user", body.user); } catch (e) { /* storage blocked */ }
  F("f-recipe").value = ""; F("f-notes").value = ""; F("f-params").value = "{}"; S.paramsTouched = false;
  closePanel();
  await poll();
  if (S.mode === "empty") select(r.body.id);
}

function wirePanel() {
  document.getElementById("new-job").addEventListener("click", openPanel);
  F("panel-close").addEventListener("click", closePanel);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) closePanel(); });
  F("f-params").addEventListener("input", () => { S.paramsTouched = true; checkParams(); });
  F("f-ref").addEventListener("blur", () => { F("f-ref").scrollLeft = F("f-ref").scrollWidth; });
  F("f-recipe").addEventListener("input", () => {
    fieldError("recipe_name", null);
    const r = S.meta.recipes.find((x) => x.name === F("f-recipe").value);
    if (r && !S.paramsTouched) {
      F("f-params").value = JSON.stringify(r.params, null, 2);
      F("f-ref").value = r.source.ref;
      F("f-ref").scrollLeft = F("f-ref").scrollWidth;
      if (!F("f-notes").value) F("f-notes").value = r.notes;
      checkParams();
    }
  });
  for (const id of ["f-ref", "f-user"]) F(id).addEventListener("input", () => fieldError(id === "f-ref" ? "ref" : "user", null));
  F("f-tier").addEventListener("click", (e) => { if (e.target.dataset.v) { S.tier = e.target.dataset.v; updateEst(); } });
  F("f-where").addEventListener("change", () => { S.where = F("f-where").value; fieldError("where", null); updateEst(); });
  F("submit").addEventListener("click", submit);
  F("form").addEventListener("submit", (e) => { e.preventDefault(); submit(); });
}

// ---------- boot ----------
(async function boot() {
  wirePanel();
  const qs0 = new URLSearchParams(location.search);
  for (const k of ["q", "tier", "where"]) S.f[k] = qs0.get(k) || "";
  document.getElementById("filter").value = S.f.q;
  const refilter = () => {
    const vis = S.jobs.filter(visible);
    if (S.mode !== "compare" && !vis.some((j) => j.id === S.sel)) {
      const next = vis.find((j) => j.status === "done") || vis[0];
      if (next) select(next.id); else { S.sel = null; emptyMain(); }
    }
    renderList(); setUrl(S.mode === "compare" ? { compare: S.checked.join(",") } : S.sel ? { job: S.sel } : {}); };
  document.getElementById("filter").addEventListener("input", (e) => { S.f.q = e.target.value; refilter(); });
  document.getElementById("ftier")?.addEventListener("click", (e) => { if (e.target.dataset.v !== undefined) { S.f.tier = e.target.dataset.v; refilter(); } });
  document.getElementById("fwhere")?.addEventListener("change", (e) => { S.f.where = e.target.value; refilter(); });
  document.getElementById("compare").addEventListener("click", openCompare);
  document.getElementById("clear").addEventListener("click", () => { S.checked = []; if (S.mode === "compare" && S.sel) select(S.sel); else if (S.mode === "compare") emptyMain(); renderList(); });
  const [meta] = await Promise.all([U.api("/api/meta"), poll()]);
  S.meta = meta.ok ? meta.body : { recipes: [], tiers: null, conditions: [], default_ref: "" };
  F("recipes").replaceChildren(...S.meta.recipes.map((r) => h("option", { value: r.name })));
  F("refs").replaceChildren(...[S.meta.default_ref, ...(S.meta.refs || [])].filter(Boolean).map((r) => h("option", { value: r })));
  updateEst();
  const qs = new URLSearchParams(location.search);
  for (const k of U.SLICE) if (qs.get(k)) S.slice[k] = qs.get(k);
  if (qs.get("compare")) {
    S.checked = qs.get("compare").split(",").filter((id) => S.jobs.some((j) => j.id === id)).slice(0, 5);
    if (S.checked.length >= 2) openCompare();
  }
  if (S.mode !== "compare") {
    const want = qs.get("job");
    const vis = S.jobs.filter(visible);
    const first = vis.find((j) => j.status === "done") || vis[0];
    if (want && S.jobs.some((j) => j.id === want)) select(want);
    else if (first) select(first.id);
    else emptyMain();
  }
  if (qs.get("new")) { openPanel(); setUrl(S.sel ? { job: S.sel } : {}); }
})();
