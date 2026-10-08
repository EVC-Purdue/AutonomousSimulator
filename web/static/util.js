// Shared helpers: element builder, formatting, metric definitions, tooltip.
const U = (() => {
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style") el.style.cssText = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "text") el.textContent = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) el.append(c.nodeType ? c : String(c));
    return el;
  }
  const num = (v, d) => (v === null || v === undefined ? "" : Number(v).toFixed(d));
  const pct = (v) => (v === null || v === undefined ? "" : String(Math.round(v * 100)));
  const METRICS = {
    cte_p50: { label: "CTE p50", unit: "m", fmt: (v) => num(v, 2), dp: 2, low: true },
    cte_p90: { label: "CTE p90", unit: "m", fmt: (v) => num(v, 2), dp: 2, low: true },
    cte_p99: { label: "CTE p99", unit: "m", fmt: (v) => num(v, 2), dp: 2, low: true },
    recovery_rate: { label: "Recovery", unit: "%", fmt: pct, scale: 100, dp: 0, low: false },
    recovery_s_median: { label: "Recovery time", unit: "s", fmt: (v) => num(v, 1), dp: 1, low: true },
    safety: { label: "Safety", unit: "%", fmt: pct, scale: 100, dp: 0, low: false },
    osc_10: { label: "Osc > 10", unit: "/min", fmt: (v) => num(v, 1), dp: 1, low: true },
    osc_20: { label: "Osc > 20", unit: "/min", fmt: (v) => num(v, 1), dp: 1, low: true },
    osc_30: { label: "Osc > 30", unit: "/min", fmt: (v) => num(v, 1), dp: 1, low: true },
    laps_pct: { label: "Laps", unit: "%", fmt: (v) => num(v, 0), dp: 0, low: false },
    loc_rms: { label: "Loc RMS", unit: "m", fmt: (v) => num(v, 2), dp: 2, low: true },
  };
  const UNSAFE = [["unsafe_lat", "lateral"], ["unsafe_rate", "rate limit"], ["unsafe_travel", "travel limit"], ["unsafe_spin", "spin"], ["unsafe_off", "off track"]];
  // safety breakdown tooltip: % of autonomous time per criterion, worst first; get(key) gives one value per job
  function unsafeText(get) {
    const rows = UNSAFE.map(([k, n]) => [n, get(k) || []]).filter(([, v]) => v.some((x) => x !== null && x !== undefined));
    rows.sort((a, b) => Math.max(...b[1]) - Math.max(...a[1]));
    return rows.map(([n, v]) => n.padEnd(13) + v.map((x) => (x === null || x === undefined ? "" : (100 * x).toFixed(1) + " %").padStart(8)).join("")).join("\n");
  }
  function rel(iso) {
    if (!iso) return "";
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 60) return "now";
    if (s < 3600) return Math.floor(s / 60) + " min";
    if (s < 86400) return Math.floor(s / 3600) + " h";
    if (s < 7 * 86400) return Math.floor(s / 86400) + " d";
    return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }
  function dur(s) {
    if (s === null || s === undefined) return "";
    if (s < 60) return Math.round(s) + " s";
    const m = Math.round(s / 60);
    return m < 60 ? m + " min" : Math.floor(m / 60) + " h " + (m % 60) + " min";
  }
  const date = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "");
  const shortRef = (ref) => (ref.startsWith("worktree:") ? ref.split("/").filter(Boolean).pop() : ref.slice(0, 7));
  const kart = (k) => (k ? k.replace("kart", "kart ") : "");
  const human = (s) => s.replace(/_/g, " ");
  const CAUSE = { spin: "spin", off_track: "off", localization_lost: "loc", no_commands: "cmd", error: "err", time_cap: "time" };
  const tipEl = () => document.getElementById("tip");
  function tip(el, text) {
    el.addEventListener("mouseenter", () => { const s = typeof text === "function" ? text() : text; if (!s) return; const t = tipEl(); t.textContent = s; t.hidden = false; });
    el.addEventListener("mousemove", (e) => { const t = tipEl(); t.style.left = e.clientX + 12 + "px"; t.style.top = e.clientY + 14 + "px"; });
    el.addEventListener("mouseleave", () => { tipEl().hidden = true; });
  }
  async function api(url, opts) {
    try {
      const r = await fetch(url, opts);
      const body = (r.headers.get("content-type") || "").includes("json") ? await r.json() : await r.text();
      return { ok: r.ok, status: r.status, body };
    } catch (e) {
      return { ok: false, status: 0, body: null };
    }
  }
  const COLORS = ["#000000", "#CFB991", "#4A6FA5", "#2E8B7A", "#B5651D"];
  const SLICE = ["kart", "line", "condition", "cls", "cap"];
  const sliceQuery = (sl) => SLICE.filter((k) => sl[k]).map((k) => `&${k}=${encodeURIComponent(sl[k])}`).join("");
  const condLabel = (conds, n) => { const c = conds.find((x) => x.name === n); return c ? (c.level ? `${c.group} ${c.level}` : c.group) : human(n); };
  function sliceText(sl, conds) {
    return [sl.kart && sl.kart !== "worst" ? kart(sl.kart) : "", sl.line || "", sl.condition ? condLabel(conds, sl.condition) : "",
      sl.cls ? human(sl.cls) : "", sl.cap ? `≤${sl.cap} m/s` : ""].filter(Boolean).join(" · ");
  }
  function filters(opts, sl, conds, onChange) {
    const order = { condition: conds.map((c) => c.name), cls: ["offset_aligned", "misaligned_on_line", "oscillation", "recoverable_failure"] };
    const all = { kart: "All karts", line: "All lines", condition: "All conditions", cls: "All classes", cap: "All speeds" };
    const label = { kart: (v) => (v === "worst" ? "Worst kart" : kart(v)), line: (v) => v, condition: (v) => condLabel(conds, v), cls: human, cap: (v) => `≤${v} m/s` };
    const bar = h("div", { class: "filt" });
    for (const k of SLICE) {
      const o = opts[k] || {};
      const rank = (v) => (order[k] && order[k].includes(v) ? order[k].indexOf(v) : 999);
      const vals = Object.keys(o).sort((a, b) => rank(a) - rank(b) || (k === "cap" ? a - b : 0));
      if (k === "cap" && !vals.length && !sl[k]) continue;   // only speed-capped jobs show the speed filter
      if (sl[k] && !vals.includes(sl[k])) vals.push(sl[k]);
      const sel = h("select", { "aria-label": all[k], "data-f": k, class: sl[k] ? "on" : "" }, h("option", { value: "", text: all[k] }),
        vals.map((v) => h("option", { value: v, text: label[k](v), disabled: o[v] === false && sl[k] !== v })));
      sel.value = sl[k] || "";
      sel.addEventListener("change", () => onChange(k, sel.value));
      bar.append(sel);
    }
    return bar;
  }
  // lap columns: known lines in order, then any others (speed caps as line@cap) in natural order
  const lineKeys = (known, present) => { const p = new Set(present); return known.filter((l) => p.has(l)).concat([...p].filter((l) => !known.includes(l)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))); };
  return { lineKeys, h, num, pct, METRICS, unsafeText, rel, dur, date, shortRef, kart, human, CAUSE, tip, api, COLORS, SLICE, sliceQuery, sliceText, filters };
})();

document.getElementById("new-job").addEventListener("click", () => {
  if (!window.JOBS_PAGE) location.href = "/?new=1";
});
