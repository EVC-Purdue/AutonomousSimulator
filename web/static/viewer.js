// Episode viewer: kart, condition and line or clip selects; track map; cursor-synced uPlot charts and unsafe strip; one or more jobs overlaid.
const Viewer = (() => {
  const { h } = U;
  const NS = "http://www.w3.org/2000/svg";
  const LINES = ["center", "line1", "line4"];
  const V = { el: null, jobs: [], multi: false, idx: null, conds: [], onOpen: null, eps: [], ep: null, T: [], M: [], plots: [], proj: null, dots: [], ro: null };
  const api = { current: null, keys: [] };
  const COLOR_BY = {
    speed: { label: "Speed", unit: "m/s", f: (L) => L.speed },
    cte_abs: { label: "|CTE|", unit: "m", f: (L) => L.cte && L.cte.map(Math.abs) },
    cte: { label: "CTE", unit: "m", signed: true, f: (L) => L.cte },
    wheel: { label: "Steering", unit: "deg", signed: true, f: (L) => L.wheel },
    yaw_rate: { label: "Yaw rate", unit: "deg/s", signed: true, f: (L) => L.yaw_rate },
    ay: { label: "Lat accel", unit: "m/s²", signed: true, f: (L) => L.ay_mps2 },
    slide: { label: "Slide", unit: "% of peak", f: (L) => L.slide_pct },
    loc_err: { label: "Loc error", unit: "m", f: (L) => L.loc_err },
    cost: { label: "Cost", unit: "cost", stops: "COST", f: (L) => L.cost },
    time: { label: "Time", unit: "s", f: (L) => L.t },
  };
  const SEQ = [[234, 223, 198], [207, 185, 145], [0, 0, 0]], DIV = [[74, 111, 165], [178, 180, 186], [181, 101, 29]];
  const COST = [[60, 110, 180], [230, 160, 40], [180, 30, 30]];   // MPC cost: low blue, high red
  const BINS = 24;

  const cells = (v) => (Array.isArray(v) ? v : [v]);

  function index(bd) {
    const n = V.jobs.length;
    const laps = {}, clips = {}, classes = {}, byKey = {}, karts = new Set(), condSet = new Set(), lineSet = new Set();
    for (const [c, ks] of Object.entries(bd.laps)) {
      for (const [k, ls] of Object.entries(ks)) {
        for (const [l, cell] of Object.entries(ls)) {
          const keys = cells(cell).map((x) => (x ? x.key : null));
          ((laps[c] ||= {})[k] ||= {})[l] = keys;
          karts.add(k); condSet.add(c); lineSet.add(l);
          for (const key of keys) if (key) byKey[key] = { kart: k, cond: c, item: "L:" + l };
        }
      }
    }
    for (const [cls, ks] of Object.entries(bd.clips)) {
      for (const [k, cell] of Object.entries(ks)) {
        cells(cell).forEach((x, i) => {
          if (!x) return;
          for (const e of x.episodes) {
            const keys = ((clips[e.clip_id] ||= {})[k] ||= Array(n).fill(null));
            keys[i] = e.key;
            classes[e.clip_id] = cls;
            karts.add(k); condSet.add("nominal");
            byKey[e.key] = { kart: k, cond: "nominal", item: "C:" + e.clip_id };
          }
        });
      }
    }
    return { laps, clips, classes, byKey, karts: [...karts].sort(), conds: condSet, lines: U.lineKeys(LINES, lineSet) };
  }

  function keyFor(kart, cond, item) {
    if (item.startsWith("L:")) return ((V.idx.laps[cond] || {})[kart] || {})[item.slice(2)] || null;
    return cond === "nominal" ? (V.idx.clips[item.slice(2)] || {})[kart] || null : null;
  }

  function colorParam() {
    const c = new URLSearchParams(location.search).get("color");
    return COLOR_BY[c] ? c : "";
  }

  function buildSelects() {
    const bar = h("div", { class: "vbar" });
    const sk = h("select", { "aria-label": "Kart", id: "v-kart" }, V.idx.karts.map((k) => h("option", { value: k, text: U.kart(k) })));
    const sc = h("select", { "aria-label": "Condition", id: "v-cond" },
      V.conds.filter((c) => V.idx.conds.has(c.name)).concat([...V.idx.conds].filter((n) => !V.conds.some((c) => c.name === n)).map((n) => ({ name: n, group: U.human(n), level: "" }))).map((c) => h("option", { value: c.name, text: c.level ? `${c.group} ${c.level}` : c.group })));
    const si = h("select", { "aria-label": "Line or clip", id: "v-item" });
    if (V.idx.lines.length) si.append(h("optgroup", { label: "Lines" }, V.idx.lines.map((l) => h("option", { value: "L:" + l, text: l }))));
    const byCls = {};
    for (const id of Object.keys(V.idx.clips).sort()) (byCls[V.idx.classes[id]] ||= []).push(id);
    for (const cls of Object.keys(byCls)) si.append(h("optgroup", { label: U.human(cls) }, byCls[cls].map((id) => h("option", { value: "C:" + id, text: id }))));
    const cb = h("select", { "aria-label": "Color by", id: "v-color", class: V.color ? "on" : "" }, h("option", { value: "", text: "Color by" }),
      Object.entries(COLOR_BY).map(([k, d]) => h("option", { value: k, text: d.label })));
    cb.value = V.color;
    cb.addEventListener("change", () => {
      V.color = cb.value;
      cb.className = V.color ? "on" : "";
      const u = new URL(location.href);
      if (V.color) u.searchParams.set("color", V.color); else u.searchParams.delete("color");
      history.replaceState(null, "", u);
      if (V.ep) drawMap();
    });
    const out = h("div", { class: "out", id: "v-out" });
    V.speed = 1;
    V.play = h("button", { type: "button", class: "ib", id: "v-play", "aria-label": "Play", onclick: toggle });
    const ic = svg("svg", { viewBox: "0 0 16 16", width: 16, height: 16 });
    ic.append(svg("path", { d: PLAY, fill: "currentColor" }));
    V.play.append(ic);
    const sp = h("select", { id: "v-speed", "aria-label": "Speed" }, [0.25, 0.5, 1, 2, 4].map((v) => h("option", { value: v, text: v + "x" })));
    sp.value = "1";
    sp.addEventListener("change", () => { V.speed = +sp.value; });
    const change = (which) => {
      let kart = sk.value, cond = sc.value, item = si.value;
      if (which === "item" && item.startsWith("C:")) cond = sc.value = "nominal";
      if (which === "cond" && cond !== "nominal" && item.startsWith("C:")) item = si.value = "L:" + (V.idx.lines[0] || "center");
      const ok = keyFor(kart, cond, item);
      refresh();
      if (ok) openAt(kart, cond, item);
    };
    sk.addEventListener("change", () => change("kart"));
    sc.addEventListener("change", () => change("cond"));
    si.addEventListener("change", () => change("item"));
    bar.append(sk, sc, si, cb, V.play, sp, out);
    V.sel = { sk, sc, si, out, cb };
    return bar;
  }

  function refresh() {
    const { sk, sc, si } = V.sel;
    for (const o of sc.options) o.disabled = o.value !== "nominal" && !(V.idx.laps[o.value] || {})[sk.value];
    for (const o of si.querySelectorAll("option")) o.disabled = !keyFor(sk.value, o.value.startsWith("C:") ? "nominal" : sc.value, o.value);
  }

  // jobs: [{id, color}]; bd: {laps, clips} with one cell per job (an array) or a single job's cells
  function mount(el, jobs, bd, conds, onOpen) {
    stop();
    V.el = el; V.jobs = jobs; V.multi = jobs.length > 1; V.conds = conds; V.onOpen = onOpen; V.idx = index(bd);
    V.color = colorParam();
    api.current = null; api.keys = [];
    for (const p of V.plots) p.destroy();
    V.plots = [];
    if (V.ro) V.ro.disconnect();
    if (!V.idx.karts.length) { el.replaceChildren(); return; }
    const charts = h("div", { class: "charts", id: "v-charts" });
    V.map = h("div", { class: "map", id: "v-map" });
    V.legend = h("div", { class: "legend", id: "v-legend" });
    V.insetBox = h("div");
    if (!V.tip) { V.tip = h("div", { class: "vtip", id: "v-tip" }); document.body.append(V.tip); }
    el.replaceChildren(h("div", { class: "viewer" }, buildSelects(), h("div", { class: "vbody" }, h("div", { class: "mapcol" }, V.map, V.legend, V.insetBox), charts)));
    V.charts = charts;
    V.ro = new ResizeObserver(() => { const w = V.charts.clientWidth; for (const p of V.plots) if (Math.abs(p.width - w) > 1) p.setSize({ width: w, height: p.height }); });
    V.ro.observe(charts);
  }

  function update(bd) {
    V.idx = index(bd);
    if (V.sel) refresh();
  }

  function openDefault() {
    if (!V.idx || !V.idx.karts.length) return;
    const k = V.idx.karts[0];
    if (keyFor(k, "nominal", "L:center")) { openAt(k, "nominal", "L:center"); return; }
    const first = Object.values(V.idx.byKey)[0];
    if (first) openAt(first.kart, first.cond, first.item);
  }

  function outcomeOf(ep) {
    const o = ep.outcome || {};
    const m = ep.metrics || {};
    if (!ep.context_valid) return h("i", { text: "context invalid" });
    if (o.status === "error") return h("span", { class: "bad", text: "error" });
    if (o.cause) return h("span", { class: "bad", text: `${U.human(o.cause)} at ${ep.events[1].t.toFixed(1)} s` });
    if (ep.scenario.type === "clip") return m.recovered ? `recovered in ${U.num(m.recovery_s, 1)} s` : h("span", { class: "bad", text: "not recovered" });
    const t = ep.log.t ? ep.log.t[ep.log.t.length - 1] : null;
    return t !== null ? `lap ${U.num(t, 1)} s` : "";
  }

  function shortOutcome(ep) {
    const o = ep.outcome || {};
    if (!ep.context_valid) return h("i", { text: "invalid" });
    if (o.status === "error") return h("span", { class: "bad", text: "error" });
    if (o.cause) return h("span", { class: "bad", text: `${U.CAUSE[o.cause] || o.cause} ${ep.events[1].t.toFixed(1)} s` });
    if (ep.scenario.type === "clip") return ep.metrics.recovered ? `${U.num(ep.metrics.recovery_s, 1)} s` : h("span", { class: "bad", text: "no" });
    return ep.log.t ? `${U.num(ep.log.t[ep.log.t.length - 1], 1)} s` : "";
  }

  function outcomeText() {
    const out = V.sel.out;
    out.replaceChildren();
    if (!V.multi) { out.append(outcomeOf(V.ep)); return; }
    V.eps.forEach((ep, i) => {
      const chip = h("span", { class: "oc" + (ep ? "" : " miss") }, h("i", { class: "sw", style: ep ? `background:${V.jobs[i].color}` : "" }));
      if (ep) chip.append(shortOutcome(ep)); else chip.append("–");
      if (!ep) U.tip(chip, "not in this job");
      out.append(chip);
    });
  }

  // ---------- map ----------
  function svg(tag, attrs) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  function mix(stops, f) {
    const t = Math.max(0, Math.min(1, f)) * 2, i = Math.min(1, Math.floor(t)), g = t - i;
    return `rgb(${stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * g)).join(",")})`;
  }

  function quantile(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]; }

  function heatRange(def) {
    const vals = [];
    for (const ep of V.eps) {
      const L = ep && ep.log, v = L && L.x ? def.f(L) : null;
      if (!v) continue;
      v.forEach((x, i) => { if (L.phase[i] >= 2 && x !== null) vals.push(x); });
    }
    if (!vals.length) return null;
    vals.sort((a, b) => a - b);
    if (def.signed) { const m = Math.max(Math.abs(quantile(vals, 0.01)), Math.abs(quantile(vals, 0.99))) || 1; return [-m, m]; }
    const lo = quantile(vals, 0.01), hi = quantile(vals, 0.99);
    return [lo, hi > lo ? hi : lo + 1];
  }

  function heatPath(root, L, v, [lo, hi], stops, pts, width) {
    const bin = (x) => Math.max(0, Math.min(BINS - 1, Math.floor(((x - lo) / (hi - lo)) * BINS)));
    let run = null, b = null;
    const flush = () => { if (run && run.length > 1) root.append(svg("polyline", { points: pts(run), fill: "none", stroke: mix(stops, (b + 0.5) / BINS), "stroke-width": width, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-linecap": "round" })); };
    L.x.forEach((x, i) => {
      if (L.phase[i] < 2 || v[i] === null || v[i] === undefined) return;
      const k = bin(v[i]);
      if (k !== b) { const last = run && run[run.length - 1]; flush(); run = last ? [last] : []; b = k; }
      run.push([x, L.y[i]]);
    });
    flush();
  }

  function drawLegend(def, range, stops) {
    if (!def || !range) { V.legend.replaceChildren(); return; }
    const f = (x) => +x.toFixed(Math.abs(range[1] - range[0]) < 3 ? 2 : Math.abs(range[1] - range[0]) < 30 ? 1 : 0);
    V.legend.replaceChildren(h("span", { text: f(range[0]) }), h("i", { style: `background:linear-gradient(90deg,${[0, 0.5, 1].map((t) => mix(stops, t)).join(",")})` }),
      h("span", { text: f(range[1]) }), h("small", { text: def.unit }));
  }

  function drawMap() {
    const size = 600, padPx = 24;
    const ep = V.ep;
    const lap = ep.scenario.type === "lap";
    let xs = [], ys = [];
    if (lap) { xs = ep.track.outer.map((p) => p[0]); ys = ep.track.outer.map((p) => p[1]); }
    else for (const e of V.eps) if (e && e.log.x) { xs = xs.concat(e.log.x); ys = ys.concat(e.log.y); }
    if (!xs.length) { xs = ep.track.outer.map((p) => p[0]); ys = ep.track.outer.map((p) => p[1]); }
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const span = Math.max(x1 - x0, y1 - y0, lap ? 0 : 36) * (lap ? 1 : 1.5);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const s = (size - 2 * padPx) / span;
    const P = (x, y) => [size / 2 + (x - cx) * s, size / 2 - (y - cy) * s];
    V.proj = P; V.projScale = s;
    const pts = (arr) => arr.map((p) => P(p[0], p[1]).map((v) => v.toFixed(1)).join(",")).join(" ");
    const root = svg("svg", { viewBox: `0 0 ${size} ${size}`, preserveAspectRatio: "xMidYMid meet" });
    const ring = (arr) => "M" + pts(arr).replace(/ /g, " L") + " Z";
    root.append(svg("path", { d: ring(ep.track.outer) + " " + ring(ep.track.inner), fill: "#F7F7F5", stroke: "#E6E5E1", "stroke-width": 1.5, "vector-effect": "non-scaling-stroke", "fill-rule": "evenodd" }));
    const line = ep.line.x.map((x, i) => [x, ep.line.y[i]]);
    if (lap) line.push(line[0]);
    if (V.color === "cost" && ep.plan && ep.plan.field) root.append(costField(ep, line, ring, P, s));
    root.append(svg("polyline", { points: pts(line), fill: "none", stroke: "#000", "stroke-width": 1, "stroke-dasharray": "5 4", opacity: 0.85, "vector-effect": "non-scaling-stroke" }));
    const def = COLOR_BY[V.color];
    const range = def ? heatRange(def) : null;
    const stops = def && def.stops ? COST : def && def.signed ? DIV : SEQ;
    drawLegend(def, range, stops);
    if (V.color === "cost" && ep.plan && ep.plan.field) {
      const c = ep.plan.field.c, f = (x) => +x.toPrecision(2);
      V.legend.append(h("span", { class: "fl", text: "field" }), h("span", { text: f(Math.min(...c)) }), h("i", { style: `background:linear-gradient(90deg,${[0, 0.5, 1].map((t) => fieldColor(t)).join(",")})` }), h("span", { text: f(Math.max(...c)) }), h("small", { text: "cost/step" }));
    }
    V.pins = [];
    V.eps.forEach((e, j) => {
      const L = e && e.log;
      if (!L || !L.x) return;
      const segs = [];
      let cur = null;
      L.x.forEach((x, i) => {
        const auto = L.phase[i] >= 2;
        if (!cur || cur.auto !== auto) { if (cur) cur.p.push([x, L.y[i]]); cur = { auto, p: [] }; segs.push(cur); }
        cur.p.push([x, L.y[i]]);
      });
      const color = V.multi ? V.jobs[j].color : "#CFB991";
      const v = range ? def.f(L) : null;
      for (const g of segs) {
        if (!g.auto) { root.append(svg("polyline", { points: pts(g.p), fill: "none", stroke: V.multi || v ? "#B9BBC0" : "#6F727B", "stroke-width": V.multi || v ? 1.5 : 2, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-linecap": "round" })); continue; }
        if (v && V.multi) root.append(svg("polyline", { points: pts(g.p), fill: "none", stroke: color, "stroke-width": 6, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-linecap": "round" }));
        if (!v) root.append(svg("polyline", { points: pts(g.p), fill: "none", stroke: color, "stroke-width": 2, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-linecap": "round", "data-job": j }));
      }
      if (v) heatPath(root, L, v, range, stops, pts, V.multi ? 3 : 3.5);
      const fail = e.events.find((x) => x.kind !== "handoff");
      if (fail) {
        const [fx, fy] = P(L.x[L.x.length - 1], L.y[L.y.length - 1]);
        const g = svg("g", { stroke: V.multi ? color : "#B42318", "stroke-width": 2.5, "stroke-linecap": "round", "data-mark": "failure" });
        g.append(svg("line", { x1: -6, y1: -6, x2: 6, y2: 6 }), svg("line", { x1: -6, y1: 6, x2: 6, y2: -6 }));
        root.append(g);
        V.pins.push([g, fx, fy]);
      }
    });
    const L0 = ep.log;
    if (L0.x) {
      const hi = L0.t.findIndex((t) => t >= 0);
      if (hi > 0) {
        const [ax, ay] = P(L0.x[hi - 1], L0.y[hi - 1]), [bx, by] = P(L0.x[hi], L0.y[hi]);
        const dx = bx - ax, dy = by - ay, n = Math.hypot(dx, dy) || 1;
        const nx = (-dy / n) * 8, ny = (dx / n) * 8;
        const hl = svg("line", { x1: -nx, y1: -ny, x2: nx, y2: ny, stroke: "#000", "stroke-width": 2.5, "data-mark": "handoff" });
        root.append(hl);
        V.pins.push([hl, bx, by]);
      }
    }
    // track edges: thin gray lines over the field and the coloured paths
    root.append(svg("path", { d: ring(ep.track.outer) + " " + ring(ep.track.inner), fill: "none", stroke: "#9A9DA3", "stroke-width": 1, "vector-effect": "non-scaling-stroke", "data-mark": "edges" }));
    V.costRange = costRange(ep);
    V.planMap = V.eps.map((e, j) => {
      if (!e || !e.plan || !e.plan.t.length) return null;
      const c = V.multi ? V.jobs[j].color : "#000";
      const pl = svg("polyline", { points: "", fill: "none", stroke: c, "stroke-width": 2, opacity: 0.85, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-linecap": "round", visibility: "hidden", "data-job": j, "data-mark": "plan" });
      pl.end = svg("circle", { r: 3, fill: c, visibility: "hidden" });
      root.append(pl, pl.end);
      if (!V.multi && e.plan.cost) { pl.segs = costSegs(e.plan.cost[0].length, 3); pl.segs.job = j; root.append(pl.segs); }
      return pl;
    });
    V.dots = V.eps.map((e, j) => {
      if (!e || !e.log.x) return null;
      const d = svg("g", { visibility: "hidden", class: "kdot", "data-job": j });
      d.append(svg("path", { d: "M9 0 L-6 -6 L-3 0 L-6 6 Z", fill: V.multi ? V.jobs[j].color : "#000", stroke: "#fff", "stroke-width": 1.5, "stroke-linejoin": "round" }));
      root.append(d);
      return d;
    });
    const first = V.dots.find(Boolean);
    if (first) first.id = "v-dot";
    const sig = `${cx},${cy},${s}`;
    if (V.mapSig !== sig) { V.vb = [0, 0, size, size]; V.mapSig = sig; V.follow = false; }
    V.mapRoot = root; V.mapSize = size;
    if (!(V.cur < V.T.length)) V.cur = null;
    zoomable(root);
    costHover(root, ep);
    const fit = h("button", { type: "button", class: "zb", id: "v-fit", "aria-label": "Fit", text: "Fit", onclick: () => { V.follow = false; V.vb = [0, 0, size, size]; applyVB(); } });
    const fol = h("button", { type: "button", class: "zb" + (V.follow ? " on" : ""), id: "v-follow", "aria-label": "Follow", text: "Follow" });
    fol.addEventListener("click", () => { V.follow = !V.follow; fol.classList.toggle("on", V.follow); if (V.follow && V.vb[2] >= size) V.vb = [V.vb[0], V.vb[1], size / 4, size / 4]; applyVB(); });
    V.followBtn = fol;
    V.map.replaceChildren(root, h("div", { class: "zbar" }, fit, fol));
    applyVB();
    V.insetBox.replaceChildren(drawInset(ep));
    if (V.costRange && !V.multi) {   // rollout colour scale
      const f = (y) => +(Math.sign(y) * (10 ** Math.abs(y) - 1)).toPrecision(2);
      V.insetBox.append(h("div", { class: "legend", id: "v-rlegend" }, h("span", { text: "rollout " + f(V.costRange[0]) }), h("i", { style: `background:linear-gradient(90deg,${[0, 0.5, 1].map((t) => mix(COST, t)).join(",")})` }), h("span", { text: f(V.costRange[1]) }), h("small", { text: "cost/step" })));
    }
  }

  // ---------- MPC cost: position field on the track, rollout coloured per step ----------
  const fieldColor = (t) => mix([0, 1, 2].map((k) => COST[k].map((c) => Math.round(c * 0.45 + 247 * 0.55))), t);   // opaque, washed toward the track fill
  function costField(ep, line, ring, P, s) {
    // the track plus 1 m beyond each edge: the surface filled, both edges stroked 2 m wide
    const F = ep.plan.field, g = svg("g", { mask: "url(#v-trk)", "data-mark": "field" });
    const mk = svg("mask", { id: "v-trk", maskUnits: "userSpaceOnUse", x: -1e4, y: -1e4, width: 2e4, height: 2e4 });
    mk.append(svg("path", { d: ring(ep.track.outer) + " " + ring(ep.track.inner), fill: "#fff", stroke: "#fff", "stroke-width": (2 * s).toFixed(2), "stroke-linejoin": "round", "fill-rule": "evenodd" }));
    g.append(mk);
    const pts = line.filter((p, i) => i % 5 === 0 || i === line.length - 1);
    const nrm = pts.map((p, i) => { const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)], l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1; return [-(b[1] - a[1]) / l, (b[0] - a[0]) / l]; });
    const lg = F.c.map((c) => Math.log10(1 + Math.max(0, c))), top = Math.max(...lg) || 1;
    const n = F.c.length, order = [];
    for (let k = 0; k + 1 < n; k += 2) order.push(k);
    order.sort((a, b) => Math.abs(F.d0 + (b + 1) * F.dd) - Math.abs(F.d0 + (a + 1) * F.dd));   // widest first: the nearest offset paints last, so folds inside tight corners are overdrawn
    for (const k of order) {   // 0.2 m bands, coloured by log cost
      const d = F.d0 + (k + 1) * F.dd;
      const q = pts.map((p, i) => P(p[0] + nrm[i][0] * d, p[1] + nrm[i][1] * d).map((v) => v.toFixed(1)).join(",")).join(" ");
      g.append(svg("polyline", { points: q, fill: "none", stroke: fieldColor(((lg[k] + lg[k + 1]) / 2) / top), "stroke-width": (2 * F.dd * s + 0.6).toFixed(2), "stroke-linejoin": "round" }));
    }
    return g;
  }
  const symlog = (x) => Math.sign(x) * Math.log10(1 + Math.abs(x));   // step costs span decades and go negative (progress)
  function costRange(ep) {
    const all = (ep.plan && ep.plan.cost || []).flat().map(symlog).sort((a, b) => a - b);
    return all.length ? [quantile(all, 0.02), Math.max(quantile(all, 0.98), quantile(all, 0.02) + 1e-6)] : null;
  }
  // one <line> per horizon step, coloured by that step's cost; hover shows the per-term breakdown
  function costSegs(n, width) {
    const g = svg("g", { "data-mark": "plancost" });
    for (let k = 0; k < n; k++) {
      const l = svg("line", { "stroke-width": width, "stroke-linecap": "round", "vector-effect": "non-scaling-stroke", "data-k": k });
      l.addEventListener("pointerenter", (ev) => showTerms(ev, g, k));
      l.addEventListener("pointerleave", () => { V.tip.style.display = "none"; });
      g.append(l);
    }
    return g;
  }
  function paintSegs(g, xy, cost, f) {
    const n = Math.min(g.childNodes.length, cost ? cost.length : 0), [lo, hi] = V.costRange || [0, 1];
    g.childNodes.forEach((l, k) => {
      if (k >= n) { l.setAttribute("visibility", "hidden"); return; }
      const [ax, ay] = f(xy[2 * k], xy[2 * k + 1]), [bx, by] = f(xy[2 * k + 2], xy[2 * k + 3]);
      l.setAttribute("x1", ax); l.setAttribute("y1", ay); l.setAttribute("x2", bx); l.setAttribute("y2", by);
      l.setAttribute("stroke", mix(COST, (symlog(cost[k]) - lo) / (hi - lo)));
      l.setAttribute("visibility", "visible");
    });
  }
  async function showTerms(ev, g, k) {
    const i = g.planIdx, ep = V.eps[g.job];
    if (i === undefined || i < 0 || !ep) return;
    const url = `/api/episodes/${encodeURIComponent(V.jobs[g.job].id)}/${ep.key}/plan/${i}`;
    V.termCache ||= new Map();
    if (!V.termCache.has(url)) V.termCache.set(url, U.api(url));
    const r = await V.termCache.get(url);
    if (!r.ok || g.planIdx !== i) return;
    const row = r.body.steps[k] || [], tot = row.reduce((a, b) => a + b, 0), f = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
    V.tip.replaceChildren(h("b", { text: `step ${k + 1}  ${f(tot)}` }), ...r.body.terms.map((t, m) => [t, row[m]]).filter(([, v]) => Math.abs(v) > 1e-3).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5).map(([t, v]) => h("div", { text: `${t} ${f(v)}` })));
    Object.assign(V.tip.style, { display: "block", left: ev.clientX + 12 + "px", top: ev.clientY + 12 + "px" });
  }

  // cost mode: hovering the driven path shows that tick's cost breakdown, hovering the field its position cost
  function bubble(ev, title, rows) {
    V.tip.replaceChildren(h("b", { text: title }), ...rows.map((r) => h("div", { text: r })));
    Object.assign(V.tip.style, { display: "block", left: ev.clientX + 12 + "px", top: ev.clientY + 12 + "px" });
  }
  function costHover(root, ep) {
    const f = (x) => (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(1));
    root.addEventListener("pointermove", (ev) => {
      if (V.color !== "cost" || ev.target.closest("[data-mark=plancost]")) return;
      const m = root.getScreenCTM(), px = (ev.clientX - m.e) / m.a, py = (ev.clientY - m.f) / m.d, r = 8 / m.a;
      let best = null;
      V.eps.forEach((e, j) => {
        const L = e && e.log, C = L && L.cost_terms;
        if (!C) return;
        L.x.forEach((x, i) => {
          if (L.phase[i] < 2) return;
          const [qx, qy] = V.proj(x, L.y[i]), d2 = (qx - px) ** 2 + (qy - py) ** 2;
          if (d2 < r * r && (!best || d2 < best.d2)) best = { d2, j, i };
        });
      });
      if (best) {
        const L = V.eps[best.j].log, C = L.cost_terms, terms = Object.keys(C).filter((t) => C[t][best.i] !== null);
        if (terms.length) {
          const tot = terms.reduce((a, t) => a + C[t][best.i], 0);
          const top = terms.map((t) => [t, C[t][best.i]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5);
          bubble(ev, `${L.t[best.i].toFixed(2)} s   total ${f(tot)}`, top.map(([t, x]) => `${TERM(t)} ${f(x)}`));
          return;
        }
      }
      const F = ep.plan && ep.plan.field;
      if (F && ev.target.closest("[data-mark=field]")) {
        // nearest line point gives the lateral offset; the field is the MPC's position cost at that offset
        let k = 0, b2 = Infinity;
        ep.line.x.forEach((x, i) => { const [qx, qy] = V.proj(x, ep.line.y[i]), d2 = (qx - px) ** 2 + (qy - py) ** 2; if (d2 < b2) { b2 = d2; k = i; } });
        const d = Math.sqrt(b2) / V.projScale, n = Math.max(0, Math.min(F.c.length - 1, Math.round((d - F.d0) / F.dd)));
        bubble(ev, `${d.toFixed(1)} m off the line`, [`position cost ${f(F.c[n])} per step`]);
        return;
      }
      V.tip.style.display = "none";
    });
    root.addEventListener("pointerleave", () => { V.tip.style.display = "none"; });
  }

  // ---------- main map zoom: wheel zooms about the pointer, drag pans, double-click fits ----------
  function applyVB() { setVB(); moveDot(V.cur); }
  function setVB() {
    const root = V.mapRoot;
    if (!root) return;
    if (V.follow && V.cur !== undefined && V.cur !== null && V.cur >= 0) {
      const j = V.dots.findIndex(Boolean), i = j >= 0 ? V.M[j][V.cur] : -1;
      if (i >= 0) { const [x, y] = V.proj(V.eps[j].log.x[i], V.eps[j].log.y[i]); V.vb[0] = x - V.vb[2] / 2; V.vb[1] = y - V.vb[3] / 2; }
    }
    root.setAttribute("viewBox", V.vb.map((v) => v.toFixed(2)).join(" "));
    const k = V.vb[2] / V.mapSize;
    for (const [el, x, y] of V.pins) el.setAttribute("transform", `translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${k.toFixed(4)})`);
  }
  function zoomable(root) {
    const pt = (ev) => { const p = root.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY; return p.matrixTransform(root.getScreenCTM().inverse()); };
    root.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const p = pt(ev), [x, y, w] = V.vb, nw = Math.min(V.mapSize, Math.max(4, w * Math.exp(ev.deltaY * 0.0015))), r = nw / w;
      V.vb = nw >= V.mapSize ? [0, 0, nw, nw] : [p.x - (p.x - x) * r, p.y - (p.y - y) * r, nw, nw];
      applyVB();
    }, { passive: false });
    let drag = null;
    root.addEventListener("pointerdown", (ev) => { if (ev.button === 0) drag = { cx: ev.clientX, cy: ev.clientY, vb: V.vb.slice(), m: root.getScreenCTM().a }; });
    root.addEventListener("pointermove", (ev) => {
      if (!drag || (!drag.on && Math.hypot(ev.clientX - drag.cx, ev.clientY - drag.cy) < 3)) return;
      if (!drag.on) { drag.on = true; root.setPointerCapture(ev.pointerId); root.classList.add("pan"); V.follow = false; V.followBtn.classList.remove("on"); }
      V.vb = [drag.vb[0] - (ev.clientX - drag.cx) / drag.m, drag.vb[1] - (ev.clientY - drag.cy) / drag.m, drag.vb[2], drag.vb[3]];
      applyVB();
    });
    const end = () => { drag = null; root.classList.remove("pan"); };
    root.addEventListener("pointerup", end);
    root.addEventListener("pointercancel", end);
    root.addEventListener("dblclick", () => { V.follow = false; V.followBtn.classList.remove("on"); V.vb = [0, 0, V.mapSize, V.mapSize]; applyVB(); });
  }

  // the planner's intended path (flat x, y list) nearest time t, within one 20 Hz plan period
  function planAt(e, t) { const i = planIdx(e, t); return i < 0 ? null : e.plan.xy[i]; }
  function planIdx(e, t) {
    const P = e && e.plan;
    if (!P || !P.t.length) return -1;
    let lo = 0, hi = P.t.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (P.t[m] < t) lo = m + 1; else hi = m; }
    if (lo > 0 && t - P.t[lo - 1] < P.t[lo] - t) lo--;
    return Math.abs(P.t[lo] - t) <= 0.05 ? lo : -1;
  }

  function pairs(xy, f) {
    const out = [];
    for (let k = 0; k + 1 < xy.length; k += 2) out.push(f(xy[k], xy[k + 1]));
    return out.join(" ");
  }

  // ---------- top-down kart inset: heading-up, metres, y flipped; wheel zooms ----------
  const VIEW = 4.0, VIEW_PLAN = 18.0;   // with a plan, a wider view that holds the rollout ahead of the kart
  function kartGlyph(g, color) {
    const k = svg("g", {});
    const wb = g.wheelbase, hw = 0.3;
    k.append(svg("rect", { x: -0.21, y: -hw, width: wb + 0.465, height: 2 * hw, rx: 0.08, fill: V.multi ? "rgba(255,255,255,.55)" : "#CFB991", stroke: color, "stroke-width": V.multi ? 2 : 1, "vector-effect": "non-scaling-stroke" }));
    k.append(svg("path", { d: `M0 ${-g.track_r / 2} V${g.track_r / 2} M${wb} ${-g.track_f / 2} V${g.track_f / 2}`, stroke: color, "stroke-width": 1.5, "vector-effect": "non-scaling-stroke" }));
    const wheel = (x, y, [d, w]) => svg("rect", { x: -d / 2, y: -w / 2, width: d, height: w, rx: 0.02, fill: "#000", transform: `translate(${x} ${y})` });
    k.append(wheel(0, -g.track_r / 2, g.tyre_r), wheel(0, g.track_r / 2, g.tyre_r));
    const front = [-1, 1].map((side) => {
      const f = svg("g", {});
      f.append(wheel(0, 0, g.tyre_f));
      k.append(f);
      return { f, y: (side * g.track_f) / 2 };
    });
    return { k, front, wb };
  }

  function drawInset(ep) {
    V.kart = null;
    if (!ep.log.x || !ep.geom) return "";
    const root = svg("svg", { class: "inset", id: "v-inset", viewBox: `0 0 ${VIEW} ${VIEW}` });
    const world = svg("g", {});
    root.append(world);
    const ring = (arr) => "M" + arr.map((p) => `${p[0]},${-p[1]}`).join(" L") + " Z";
    world.append(svg("path", { d: ring(ep.track.outer) + " " + ring(ep.track.inner), fill: "#F7F7F5", stroke: "#6F727B", "stroke-width": 1, "vector-effect": "non-scaling-stroke", "fill-rule": "evenodd" }));
    const line = ep.line.x.map((x, i) => `${x},${-ep.line.y[i]}`).join(" ");
    world.append(svg("polyline", { points: line, fill: "none", stroke: "#000", "stroke-width": 1, "stroke-dasharray": "4 3", "vector-effect": "non-scaling-stroke" }));
    const plans = V.eps.map((e, j) => {
      if (!e || !e.plan || !e.plan.t.length) return null;
      const c = V.multi ? V.jobs[j].color : "#CFB991";
      const g = svg("g", { visibility: "hidden", "data-job": j, "data-mark": "plan" });
      g.append(svg("polyline", { points: "", fill: "none", stroke: c, "stroke-width": 2.5, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-linecap": "round" }), svg("circle", { r: 0.12, fill: c }));
      if (!V.multi && e.plan.cost) { g.segs = costSegs(e.plan.cost[0].length, 4); g.segs.job = j; g.append(g.segs); }
      world.append(g);
      return g;
    });
    const glyphs = V.eps.map((e, j) => {
      if (!e || !e.log.x || !e.geom) return null;
      const gl = kartGlyph(e.geom, V.multi ? V.jobs[j].color : "#000");
      gl.k.setAttribute("data-job", j);
      return gl;
    });
    for (let j = glyphs.length - 1; j >= 0; j--) if (glyphs[j]) world.append(glyphs[j].k);   // first job on top
    const view = plans.some(Boolean) ? VIEW_PLAN : VIEW;
    V.kart = { root, world, glyphs, plans, view: V.insetView && V.insetView[0] === view ? V.insetView[1] : view, base: view };
    root.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      V.kart.view = Math.min(80, Math.max(2, V.kart.view * Math.exp(ev.deltaY * 0.0015)));
      V.insetView = [V.kart.base, V.kart.view];
      moveInset(V.insetU);
    }, { passive: false });
    root.addEventListener("dblclick", () => { V.kart.view = V.kart.base; V.insetView = null; moveInset(V.insetU); });
    return root;
  }

  function placeGlyph(K, L, i) {
    const yaw = L.yaw[i];
    K.k.setAttribute("transform", `translate(${L.x[i]} ${-L.y[i]}) rotate(${(-yaw * 180) / Math.PI})`);
    // front[0] is y = -track/2 in svg, the kart's left wheel (kart y is flipped)
    K.front[0].f.setAttribute("transform", `translate(${K.wb} ${K.front[0].y}) rotate(${-L.wheel_l[i]})`);
    K.front[1].f.setAttribute("transform", `translate(${K.wb} ${K.front[1].y}) rotate(${-L.wheel_r[i]})`);
  }

  function moveInset(u) {
    const K = V.kart;
    if (!K) return;
    V.insetU = u;
    let centred = false;
    V.eps.forEach((e, j) => {
      const gl = K.glyphs[j];
      if (!gl) return;
      const i = V.M[j][u];
      gl.k.setAttribute("visibility", i < 0 ? "hidden" : "visible");
      if (i < 0) return;
      const L = e.log;
      placeGlyph(gl, L, i);
      const pg = K.plans[j], pi = pg ? planIdx(e, L.t[i]) : -1, xy = pi >= 0 ? e.plan.xy[pi] : null;
      if (pg) pg.setAttribute("visibility", xy ? "visible" : "hidden");
      if (xy && pg.segs) { pg.segs.planIdx = pi; paintSegs(pg.segs, xy, e.plan.cost[pi], (x, y) => [x, -y]); }
      if (xy) {
        const n = xy.length;
        pg.firstChild.setAttribute("points", pairs(xy, (x, y) => `${x.toFixed(3)},${(-y).toFixed(3)}`));
        const dot = pg.childNodes[1];
        dot.setAttribute("cx", xy[n - 2]); dot.setAttribute("cy", -xy[n - 1]); dot.setAttribute("r", (K.view * 0.012).toFixed(3));
      }
      if (!centred) {
        // heading-up: the world turns so the kart points up; with a plan the kart sits low and the rollout fills the view
        const c = Math.cos(L.yaw[i]), s = Math.sin(L.yaw[i]), v = K.view, ahead = gl.wb / 2 + (K.base === VIEW ? 0 : v * 0.3);
        const mx = L.x[i] + c * ahead, my = L.y[i] + s * ahead;
        K.root.setAttribute("viewBox", `${(-v / 2).toFixed(3)} ${(-v / 2).toFixed(3)} ${v.toFixed(3)} ${v.toFixed(3)}`);
        K.world.setAttribute("transform", `rotate(${((L.yaw[i] * 180) / Math.PI - 90).toFixed(2)}) translate(${(-mx).toFixed(3)} ${my.toFixed(3)})`);
        centred = true;
      }
    });
  }

  function moveDot(u) {
    V.cur = u;
    if (u !== null && u !== undefined && u >= 0) V.last = u;
    if (!V.ep || u === null || u === undefined || u < 0) { for (const d of V.dots.concat(V.planMap, V.planMap.map((p) => p && p.end))) if (d) d.setAttribute("visibility", "hidden"); return; }
    if (V.follow) setVB();
    const k = V.vb[2] / V.mapSize;
    V.eps.forEach((e, j) => {
      const d = V.dots[j], pl = V.planMap[j];
      if (!d) return;
      const i = V.M[j][u];
      const pi = i >= 0 && pl ? planIdx(e, e.log.t[i]) : -1, xy = pi >= 0 ? e.plan.xy[pi] : null;
      if (pl) { for (const el of [pl, pl.end, pl.segs]) if (el) el.setAttribute("visibility", xy ? "visible" : "hidden"); }
      if (xy && pl.segs) { pl.segs.planIdx = pi; paintSegs(pl.segs, xy, e.plan.cost[pi], V.proj); }
      if (xy) {
        pl.setAttribute("points", pairs(xy, (x, y) => V.proj(x, y).map((q) => q.toFixed(2)).join(",")));
        const [ex, ey] = V.proj(xy[xy.length - 2], xy[xy.length - 1]);
        pl.end.setAttribute("transform", `translate(${ex.toFixed(2)} ${ey.toFixed(2)}) scale(${k.toFixed(4)})`);
      }
      if (i < 0) { d.setAttribute("visibility", "hidden"); return; }
      const [x, y] = V.proj(e.log.x[i], e.log.y[i]);
      d.setAttribute("transform", `translate(${x.toFixed(2)} ${y.toFixed(2)}) rotate(${((-e.log.yaw[i] * 180) / Math.PI).toFixed(1)}) scale(${k.toFixed(4)})`);
      d.setAttribute("visibility", "visible");
    });
    moveInset(u);
  }

  // ---------- shared time base: union of every episode's samples ----------
  function merge() {
    const all = [];
    for (const e of V.eps) if (e && e.log.t) all.push(...e.log.t);
    all.sort((a, b) => a - b);
    V.T = all.filter((t, i) => !i || t - all[i - 1] > 1e-6);
    V.M = V.eps.map((e) => {
      const t = e && e.log.t;
      if (!t) return V.T.map(() => -1);
      let i = -1;
      return V.T.map((u) => { while (i < t.length - 1 && t[i + 1] <= u + 1e-6) i++; return u > t[t.length - 1] + 0.05 ? t.length - 1 : i; });
    });
  }

  function aligned(j, key) {
    const e = V.eps[j], t = e.log.t, v = e.log[key] || [];   // a key the server does not send yet plots empty
    return V.T.map((u, k) => { const i = V.M[j][k]; return i >= 0 && Math.abs(t[i] - u) < 1e-6 ? v[i] : null; });
  }

  // ---------- play ----------
  const PLAY = "M5 3.5v9l7.5-4.5z", PAUSE = "M4.5 3.5h2.5v9H4.5zM9 3.5h2.5v9H9z";
  function playIcon(on) { V.play.firstChild.firstChild.setAttribute("d", on ? PAUSE : PLAY); V.play.setAttribute("aria-label", on ? "Pause" : "Play"); }
  function stop() { if (V.raf) cancelAnimationFrame(V.raf); V.raf = null; if (V.play) playIcon(false); }
  function cursorAt(i) {
    for (const p of V.plots) p.setCursor({ left: p.valToPos(V.T[i], "x"), top: p.bbox.height / devicePixelRatio / 2 });
    moveDot(i);
  }
  // ---------- pinned cursor: click a chart to pin, click again or Escape to unpin; arrows step and hold at 0.25x ----------
  const pinned = () => V.pin !== null && V.pin !== undefined;
  function holdPin(u) {   // a hover moved a chart's cursor off the pin: put every chart back on it
    if (!pinned() || u.cursor.idx === V.pin || V.pinFix) return false;
    V.pinFix = requestAnimationFrame(() => { V.pinFix = null; if (pinned()) cursorAt(V.pin); });
    return true;
  }
  function setPin(i) {
    V.pin = i;
    if (V.charts) V.charts.classList.toggle("pinned", pinned());
    if (pinned()) cursorAt(i);
  }
  function pinClick(u) {
    if (pinned()) { setPin(null); return; }
    const i = u.cursor.idx;
    if (i !== null && i !== undefined) { stop(); setPin(i); }
  }
  function keys(ev) {
    if (!V.T.length || !V.el || !V.el.isConnected || /^(INPUT|SELECT|TEXTAREA)$/.test((ev.target || {}).tagName || "")) return;
    if (ev.type === "keydown" && ev.key === "Escape" && pinned()) { setPin(null); return; }
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    ev.preventDefault();
    const dir = ev.key === "ArrowRight" ? 1 : -1;
    if (ev.type === "keyup") { if (V.hold && V.hold.dir === dir) { cancelAnimationFrame(V.hold.raf); V.hold = null; } return; }
    if (ev.repeat || (V.hold && V.hold.dir === dir)) return;
    stop();
    if (V.hold) cancelAnimationFrame(V.hold.raf);
    const from = pinned() ? V.pin : V.cur !== null && V.cur !== undefined && V.cur >= 0 ? V.cur : 0;
    let i = Math.max(0, Math.min(V.T.length - 1, from + dir));
    setPin(i);
    // held: after a tap's grace period, sweep sim time at 0.25 s per wall second
    const t0 = performance.now(), sim0 = V.T[i];
    const hold = { dir, raf: null };
    const step = (now) => {
      const dt = (now - t0) / 1000 - 0.25;
      if (dt > 0) {
        const t = sim0 + dir * dt * 0.25;
        let j = i;
        while (dir > 0 && j < V.T.length - 1 && V.T[j + 1] <= t) j++;
        while (dir < 0 && j > 0 && V.T[j - 1] >= t) j--;
        if (j !== i) { i = j; setPin(i); }
      }
      hold.raf = requestAnimationFrame(step);
    };
    hold.raf = requestAnimationFrame(step);
    V.hold = hold;
  }
  document.addEventListener("keydown", keys);
  document.addEventListener("keyup", keys);
  window.addEventListener("blur", () => { if (V.hold) { cancelAnimationFrame(V.hold.raf); V.hold = null; } });

  function toggle() {
    if (V.raf) { stop(); return; }
    const T = V.T;
    if (!T.length) return;
    let i = pinned() ? V.pin : V.last || 0;   // from the cursor (the last one shown), or the start
    if (pinned()) setPin(null);
    if (i >= T.length - 1) i = 0;
    let t = T[i], last = null;
    playIcon(true);
    const step = (now) => {
      if (last !== null) t += ((now - last) / 1000) * V.speed;
      last = now;
      while (i < T.length - 1 && T[i + 1] <= t) i++;
      cursorAt(i);
      if (i >= T.length - 1) { stop(); return; }
      V.raf = requestAnimationFrame(step);
    };
    V.raf = requestAnimationFrame(step);
  }

  // ---------- charts ----------
  function bands() {
    return (u) => {
      const ctx = u.ctx, { left, top, width, height } = u.bbox;
      const x0 = u.valToPos(0, "x", true);
      ctx.save();
      if (x0 > left) { ctx.fillStyle = "#F7F7F5"; ctx.fillRect(left, top, Math.min(x0, left + width) - left, height); }
      V.eps.forEach((ep, j) => {
        if (!ep) return;
        for (const e of ep.events) {
          if (e.kind === "handoff" && ep !== V.ep) continue;
          const x = Math.round(u.valToPos(e.t, "x", true)) + 0.5;
          if (x < left || x > left + width) continue;
          ctx.strokeStyle = e.kind === "handoff" ? "#000" : V.multi ? V.jobs[j].color : "#B42318";
          ctx.lineWidth = devicePixelRatio;
          ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
        }
      });
      ctx.restore();
    };
  }

  // faint band between the axles' peak-slip angles on both sides; beyond it the tyre slides
  function peakBand([lo, hi]) {
    return (u) => {
      const ctx = u.ctx, { left, width } = u.bbox;
      ctx.save();
      ctx.fillStyle = "rgba(181,101,29,0.16)";
      for (const sg of [1, -1]) {
        const a = u.valToPos(sg * lo, "y", true), b = u.valToPos(sg * Math.max(hi, lo + 0.3), "y", true);
        ctx.fillRect(left, Math.min(a, b), width, Math.abs(b - a));
      }
      ctx.restore();
    };
  }

  // thin horizontal lines at +/- each limit
  function limits(lines) {
    return (u) => {
      const ctx = u.ctx, { left, width } = u.bbox;
      ctx.save();
      ctx.lineWidth = devicePixelRatio;
      for (const l of lines) {
        ctx.strokeStyle = l.color;
        ctx.setLineDash(l.dash ? l.dash.map((d) => d * devicePixelRatio) : []);
        for (const sg of [1, -1]) {
          const y = Math.round(u.valToPos(sg * l.v, "y", true)) + 0.5;
          if (y < u.bbox.top || y > u.bbox.top + u.bbox.height) continue;
          ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + width, y); ctx.stroke();
        }
      }
      ctx.restore();
    };
  }

  // unsafe intervals: one lane per criterion (one job) or per job (compare), colored by criterion
  const UNSAFE = { lat: ["lat", "#B5651D"], rate: ["rate", "#4A6FA5"], travel: ["travel", "#2E8B7A"], spin: ["spin", "#B42318"], off: ["off", "#6F727B"] };
  function strip() {
    const lanes = V.multi ? V.eps.map((e, j) => j) : Object.keys(UNSAFE);
    const keys = Object.entries(UNSAFE).map(([c, [name, color]]) => h("span", { class: "key", "data-c": c }, h("i", { style: `background:${color}` }), name));
    const box = h("div", { class: "chart strip", "data-chart": "Unsafe" }, h("div", { class: "ch" }, h("span", { class: "lead" }, h("b", { text: "Unsafe" })), keys));
    V.charts.append(box);
    const axis = { stroke: "#6F727B", grid: { show: false }, ticks: { show: false }, font: "10px -apple-system, BlinkMacSystemFont, sans-serif", size: 48, gap: 6 };
    const draw = (u) => {
      const ctx = u.ctx, { left, top, width, height } = u.bbox, lh = height / lanes.length;
      ctx.save();
      lanes.forEach((lane, k) => {
        const y = top + k * lh + lh * 0.15, hh = lh * 0.7;
        const items = V.multi ? Object.keys(UNSAFE).map((c) => [c, ((V.eps[lane] || {}).unsafe || {})[c] || []]) : V.eps.map((e) => [lane, (e && e.unsafe && e.unsafe[lane]) || []]);
        ctx.fillStyle = "#F7F7F5"; ctx.fillRect(left, y, width, hh);
        if (V.multi) { ctx.fillStyle = V.jobs[lane].color; ctx.fillRect(left, y, 3 * devicePixelRatio, hh); }
        for (const [c, iv] of items) {
          ctx.fillStyle = UNSAFE[c][1];
          for (const [a, b] of iv) {
            const x0 = Math.max(left, u.valToPos(a, "x", true)), x1 = Math.min(left + width, u.valToPos(b, "x", true));
            if (x1 > x0) ctx.fillRect(x0, y, Math.max(x1 - x0, devicePixelRatio), hh);
          }
        }
      });
      ctx.restore();
    };
    const opts = {
      width: V.charts.clientWidth, height: 14 + lanes.length * 14, legend: { show: false },
      cursor: { sync: { key: "ep" }, y: false, points: { show: false }, drag: { x: false, y: false } },
      scales: { x: { time: false, range: () => [V.T[0], V.T[V.T.length - 1]] }, y: { range: () => [0, lanes.length] } },
      axes: [Object.assign({}, axis, { size: 6, values: (u, v) => v.map(() => "") }),
        Object.assign({}, axis, { splits: () => lanes.map((l, k) => lanes.length - k - 0.5), values: (u, v) => (V.multi ? v.map(() => "") : lanes.map((c) => UNSAFE[c][0])) })],
      series: [{}],
      hooks: {
        draw: [draw],
        setCursor: [(u) => {
          if (holdPin(u)) return;
          const i = u.cursor.idx, t = i === null || i === undefined ? null : V.T[i];
          for (const el of keys) {
            const c = el.dataset.c;
            el.classList.toggle("on", t !== null && V.eps.some((e) => e && e.unsafe && (e.unsafe[c] || []).some(([a, b]) => t >= a && t < b)));
          }
        }],
      },
    };
    V.plots.push(new uPlot(opts, [V.T], box));
  }

  // MPC cost of the chosen trajectory per term, summed over the horizon: one lane per term plus the total (one job) or one total lane per job (compare)
  const TERM = (t) => ({ terminal_heading: "end hdg", delta_rate: "rate", delta: "steer", a_lat: "lat acc" })[t] || t;
  function costStrip() {
    const jobs = V.eps.map((e, j) => (e && e.log.cost_terms ? j : -1)).filter((j) => j >= 0);
    const terms = Object.keys(V.eps[jobs[0]].log.cost_terms);
    const lanes = V.multi ? jobs.map((j) => ({ job: j, key: null })) : terms.map((t) => ({ job: jobs[0], key: t })).concat([{ job: jobs[0], key: null }]);
    const series = (ln) => { const T = V.eps[ln.job].log.cost_terms; return ln.key ? T[ln.key] : T[terms[0]].map((_, i) => (T[terms[0]][i] === null ? null : terms.reduce((a, t) => a + T[t][i], 0))); };
    const data = lanes.map(series);
    const label = (ln) => (V.multi ? "" : ln.key ? TERM(ln.key) : "total");
    // readout: the total, then the five largest terms at the cursor (compare: each job's total)
    const read = h("span", { class: "key", id: "v-costread" });
    const box = h("div", { class: "chart strip cost", "data-chart": "Cost" }, h("div", { class: "ch" }, h("span", { class: "lead" }, h("b", { text: "Cost" })), read));
    V.charts.append(box);
    // each lane on its own p5-p95 range, low blue through light gray to high red
    const scale = data.map((v) => { const a = v.filter((x) => x !== null && x !== undefined).sort((x, y) => x - y); if (!a.length) return [0, 1]; const lo = quantile(a, 0.05), hi = quantile(a, 0.95); return [lo, hi > lo ? hi : lo + 1e-6]; });
    const LANE = [[74, 111, 165], [226, 225, 221], [180, 30, 30]];
    const axis = { stroke: "#6F727B", grid: { show: false }, ticks: { show: false }, font: "10px -apple-system, BlinkMacSystemFont, sans-serif", size: 48, gap: 6 };
    const draw = (u) => {
      const ctx = u.ctx, { left, top, width, height } = u.bbox, lh = height / lanes.length;
      ctx.save();
      lanes.forEach((ln, k) => {
        const y = top + k * lh + lh * 0.15, hh = lh * 0.7, v = data[k], M = V.M[ln.job];
        ctx.fillStyle = "#F7F7F5"; ctx.fillRect(left, y, width, hh);
        if (V.multi) { ctx.fillStyle = V.jobs[ln.job].color; ctx.fillRect(left - 5 * devicePixelRatio, y, 3 * devicePixelRatio, hh); }
        V.T.forEach((t, n) => {
          const i = M[n], x = v[i];
          const xa = u.valToPos(t, "x", true), xb = n + 1 < V.T.length ? u.valToPos(V.T[n + 1], "x", true) : xa + devicePixelRatio;
          if (i < 0 || x === null || x === undefined) return;
          ctx.fillStyle = mix(LANE, (x - scale[k][0]) / (scale[k][1] - scale[k][0]));
          ctx.fillRect(xa, y, Math.max(xb - xa, devicePixelRatio), hh);
        });
      });
      ctx.restore();
    };
    const f = (x) => (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(1));
    const opts = {
      width: V.charts.clientWidth, height: 14 + lanes.length * 14, legend: { show: false },
      cursor: { sync: { key: "ep" }, y: false, points: { show: false }, drag: { x: false, y: false } },
      scales: { x: { time: false, range: () => [V.T[0], V.T[V.T.length - 1]] }, y: { range: () => [0, lanes.length] } },
      axes: [Object.assign({}, axis, { size: 6, values: (u, v) => v.map(() => "") }),
        Object.assign({}, axis, { splits: () => lanes.map((l, k) => lanes.length - k - 0.5), values: () => lanes.map(label) })],
      series: [{}],
      hooks: {
        draw: [draw],
        setCursor: [(u) => {
          if (holdPin(u)) return;
          const n = u.cursor.idx;
          const at = lanes.map((ln, k) => { const i = n === null || n === undefined ? -1 : V.M[ln.job][n]; return i >= 0 ? data[k][i] : null; });
          if (V.multi) { read.replaceChildren(...lanes.map((ln, k) => (at[k] === null || at[k] === undefined ? "" : h("span", { class: "key" }, h("i", { style: `background:${V.jobs[ln.job].color}` }), f(at[k]))))); return; }
          const tot = at[at.length - 1];
          if (tot === null || tot === undefined) { read.textContent = ""; return; }
          const top = lanes.slice(0, -1).map((ln, k) => [ln.key, at[k]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5);
          read.textContent = `total ${f(tot)}   ` + top.map(([t, x]) => `${TERM(t)} ${f(x)}`).join("   ");
        }],
      },
    };
    V.plots.push(new uPlot(opts, [V.T], box));
  }

  function chart(spec, last) {
    const width = V.charts.clientWidth;
    const vals = spec.series.map(() => h("span", { text: "" }));
    const keyed = spec.series.length > 1;
    const head = h("div", { class: "ch" }, h("span", { class: "lead" }, h("b", { text: spec.name }), " ", spec.unit),
      spec.note ? h("span", { text: spec.note }) : null, spec.series.map((s, i) => h("span", { class: "key" }, keyed ? h("i", { style: `background:${s.color}` }) : null, keyed && !V.multi ? s.label : null, vals[i])));
    const box = h("div", { class: "chart", "data-chart": spec.name });
    box.append(head);
    V.charts.append(box);
    const axis = { stroke: "#6F727B", grid: { stroke: "#EFEEEA", width: 1 }, ticks: { show: false }, font: "13px -apple-system, BlinkMacSystemFont, sans-serif", size: 48, gap: 6 };
    const opts = {
      width, height: last ? 108 : 84, legend: { show: false },
      cursor: { sync: { key: "ep" }, y: false, points: { size: 7, width: 0 }, drag: { x: false, y: false } },
      scales: { x: { time: false }, y: spec.span ? { range: (u, lo, hi) => { const m = Math.max(Math.abs(lo || 0), Math.abs(hi || 0), spec.span); return [-m, m]; } } : spec.floor ? { range: (u, lo, hi) => [0, Math.max(hi || 0, spec.floor)] } : {} },
      axes: [
        Object.assign({}, axis, { size: last ? 28 : 6, values: last ? (u, v) => v.map((t) => t + " s") : (u, v) => v.map(() => ""), grid: { show: false } }),
        Object.assign({}, axis, { space: 20, values: (u, v) => v.map((x) => +x.toFixed(2)) }),
      ],
      series: [{}].concat(spec.series.map((s) => ({ stroke: s.color, width: s.width || 1.5, dash: s.dash, spanGaps: true, points: { show: false } }))),
      hooks: {
        drawAxes: [bands()].concat(spec.peak ? [peakBand(spec.peak)] : [], spec.lines ? [limits(spec.lines)] : []),
        setCursor: [(u) => {
          if (holdPin(u)) return;
          const i = u.cursor.idx;
          spec.series.forEach((s, k) => {
            let v = null;
            if (i !== null && i !== undefined) { const m = V.M[s.job][i]; v = m >= 0 ? (V.eps[s.job].log[s.key] || [])[m] : null; }
            vals[k].textContent = v === null || v === undefined ? "" : v.toFixed(s.dp);
          });
          moveDot(i);
        }],
      },
    };
    const data = [V.T].concat(spec.series.map((s) => aligned(s.job, s.key)));
    const p = new uPlot(opts, data, box);
    V.plots.push(p);
  }

  function drawCharts() {
    for (const p of V.plots) p.destroy();
    V.plots = [];
    V.charts.replaceChildren();
    if (!V.T.length) return;
    let specs;
    const sx = V.ep.geom && V.ep.geom.peak_sx && +(V.ep.geom.peak_sx.rear * 100).toFixed(1);
    const lim = V.ep.lat_limit, lat = lim ? { span: lim.envelope * 1.15, lines: [{ v: lim.envelope, color: "#B42318", dash: [4, 3] }] } : {};
    if (V.multi) {
      const per = (key, dp) => V.eps.map((e, j) => (e && e.log.t ? { key, job: j, color: V.jobs[j].color, dp } : null)).filter(Boolean);
      specs = [{ name: "CTE", unit: "m", series: per("cte", 2) }, { name: "Steering", unit: "deg", series: per("wheel", 1) },
        { name: "Speed", unit: "m/s", series: per("speed", 2) }, Object.assign({ name: "Lateral", unit: "m/s²", series: per("ay_mps2", 1) }, lat),
        { name: "Slide", unit: "% of peak", floor: 120, lines: [{ v: 100, color: "#B42318", dash: [4, 3] }], series: per("slide_pct", 0) },
        { name: "Loc error", unit: "m", lines: [{ v: 2, color: "#B42318", dash: [4, 3] }], series: per("loc_err", 2) }];
    } else {
      specs = [
        { name: "CTE", unit: "m", series: [{ key: "cte", job: 0, label: "CTE", color: "#000", dp: 2 }] },
        { name: "Steering", unit: "deg", series: [{ key: "wheel", job: 0, label: "wheel", color: "#000", dp: 1 }, { key: "column", job: 0, label: "column", color: "#CFB991", dp: 1, width: 2 }] },
        { name: "Speed", unit: "m/s", series: [{ key: "speed", job: 0, label: "speed", color: "#000", dp: 2 }, { key: "cmd_speed", job: 0, label: "command", color: "#CFB991", dp: 2, width: 2 }] },
        Object.assign({ name: "Lateral", unit: "m/s²", series: [{ key: "ay_mps2", job: 0, label: "lateral", color: "#000", dp: 1 }] }, lat),
        { name: "Slide", unit: "% of peak", floor: 120, lines: [{ v: 100, color: "#B42318", dash: [4, 3] }], series: [{ key: "lat_front_pct", job: 0, label: "front", color: "#000", dp: 0 }, { key: "lat_rear_pct", job: 0, label: "rear", color: "#CFB991", dp: 0, width: 2 }] },
        { name: "Rear slip ratio", unit: "%", span: sx && sx * 1.5, lines: sx ? [{ v: sx, color: "#B42318", dash: [4, 3] }] : null, series: [{ key: "rear_slip_ratio", job: 0, label: "rear", color: "#000", dp: 1 }] },
        { name: "Loc error", unit: "m", note: V.ep.loc ? `RMS ${V.ep.loc.rms.toFixed(2)} · p95 ${V.ep.loc.p95.toFixed(2)}` : "", lines: [{ v: 2, color: "#B42318", dash: [4, 3] }], series: [{ key: "loc_err", job: 0, label: "loc error", color: "#000", dp: 2 }] },
      ];
    }
    const cost = V.eps.some((e) => e && e.log.cost_terms);
    specs.forEach((s, i) => { chart(s, i === specs.length - 1); if (!i) { strip(); if (cost) costStrip(); } });
    for (const p of V.plots) p.over.addEventListener("click", () => pinClick(p));
    V.pin = null;
    V.charts.classList.remove("pinned");
  }

  async function openAt(kart, cond, item) {
    const keys = keyFor(kart, cond, item);
    if (!keys) return;
    const cur = V.multi ? `${kart}|${cond}|${item}` : keys[0];
    api.current = cur; api.keys = keys;
    if (V.onOpen) V.onOpen(cur);
    const rs = await Promise.all(keys.map((k, i) => (k ? U.api(`/api/episodes/${encodeURIComponent(V.jobs[i].id)}/${k}`) : null)));
    if (api.current !== cur) return;
    const eps = rs.map((r) => (r && r.ok ? r.body : null));
    if (!eps.some(Boolean)) return;
    stop();
    V.eps = eps;
    V.ep = eps.find(Boolean);
    V.sel.sk.value = kart; V.sel.sc.value = cond; V.sel.si.value = item;
    refresh();
    const co = V.sel.cb.querySelector("option[value=cost]"), hasCost = eps.some((e) => e && e.log.cost);
    co.disabled = co.hidden = !hasCost;
    if (!hasCost && V.color === "cost") { V.color = V.sel.cb.value = ""; V.sel.cb.className = ""; }
    outcomeText();
    merge();
    drawMap();
    drawCharts();
    V.last = 0;
    if (V.T.length) { moveDot(0); requestAnimationFrame(() => { if (api.current === cur && V.last === 0) cursorAt(0); }); }   // the episode as shown starts at its first sample
  }

  function open(key) {
    const loc = V.idx && V.idx.byKey[key];
    if (loc) openAt(loc.kart, loc.cond, loc.item);
  }

  return Object.assign(api, { mount, update, open, openAt, openDefault, pinned: () => (pinned() ? V.pin : null), time: () => (V.cur >= 0 ? V.T[V.cur] : null) });
})();
