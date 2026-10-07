// Compare: the job view for 2 to 5 jobs: headline tiles with differences against the first job, laps and clips matrices, overlaid viewer.
const Compare = (() => {
  const { h } = U;
  const C = { tab: "laps", clipCell: null, d: null };

  function diff(m, v, v1) {
    const d = U.METRICS[m];
    const scale = d.scale || 1;
    const dp = d.dp;
    const delta = (v - v1) * scale;
    const shown = Math.abs(delta).toFixed(dp);
    if (Number(shown) === 0) return h("span", { class: "d same", text: "±0" });
    const better = d.low ? delta < 0 : delta > 0;
    return h("span", { class: "d " + (better ? "better" : "worse"), text: (delta > 0 ? "+" : "−") + shown });
  }

  // green, red or gray for value v against the first job's v0
  function tone(v, v0, low, dp) {
    if (v === null || v === undefined || v0 === null || v0 === undefined) return "";
    if (Number(Math.abs(v - v0).toFixed(dp)) === 0) return "same";
    return (low ? v < v0 : v > v0) ? "better" : "worse";
  }

  const tick = (i) => h("i", { class: "jt", style: `background:${U.COLORS[i]}` });

  function legend(d) {
    return h("div", { class: "clegend" }, d.jobs.map((j, i) => {
      const el = h("span", { class: "cj" }, tick(i), h("span", { class: "n", text: j.label }));
      U.tip(el, `${j.recipe}  ${j.tier}  ${j.user}`);
      return el;
    }));
  }

  function tiles(d) {
    const out = h("div", { class: "ctiles", id: "ctiles" });
    for (const m of d.metrics) {
      const def = U.METRICS[m.name];
      const el = h("div", { class: "ctile", "data-metric": m.name }, h("div", { class: "l" }, def.label, h("small", { text: " " + def.unit })),
        m.values.map((v, i) => h("div", { class: "cv", title: m.worst_kart[i] ? U.kart(m.worst_kart[i]) : null }, tick(i), h("span", { class: "v", text: def.fmt(v) }),
          i ? diff(m.name, v, m.values[0]) : null)));
      if (m.name === "safety" && d.unsafe) U.tip(el, U.unsafeText((k) => d.unsafe[k]));
      out.append(el);
    }
    return out;
  }

  function lapCells(cells, c, k, l) {
    const loc = `${k}|${c}|L:${l}`;
    const v0 = cells[0].status === "ok" ? cells[0].cte_p90 : null;
    const el = h("div", { class: "cell multi", "data-loc": loc }, cells.map((x, i) => {
      if (x.status !== "ok") return h("span", { class: "jv bad" }, tick(i), U.CAUSE[x.status === "error" ? "error" : x.cause] || "fail");
      return h("span", { class: "jv " + (i ? tone(x.cte_p90, v0, true, 2) : "") }, tick(i), cteText(x.cte_p90));
    }));
    U.tip(el, C.d.jobs.map((j, i) => `${j.label}  ${cells[i].status === "ok" ? U.num(cells[i].cte_p90, 2) + " m" : U.human(cells[i].cause || cells[i].status)}`).join("\n"));
    el.addEventListener("click", () => Viewer.openAt(k, c, "L:" + l));
    S.cells[loc] = el;
    return el;
  }

  function clipsView(d) {
    const wrap = h("div");
    const classes = CLASSES.filter((c) => d.clips[c]).concat(Object.keys(d.clips).filter((c) => !CLASSES.includes(c)));
    const karts = [...new Set(classes.flatMap((c) => Object.keys(d.clips[c])))].sort();
    const g = h("div", { class: "cm" });
    g.append(h("div"));
    karts.forEach((k) => g.append(h("div", { class: "kh", text: U.kart(k) })));
    for (const c of classes) {
      g.append(h("div", { class: "g", text: U.human(c) }));
      for (const k of karts) {
        const cells = d.clips[c][k];
        if (!cells) { g.append(h("div", { class: "ccell na", text: "–" })); continue; }
        const on = C.clipCell && C.clipCell[0] === c && C.clipCell[1] === k;
        const r0 = cells[0].recovery_rate;
        const el = h("div", { class: "ccell multi" + (on ? " on" : ""), "data-cls": c, "data-kart": k }, cells.map((x, i) =>
          h("div", { class: "jv" }, tick(i), h("span", { class: "r " + (i ? tone(x.recovery_rate, r0, false, 2) : ""), text: x.recovery_rate === null ? "–" : U.pct(x.recovery_rate) + "%" }),
            h("span", { class: "m", text: x.median_recovery_s !== null ? U.num(x.median_recovery_s, 1) + " s" : "" }))));
        el.addEventListener("click", () => { C.clipCell = on ? null : [c, k]; wrap.replaceWith(clipsView(d)); markCell(Viewer.current); });
        g.append(el);
      }
    }
    wrap.append(g);
    const cell = C.clipCell && d.clips[C.clipCell[0]] && d.clips[C.clipCell[0]][C.clipCell[1]];
    if (cell) {
      const [, k] = C.clipCell;
      const ids = [...new Set(cell.flatMap((x) => x.episodes.map((e) => e.clip_id)))].sort();
      const tb = h("tbody");
      for (const id of ids) {
        const loc = `${k}|nominal|C:${id}`;
        const tr = h("tr", { "data-loc": loc, onclick: () => Viewer.openAt(k, "nominal", "C:" + id) }, h("td", { text: id }), cell.map((x) => {
          const e = x.episodes.find((y) => y.clip_id === id);
          if (!e) return h("td", { class: "miss", text: "–" });
          if (e.context_valid === false) return h("td", { class: "inv", text: "invalid" });
          return h("td", { class: e.recovered ? "" : "no", text: e.recovered ? U.num(e.recovery_s, 1) + " s" : "no" });
        }));
        S.cells[loc] = tr;
        tb.append(tr);
      }
      wrap.append(h("table", { class: "clist cmp" }, h("thead", {}, h("tr", {}, h("th", { text: "Clip" }), cell.map((x, i) => h("th", {}, tick(i))))), tb));
    }
    markCell(Viewer.current);
    return wrap;
  }

  function breakdown(box, d) {
    S.cells = {};
    const hasLaps = Object.keys(d.laps).length > 0, hasClips = Object.keys(d.clips).length > 0;
    if (!hasLaps && !hasClips) { box.replaceChildren(); return; }
    if (C.tab === "clips" && !hasClips) C.tab = "laps";
    if (C.tab === "laps" && !hasLaps) C.tab = "clips";
    const tabs = h("div", { class: "tabs", role: "tablist" });
    for (const [t, label, ok] of [["laps", "Laps", hasLaps], ["clips", "Clips", hasClips]]) {
      if (!ok) continue;
      tabs.append(h("button", { type: "button", role: "tab", class: C.tab === t ? "on" : "", "data-tab": t, text: label,
        onclick: () => { C.tab = t; breakdown(box, d); } }));
    }
    box.replaceChildren(tabs, C.tab === "laps" ? renderLaps(d, lapCells) : clipsView(d));
    markCell(Viewer.current);
  }

  let seq = 0;
  async function render(el, ids, conds, sl, filters) {
    const tok = ++seq;
    const r = await U.api("/api/compare?ids=" + ids.map(encodeURIComponent).join(",") + U.sliceQuery(sl));
    if (!r.ok || tok !== seq) return;
    const d = r.body;
    C.d = d;
    const box = h("div", { id: "c-break" }), vbox = h("div", { id: "c-viewer" });
    el.replaceChildren(h("div", { class: "pad" }, filters(d.options), legend(d), tiles(d), box, vbox));
    el.scrollTop = 0;
    Viewer.mount(vbox, d.jobs.map((j, i) => ({ id: j.id, color: U.COLORS[i] })), d, conds, markCell);
    breakdown(box, d);
    Viewer.openDefault();
  }

  return { render };
})();
