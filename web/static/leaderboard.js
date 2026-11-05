// Leaderboard: Artificial Analysis-style composite index, suite sub-scores, and engineering metrics per recipe.
(() => {
  const { h } = U;
  const M = U.METRICS;
  const mCol = (k, label, group, cls) => ({
    key: k, label: label || M[k].label, group: group || null, num: true, cls: cls || "",
    dir: M[k].low ? "asc" : "desc",
    fmt: (v) => (v === null || v === undefined ? "" : M[k].fmt(v) + (M[k].unit === "%" ? "%" : "")),
  });
  const COLS = [
    { key: "recipe", label: "Recipe", cls: "rec", dir: "asc" },
    { key: "notes", label: "Notes", cls: "notes", dir: "asc" },
    { key: "user", label: "User", cls: "user", dir: "asc" },
    mCol("composite_score", "Index", null, "score"),
    mCol("score_control", "Ctrl", "Suites"),
    mCol("score_localization", "Loc", "Suites"),
    mCol("score_overtake", "Pass", "Suites"),
    mCol("score_slide", "Slide", "Suites"),
    mCol("score_endurance", "50L", "Suites"),
    mCol("score_clips", "Clips", "Suites"),
    mCol("best_lap_s", "best", "Lap (s)"),
    mCol("lap_time_90", "90%", "Lap (s)"),
    mCol("lap_time_99", "99%", "Lap (s)"),
    mCol("cte_p50", "p50", "CTE"),
    mCol("cte_p90", "p90", "CTE"),
    mCol("cte_p99", "p99", "CTE"),
    mCol("loc_p50", "p50", "Loc Err"),
    mCol("loc_p95", "p95", "Loc Err"),
    mCol("loc_p99", "p99", "Loc Err"),
    mCol("overtake_rate", "Pass %", "Overtake"),
    mCol("min_clearance_m", "Clear (m)", "Overtake"),
    mCol("collision_rate", "Col %", "Overtake"),
    mCol("lap_improvement_s", "Lap Gain"),
    mCol("recovery_rate", "Recovery"),
    mCol("safety", "Safety"),
    mCol("osc_10", "Osc > 10"),
    mCol("laps_pct", "Laps"),
    { key: "errors", label: "Err", num: true, cls: "bad", dir: "desc", fmt: (v) => (v ? String(v) : "") },
    { key: "ran_at", label: "Ran", num: true, cls: "dim", dir: "desc", fmt: (v) => (v ? new Date(v).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "") },
    { key: "duration_s", label: "Duration", num: true, cls: "dim", dir: "asc", fmt: (s) => dur(s) },
  ];
  const S = { sort: "cte_p90", dir: "asc", tier: null };

  function dur(s) {
    if (s === null || s === undefined) return "";
    const m = Math.round(s / 60);
    return m < 60 ? m + "m" : Math.floor(m / 60) + "h " + String(m % 60).padStart(2, "0") + "m";
  }

  function th(c, attrs) {
    return h("th", Object.assign({
      class: (c.num ? "num " : "") + (S.sort === c.key ? "on" : ""), "data-key": c.key,
      "aria-sort": S.sort === c.key ? (S.dir === "asc" ? "ascending" : "descending") : null,
      onclick: () => { if (S.sort === c.key) S.dir = S.dir === "asc" ? "desc" : "asc"; else { S.sort = c.key; S.dir = c.dir; } load(); },
    }, attrs), c.label, h("span", { class: "ar", text: S.sort === c.key ? (S.dir === "asc" ? "↑" : "↓") : "" }));
  }

  function head() {
    const top = h("tr"), sub = h("tr");
    let i = 0;
    while (i < COLS.length) {
      const c = COLS[i];
      if (!c.group) { top.append(th(c, { rowspan: 2 })); i++; continue; }
      let span = 0;
      while (i + span < COLS.length && COLS[i + span].group === c.group) {
        sub.append(th(COLS[i + span]));
        span++;
      }
      top.append(h("th", { class: "grp", colspan: span }, h("span", { text: c.group })));
      i += span;
    }
    document.getElementById("lb-head").replaceChildren(top, sub);
  }

  let seq = 0;
  async function load() {
    head();
    const tok = ++seq;
    const r = await U.api(`/api/leaderboard?tier=${S.tier}&sort=${S.sort}&dir=${S.dir}`);
    if (tok !== seq) return;
    const body = document.getElementById("lb-body");
    const rows = r.ok ? r.body.slice().sort((a, b) => Number(Boolean(a.stale)) - Number(Boolean(b.stale))) : [];
    body.replaceChildren(...rows.map((row) => {
      const tr = h("tr", { "data-job": row.job_id, class: row.stale ? "old" : null, onclick: () => { location.href = "/?job=" + encodeURIComponent(row.job_id); } },
        COLS.map((c) => {
          const td = h("td", { class: (c.num ? "num " : "") + (c.cls || ""), text: c.fmt ? c.fmt(row[c.key]) : row[c.key] || "" });
          if (c.key === "notes" && row.notes) U.tip(td, () => (td.scrollWidth > td.clientWidth ? row.notes : ""));
          if (c.key === "recipe" && row.stale) { const m = h("i", { class: "oldm" }); U.tip(m, "older engine or tier"); td.append(m); }
          return td;
        }));
      return tr;
    }));
  }

  (async () => {
    const m = await U.api("/api/meta");
    const tiers = m.ok ? Object.keys(m.body.tiers) : ["dev"];
    const want = new URLSearchParams(location.search).get("tier");
    S.tier = tiers.includes(want) ? want : m.ok ? m.body.default_tier : "dev";
    const sel = document.getElementById("lb-tier");
    sel.replaceChildren(...tiers.map((t) => U.h("option", { value: t, text: t })));
    sel.value = S.tier;
    sel.addEventListener("change", () => {
      S.tier = sel.value;
      history.replaceState(null, "", "?tier=" + S.tier);
      load();
    });
    load();
  })();
})();
