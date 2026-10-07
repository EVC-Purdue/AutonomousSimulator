// Leaderboard: one row per recipe, every column sorts on the server, row click opens the job.
(() => {
  const { h } = U;
  const M = U.METRICS;
  const COLS = [
    { key: "recipe", label: "Recipe", cls: "rec", dir: "asc" },
    { key: "notes", label: "Notes", cls: "notes", dir: "asc" },
    { key: "user", label: "User", cls: "user", dir: "asc" },
    ...["cte_p50", "cte_p90", "cte_p99", "recovery_rate", "safety", "osc_10", "laps_pct"].map((k) => ({
      key: k, label: k.startsWith("cte_") ? k.slice(4) : M[k].label, group: k.startsWith("cte_"), num: true,
      dir: M[k].low ? "asc" : "desc", fmt: (v) => (v === null || v === undefined ? "" : M[k].fmt(v) + (M[k].unit === "%" ? "%" : "")) })),
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
    for (const c of COLS) {
      if (!c.group) top.append(th(c, { rowspan: 2 }));
      else {
        if (c.key === "cte_p50") top.append(h("th", { class: "grp", colspan: 3 }, h("span", { text: "CTE" })));
        sub.append(th(c));
      }
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
    body.replaceChildren(...(r.ok ? r.body : []).map((row) => {
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
