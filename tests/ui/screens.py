"""Screenshots of the 9 frontend states at 1440x900 and 1024x768, and a click-through of every control.

Usage: .venv/bin/python tests/ui/screens.py [--fixture DIR]   (DIR: a prebuilt fixture results dir, copied first)
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
import threading

import pytest
from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import fixture_data  # noqa: E402

from runner import local, store  # noqa: E402

OUT = os.path.join(store.ROOT, "results", "screens")
PORT = int(os.environ.get("UI_PORT", 5050))
URL = f"http://127.0.0.1:{PORT}"
SIZES = [(1440, 900), (1024, 768)]


def row(page, recipe):
    return page.locator("#rows .row", has=page.locator(".name", has_text=recipe)).first


def check(page, recipe):
    r = row(page, recipe)
    r.hover()
    r.locator("input[type=checkbox]").click()


def shoot(page, name, w):
    path = os.path.join(OUT, f"{name}-{w}.png")
    page.screenshot(path=path)
    print(path)


def hover_turn(page):
    """Hovers the first chart where the front wheels turn most, and returns that wheel angle."""
    jid = re.search(r"job=([^&]+)", page.url).group(1)
    ep = page.request.get(f"{URL}/api/episodes/{jid}/{page.evaluate('Viewer.current')}").json()
    t, wl = ep["log"]["t"], ep["log"]["wheel_l"]
    i = max((k for k in range(len(t)) if t[k] >= 0), key=lambda k: abs(wl[k]))
    box = page.locator(".chart").first.locator(".u-over").bounding_box()
    page.mouse.move(box["x"] + box["width"] * (t[i] - t[0]) / (t[-1] - t[0]), box["y"] + box["height"] / 2)
    return wl[i]


def check_pin(page):
    """Click a chart to pin the cursor; hover no longer moves it; arrows step one sample; holding sweeps at 0.25x; click or Escape unpins."""
    box = page.locator(".chart").first.locator(".u-over").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.4, box["y"] + box["height"] / 2)
    page.mouse.click(box["x"] + box["width"] * 0.4, box["y"] + box["height"] / 2)
    i = page.evaluate("Viewer.pinned()")
    assert i is not None and page.locator("#v-charts.pinned").count() == 1
    dot = page.locator("#v-dot").get_attribute("transform")
    page.mouse.move(box["x"] + box["width"] * 0.7, box["y"] + box["height"] / 2)
    page.wait_for_timeout(150)
    assert page.evaluate("Viewer.pinned()") == i and page.locator("#v-dot").get_attribute("transform") == dot
    page.mouse.move(10, 10)
    page.keyboard.press("ArrowRight")
    assert page.evaluate("Viewer.pinned()") == i + 1
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowLeft")
    assert page.evaluate("Viewer.pinned()") == i - 1
    t0 = page.evaluate("Viewer.time()")
    page.keyboard.down("ArrowRight")
    page.wait_for_timeout(2250)
    page.keyboard.up("ArrowRight")
    dt = page.evaluate("Viewer.time()") - t0
    assert 0.35 < dt < 0.65, dt   # 2 s past the 0.25 s grace at 0.25x: about 0.5 s of sim
    page.keyboard.press("Escape")
    assert page.evaluate("Viewer.pinned()") is None and page.locator("#v-charts.pinned").count() == 0
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2)
    assert page.evaluate("Viewer.pinned()") is not None
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2)
    assert page.evaluate("Viewer.pinned()") is None


def check_play_and_clip_plan(page):
    """Play starts at the episode's first sample and advances; a clip shows its rollout after the handoff, none before."""
    page.mouse.move(10, 10)
    cur = page.locator("#v-item").input_value()
    other = [v for v in page.locator("#v-item optgroup[label=Lines] option").evaluate_all("os => os.map(o => o.value)") if v != cur][0]
    page.select_option("#v-item", other)
    page.wait_for_timeout(600)
    t0 = page.evaluate("Viewer.time()")
    jid = re.search(r"job=([^&]+)", page.url).group(1)
    ep = page.request.get(f"{URL}/api/episodes/{jid}/{page.evaluate('Viewer.current')}").json()
    assert t0 == ep["log"]["t"][0], (t0, ep["log"]["t"][:3], page.evaluate("Viewer.current"), other)
    page.click("#v-play")
    page.wait_for_timeout(700)
    page.click("#v-play")
    assert page.evaluate("Viewer.time()") > t0 + 0.2
    page.mouse.move(10, 10)
    clip = page.locator("#v-item optgroup:not([label=Lines]) option").first.get_attribute("value")
    page.select_option("#v-item", clip)
    page.wait_for_timeout(600)
    box = page.locator(".chart").first.locator(".u-over").bounding_box()
    page.mouse.move(box["x"] + 2, box["y"] + box["height"] / 2)
    expect(page.locator("#v-map [data-mark=plan]")).to_have_attribute("visibility", "hidden")   # context: MANUAL, no plan
    page.mouse.move(box["x"] + box["width"] * 0.9, box["y"] + box["height"] / 2)
    expect(page.locator("#v-map [data-mark=plan]")).to_have_attribute("visibility", "visible")
    expect(page.locator("#v-inset [data-mark=plan]")).to_have_attribute("visibility", "visible")
    page.mouse.move(10, 10)
    page.select_option("#v-item", cur)
    page.wait_for_timeout(400)


def run(page, w, submit):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    # 1 jobs list with the running job
    page.goto(URL + "/")
    expect(page.locator("#rows .row")).to_have_count(7)
    expect(page.locator("#rows .row").first.locator(".bar")).to_be_visible()
    expect(page.locator("#rows .row").first.locator(".when")).to_contain_text("ETA")
    expect(page.locator(".tiles .tile").first).to_be_visible()
    page.mouse.move(w - 10, 10)
    shoot(page, "01-jobs", w)
    page.screenshot(path=os.path.join(OUT, f"20-topbar-{w}.png"), clip={"x": 0, "y": 0, "width": w, "height": 48})
    expect(page.locator("header.top .mark")).to_have_count(0)

    # filters: tier chips, where select and text combine, persist in the URL, and the running job hides when it does not match
    js = page.request.get(f"{URL}/api/jobs").json()
    n_smoke = sum(j["tier"] == "smoke" for j in js)
    page.click('#ftier [data-v="smoke"]')
    expect(page.locator("#rows .row")).to_have_count(n_smoke)
    assert "tier=smoke" in page.url
    page.fill("#filter", "alex")
    expect(page.locator("#rows .row")).to_have_count(sum(j["tier"] == "smoke" and "alex" in j["user"] for j in js))
    expect(page.locator("#rows .row.sel")).to_have_count(1)
    page.select_option("#fwhere", "gautschi")
    expect(page.locator("#rows .row")).to_have_count(0)
    expect(page.locator(".empty .btn")).to_have_text("New job")
    page.reload()
    expect(page.locator('#ftier [data-v="smoke"]')).to_have_class("on")
    expect(page.locator("#fwhere")).to_have_value("gautschi")
    expect(page.locator("#filter")).to_have_value("alex")
    page.goto(URL + "/")
    expect(page.locator("#rows .row")).to_have_count(len(js))
    expect(page.locator(".tiles .tile").first).to_be_visible()

    # build-failed row shows the build log
    row(page, "mpc-broken").click()
    expect(page.locator("#build-log")).to_contain_text("SyntaxError")
    expect(row(page, "mpc-broken").locator(".when")).to_have_text("build failed")

    # 2 new job with invalid params
    page.click("#new-job")
    expect(page.locator("#panel")).to_be_visible()
    expect(page.locator("#f-ref")).to_have_value(fixture_data.WT)
    page.fill("#f-recipe", "mpc-h50")
    page.fill("#f-params", '{"pathfinder_node": {"planner": "mpc", "horizon": 50,}}')
    page.locator("#f-params").dispatch_event("input")
    expect(page.locator('.field[data-f="params"]')).to_have_class("field err")
    expect(page.locator('.field[data-f="params"] .e')).to_be_visible()
    page.click('#f-tier [data-v="dev"]')
    expect(page.locator('#f-where option[value="modal"]')).to_be_disabled()
    expect(page.locator("#est-n")).to_contain_text("195 episodes")
    page.click('#f-tier [data-v="smoke"]')
    expect(page.locator('#f-where option[value="modal"]')).to_be_enabled()
    page.fill("#f-user", "shay")
    page.fill("#f-notes", "Horizon 50")
    shoot(page, "02-new-job-invalid", w)
    page.click("#panel-close")
    expect(page.locator("#panel")).to_be_hidden()

    # 3 finished dev result, laps tab
    row(page, "mpc-h60").click()
    expect(page.locator(".lm .cell")).to_have_count(171)
    expect(page.locator(".tiles .tile")).to_have_count(8)
    expect(page.locator("#v-item")).to_have_value("L:center")
    page.locator("#main").evaluate("m => m.scrollTop = 0")
    page.mouse.move(w - 10, 10)
    shoot(page, "03-dev-laps", w)
    cell = page.locator(".lm .cell:not(.fail):not(.none)").nth(40)
    key = cell.get_attribute("data-key")
    cell.click()
    expect(cell).to_have_class("cell on")
    expect(page.locator("#v-out")).not_to_be_empty()
    page.select_option("#v-kart", "kart3")
    expect(page.locator("#v-kart")).to_have_value("kart3")
    expect(page.locator(f'.lm .cell[data-key="{key}"]')).not_to_have_class("cell on")
    page.select_option("#v-cond", "grip_high")
    expect(page.locator(".lm .cell.on")).to_have_count(1)

    # 22 viewer mid-turn on a lap: top-down inset with both front wheels steered, kart glyph on the map
    page.select_option("#v-cond", "nominal")
    page.wait_for_timeout(500)
    page.locator("#main").evaluate("m => m.scrollTop = m.querySelector('.viewer').offsetTop - 16")
    assert abs(hover_turn(page)) > 2
    expect(page.locator("#v-dot")).to_have_attribute("visibility", "visible")
    expect(page.locator("#v-inset")).to_be_visible()
    shoot(page, "22-viewer-lap-turn", w)
    page.select_option("#v-speed", "4")
    expect(page.locator("#v-speed")).to_have_value("4")
    page.click("#v-play")
    expect(page.locator("#v-play")).to_have_attribute("aria-label", "Pause")
    page.wait_for_timeout(400)
    page.click("#v-play")
    expect(page.locator("#v-play")).to_have_attribute("aria-label", "Play")
    check_pin(page)
    check_play_and_clip_plan(page)

    # 4 clips tab with a cell list open
    page.click('.tabs [data-tab="clips"]')
    expect(page.locator(".cm .ccell")).to_have_count(12)
    page.locator('.ccell[data-cls="oscillation"][data-kart="kart2"]').click()
    expect(page.locator(".clist tbody tr")).to_have_count(2)
    expect(page.locator(".clist tbody tr.inv")).to_have_count(1)
    expect(page.locator(".clist tbody tr.inv td").nth(1)).to_have_text("invalid")
    page.locator(".tabs").scroll_into_view_if_needed()
    page.locator("#main").evaluate("m => m.scrollTop = m.querySelector('.tabs').offsetTop - 24")
    shoot(page, "04-clips", w)

    # 5 viewer with a clip selected and the cursor mid-chart
    clip_row = page.locator(".clist tbody tr:not(.inv)").first
    clip_row.click()
    expect(clip_row).to_have_class("on")
    expect(page.locator("#v-item")).to_have_value(f"C:{clip_row.locator('td').first.inner_text()}")
    expect(page.locator("#v-cond")).to_have_value("nominal")
    page.locator("#main").evaluate("m => m.scrollTop = m.querySelector('.viewer').offsetTop - 16")
    page.locator(".chart").first.scroll_into_view_if_needed()
    box = page.locator(".chart").first.locator(".u-over").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.55, box["y"] + box["height"] / 2)
    expect(page.locator("#v-dot")).to_have_attribute("visibility", "visible")
    shoot(page, "05-viewer-clip", w)
    hover_turn(page)
    expect(page.locator("#v-dot")).to_have_attribute("visibility", "visible")
    shoot(page, "23-viewer-clip-turn", w)
    page.click('.tabs [data-tab="laps"]')
    expect(page.locator(".lm")).to_be_visible()

    # 10 laps matrix with failed cells, tooltip on hover
    row(page, "pure-pursuit").click()
    expect(page.locator(".head h1")).to_have_text("pure-pursuit")
    fail = page.locator(".lm .cell.fail")
    expect(fail.first).to_be_visible()
    words = set(fail.all_inner_texts())
    assert words and words <= {"spin", "off", "loc", "cmd", "err"}, words
    fail.last.scroll_into_view_if_needed()
    fail.last.hover()
    expect(page.locator("#tip")).to_be_visible()
    expect(page.locator("#tip")).to_contain_text(" at ")
    shoot(page, "10-laps-failures", w)
    fail.last.click()
    expect(page.locator("#v-out .bad")).to_contain_text(" at ")
    expect(page.locator("#v-map [data-mark=failure]")).to_have_count(1)

    # 14-17 slices: averaged default, kart 2, kart 1 x line1 from a shared URL, kart 1 x grip low
    kart, cond = '.filt [data-f="kart"]', '.filt [data-f="condition"]'
    row(page, "mpc-h60").click()
    expect(page.locator(".filt select")).to_have_count(4)
    expect(page.locator(".lm .cell")).to_have_count(171)
    expect(page.locator(".tiles .tile .k").first).to_have_text("\u00a0")
    page.mouse.move(w - 10, 10)
    shoot(page, "14-slice-all", w)
    page.select_option(kart, "kart2")
    expect(page).to_have_url(re.compile(r"kart=kart2"))
    expect(page.locator(".lm .kh")).to_have_text(["kart 2"])
    expect(page.locator(".tiles .tile .k").first).to_have_text("kart 2")
    page.mouse.move(w - 10, 10)
    shoot(page, "15-slice-kart2", w)
    jid = row(page, "mpc-h60").get_attribute("data-id")
    page.goto(URL + f"/?job={jid}&kart=kart1&line=line1")
    expect(page.locator('.filt [data-f="line"]')).to_have_value("line1")
    expect(page.locator(".lm .cell")).to_have_count(19)
    expect(page.locator('.tabs [data-tab="clips"]')).to_have_count(0)
    expect(page.locator('.filt [data-f="cls"] option:disabled')).to_have_count(4)
    expect(page.locator(".tiles .tile .k").first).to_have_text("kart 1 · line1")
    page.mouse.move(w - 10, 10)
    shoot(page, "16-slice-kart1-line1", w)
    page.select_option('.filt [data-f="line"]', "")
    page.select_option(cond, "grip_low")
    expect(page.locator(".lm .cell")).to_have_count(3)
    expect(page.locator(".tiles .tile .k").first).to_have_text("kart 1 · grip low")
    page.mouse.move(w - 10, 10)
    shoot(page, "17-slice-kart1-grip-low", w)
    page.select_option(cond, "")
    page.select_option(kart, "")
    expect(page).not_to_have_url(re.compile(r"kart="))

    # 26-27 color by: a lap by speed, a clip by signed CTE
    expect(page.locator(".lm .cell")).to_have_count(171)
    page.wait_for_load_state("networkidle")
    page.select_option("#v-item", "L:center")
    page.select_option("#v-color", "speed")
    expect(page).to_have_url(re.compile(r"color=speed"))
    expect(page.locator("#v-legend")).to_contain_text("m/s")
    page.locator("#main").evaluate("m => m.scrollTop = m.querySelector('.viewer').offsetTop - 16")
    page.mouse.move(w - 10, 10)
    shoot(page, "26-viewer-lap-speed", w)
    page.click('.tabs [data-tab="clips"]')
    page.locator('.ccell[data-cls="oscillation"][data-kart="kart1"]').click()
    page.locator(".clist tbody tr:not(.inv)").first.click()
    page.select_option("#v-color", "cte")
    expect(page.locator("#v-legend")).to_contain_text("m")
    page.locator("#main").evaluate("m => m.scrollTop = m.querySelector('.viewer').offsetTop - 16")
    page.mouse.move(w - 10, 10)
    shoot(page, "27-viewer-clip-cte", w)
    page.select_option("#v-color", "")
    expect(page).not_to_have_url(re.compile(r"color="))
    page.click('.tabs [data-tab="laps"]')

    # 6 compare, 2 jobs
    check(page, "mpc-h60")
    expect(page.locator("#compare")).to_be_hidden()
    check(page, "pure-pursuit")
    expect(page.locator("#compare")).to_have_text("Compare 2")
    page.click("#compare")
    expect(page.locator(".clegend .cj")).to_have_count(2)
    expect(page.locator('#ctiles .ctile[data-metric="recovery_rate"] .cv')).to_have_count(2)
    expect(page.locator(".filt select")).to_have_count(4)
    page.mouse.move(w - 10, 10)
    shoot(page, "06-compare-2", w)

    # 24 compare laps matrix, then a cell opens both episodes overlaid
    expect(page.locator(".lm .cell.multi")).to_have_count(171)
    page.locator("#main").evaluate("m => m.scrollTop = m.querySelector('.tabs').offsetTop - 24")
    page.mouse.move(w - 10, 10)
    shoot(page, "24-compare-laps", w)
    cell = page.locator('.lm .cell.multi[data-loc="kart2|grip_low|L:line1"]')
    cell.click()
    expect(cell).to_have_class(re.compile(r"\bon\b"))
    expect(page.locator("#v-kart")).to_have_value("kart2")
    expect(page.locator("#v-map polyline[data-job]:not([data-mark])")).to_have_count(2)
    expect(page.locator("#v-map .kdot")).to_have_count(2)
    expect(page.locator("#v-inset g[data-job]:not([data-mark])")).to_have_count(2)
    expect(page.locator(".chart").first.locator(".ch .key")).to_have_count(2)
    page.click("#v-play")
    expect(page.locator("#v-play")).to_have_attribute("aria-label", "Pause")
    page.wait_for_timeout(300)
    page.click("#v-play")

    # 25 compare clip viewer overlay
    page.click('.tabs [data-tab="clips"]')
    page.locator('.ccell[data-cls="oscillation"][data-kart="kart2"]').click()
    expect(page.locator(".clist.cmp thead th")).to_have_count(3)
    page.locator(".clist.cmp tbody tr").last.click()
    expect(page.locator("#v-item")).to_have_value(re.compile(r"^C:"))
    expect(page.locator("#v-map polyline[data-job]:not([data-mark])")).to_have_count(2)
    page.locator("#main").evaluate("m => m.scrollTop = m.querySelector('.viewer').offsetTop - 16")
    box = page.locator(".chart").first.locator(".u-over").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.55, box["y"] + box["height"] / 2)
    expect(page.locator("#v-dot")).to_have_attribute("visibility", "visible")
    shoot(page, "25-compare-clip-viewer", w)
    page.click('.tabs [data-tab="laps"]')

    # 18 compare with a slice
    page.select_option(kart, "kart1")
    page.select_option('.filt [data-f="line"]', "center")
    expect(page).to_have_url(re.compile(r"compare=.*kart=kart1.*line=center"))
    expect(page.locator('#ctiles .ctile[data-metric="recovery_rate"]')).to_have_count(0)
    expect(page.locator(".lm .cell.multi")).to_have_count(19)
    page.mouse.move(w - 10, 10)
    shoot(page, "18-compare-slice", w)

    # 7 compare, 5 jobs (smoke and dev mixed), with and without the slice
    for r in ("stack-default", "mpc-h40", "pp-lookahead-3"):
        check(page, r)
    expect(page.locator("#compare")).to_have_text("Compare 5")
    expect(page.locator(".clegend .cj")).to_have_count(5)
    expect(page.locator(".lm .cell.multi")).to_have_count(1)
    expect(page.locator("#v-map polyline[data-job]:not([data-mark])")).to_have_count(5)
    page.mouse.move(w - 10, 10)
    shoot(page, "28-compare-5-slice", w)
    page.select_option('.filt [data-f="line"]', "")
    page.select_option(kart, "")
    expect(page.locator('#ctiles .ctile[data-metric="recovery_rate"] .cv')).to_have_count(5)   # smoke runs clips too
    expect(row(page, "mpc-h80").locator("input[type=checkbox]")).to_be_disabled()
    expect(page.locator(".lm .cell.multi")).to_have_count(3)
    page.mouse.move(w - 10, 10)
    shoot(page, "07-compare-5", w)
    page.locator(".lm .cell.multi").first.click()
    page.locator("#main").evaluate("m => m.scrollTop = m.querySelector('.viewer').offsetTop - 16")
    expect(page.locator("#v-map polyline[data-job]:not([data-mark])")).to_have_count(5)
    page.select_option("#v-color", "speed")
    expect(page.locator("#v-legend")).to_contain_text("m/s")
    page.mouse.move(w - 10, 10)
    shoot(page, "29-compare-5-viewer-speed", w)
    page.select_option("#v-color", "")
    page.click("#clear")
    expect(page.locator("#compare")).to_be_hidden()
    expect(page.locator(".tiles")).to_be_visible()

    # 8 leaderboard by CTE p90
    page.click("nav >> text=Leaderboard")
    expect(page.locator("#lb-body tr")).to_have_count(3)
    expect(page.locator("#lb-tier")).to_have_value("dev")
    expect(page.locator('th[data-key="cte_p90"]')).to_have_attribute("aria-sort", "ascending")
    vals = [float(v) for v in page.locator("#lb-body tr td:nth-child(5)").all_inner_texts()]
    assert vals == sorted(vals), vals
    page.mouse.move(w - 10, 10)
    shoot(page, "08-leaderboard-cte", w)

    # 9 leaderboard by Recovery descending
    page.click('th[data-key="recovery_rate"]')
    expect(page.locator('th[data-key="recovery_rate"]')).to_have_attribute("aria-sort", "descending")
    page.wait_for_function("""() => { const v = [...document.querySelectorAll('#lb-body tr td:nth-child(7)')]
      .map((t) => parseFloat(t.textContent)); return v.length === 3 && v.every((x, i) => !i || v[i - 1] >= x); }""",
                           timeout=5000)
    vals = [float(v.rstrip("%")) for v in page.locator("#lb-body tr td:nth-child(7)").all_inner_texts()]
    assert vals == sorted(vals, reverse=True), vals
    page.mouse.move(w - 10, 10)
    shoot(page, "09-leaderboard-recovery", w)
    page.click('th[data-key="recovery_rate"]')
    expect(page.locator('th[data-key="recovery_rate"]')).to_have_attribute("aria-sort", "ascending")
    page.select_option("#lb-tier", "smoke")
    expect(page).to_have_url(re.compile(r"tier=smoke"))
    expect(page.locator("#lb-body tr")).to_have_count(2)
    page.mouse.move(w - 10, 10)
    shoot(page, "19-leaderboard-smoke", w)
    page.select_option("#lb-tier", "dev")
    expect(page.locator("#lb-body tr")).to_have_count(3)
    page.click('th[data-key="recipe"]')
    expect(page.locator("#lb-body tr td:first-child").first).to_have_text("mpc-h60")
    page.locator("#lb-body tr").first.click()
    page.wait_for_url("**/?job=*")
    expect(page.locator(".head h1")).to_have_text("mpc-h60")

    if submit:   # submit with valid input against the fake runner
        page.click("#new-job")
        page.fill("#f-recipe", "mpc-h50")
        page.fill("#f-params", '{"pathfinder_node": {"planner": "mpc", "mpc.horizon_steps": 50}}')
        page.locator("#f-params").dispatch_event("input")
        page.fill("#f-user", "shay")
        page.fill("#f-notes", "Horizon 50")
        page.click("#submit")
        expect(page.locator("#panel")).to_be_hidden()
        expect(row(page, "mpc-h50")).to_be_visible()
        expect(row(page, "mpc-h50").locator(".when")).to_have_text("now", timeout=30000)
        page.reload()
        page.click("#new-job")
        expect(page.locator("#f-user")).to_have_value("shay")
        page.click("#panel-close")
        row(page, "mpc-h50").click()   # delete: trash, inline confirm for 4 s, then gone
        expect(page.locator(".head h1")).to_have_text("mpc-h50")
        page.click('.head .del[aria-label="Delete job"]')
        expect(page.locator(".head .del")).to_have_text("Delete")
        page.mouse.move(w - 10, 10)
        shoot(page, "22-delete-confirm", w)
        expect(page.locator('.head .del[aria-label="Delete job"]')).to_be_visible(timeout=6000)
        page.click('.head .del[aria-label="Delete job"]')
        page.click(".head .del.red")
        expect(row(page, "mpc-h50")).to_have_count(0)
        expect(page.locator(".empty .btn")).to_have_text("New job")
        expect(page.locator("#rows .row.sel")).to_have_count(0)
        assert "job=" not in page.url
    assert not errors, errors


def empty(page, w):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL + "/")
    expect(page.locator(".empty .btn")).to_have_text("New job")
    expect(page.locator("#rows .row")).to_have_count(0)
    shoot(page, "12-empty-jobs", w)
    page.click(".empty .btn")
    expect(page.locator("#panel")).to_be_visible()
    page.goto(URL + "/leaderboard")
    expect(page.locator("#lb-head th[data-key]")).to_have_count(13)
    expect(page.locator("#lb-body tr")).to_have_count(0)
    shoot(page, "13-empty-leaderboard", w)
    page.set_content(f'''<body style="margin:0;background:#DFE1E5;font:13px -apple-system,sans-serif">
      <div style="margin:8px 0 0 8px;width:240px;height:34px;background:#fff;border-radius:8px 8px 0 0;display:flex;
      align-items:center;gap:8px;padding:0 12px"><img src="{URL}/favicon.ico" width="16" height="16">EVC Sim</div>
      <img src="{URL}/favicon.ico" width="32" style="margin:12px"><img src="{URL}/favicon.ico" width="64"></body>''')
    page.screenshot(path=os.path.join(OUT, f"21-tab-{w}.png"), clip={"x": 0, "y": 0, "width": 280, "height": 130})
    assert not errors, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="evc-ui-")
    root = os.path.join(tmp, "results")
    with pytest.MonkeyPatch.context() as mp:
        if args.fixture and os.path.isdir(os.path.join(args.fixture, "jobs")):
            shutil.copytree(args.fixture, root)
            mp.setattr(store, "RESULTS", root)
            fixture_data.install(mp)
        else:
            fixture_data.build(root, mp)
        from web.app import create_app
        server = make_server("127.0.0.1", PORT, create_app(), threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        mp.setattr(local.Runner, "recover", lambda self: None)   # keep the fixture's running job running
        runner = local.Runner(workers=2).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                for i, (w, ht) in enumerate(SIZES):
                    page = browser.new_page(viewport={"width": w, "height": ht}, device_scale_factor=2)
                    run(page, w, submit=i == len(SIZES) - 1)
                    page.close()
                mp.setattr(store, "RESULTS", os.path.join(tmp, "empty"))
                for w, ht in SIZES:
                    page = browser.new_page(viewport={"width": w, "height": ht}, device_scale_factor=2)
                    empty(page, w)
                    page.close()
                browser.close()
        finally:
            runner.stop()
            server.shutdown()
    print("ok")


if __name__ == "__main__":
    main()
