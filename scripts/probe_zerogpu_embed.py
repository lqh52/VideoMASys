"""LOCAL fixture-only probe of Gradio's parentIFrame height reporting.

The outer page emulates a scrolling-disabled HF iframe. Its minimal height bridge
records Gradio's real size() requests; it is not the full Hugging Face shell.
No hosted URL or model inference is allowed. Start preview_zerogpu_ui.py first.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

from smoke_zerogpu_ui import is_loopback_url

BRIDGE = """
if (window.parent !== window) {
    window.parentIFrame = {
        autoResize: () => {},
        size: (height) => window.parent.postMessage({
            kind: 'fixture-height', height, reportedAt: performance.now()
        }, '*'),
        scrollTo: () => {},
        getPageInfo: (callback) => callback({scrollTop: 0, offsetTop: 0})
    };
}
"""

MEASURE = """() => ({
    viewport: innerHeight,
    innerScrollY: scrollY,
    documentHeight: document.documentElement.scrollHeight,
    bodyHeight: document.body.scrollHeight,
    footerBottom: document.querySelector('.app-footer').getBoundingClientRect().bottom + scrollY,
    openSteps: [...document.querySelectorAll('#pipeline-flow details[open]')]
        .map(step => step.dataset.stepKey),
    overflow: getComputedStyle(document.body).overflow,
})"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8203/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=1453)
    parser.add_argument("--height", type=int, default=875)
    parser.add_argument("--color-scheme", choices=("light", "dark"), default="dark")
    parser.add_argument(
        "--reduced-motion", choices=("reduce", "no-preference"), default="no-preference"
    )
    parser.add_argument("--cross-origin", action="store_true")
    parser.add_argument(
        "--assert-fit", action="store_true", help="Fail on clipping/state/resize issues"
    )
    args = parser.parse_args()
    if not is_loopback_url(args.url):
        parser.error("Only a local fixture URL is allowed.")
    from playwright.sync_api import expect, sync_playwright

    expect.set_options(timeout=45000)
    outer_url = urljoin(args.url, "/__fixture_embed_probe__")
    if args.cross_origin:
        parts = urlsplit(outer_url)
        other_host = "localhost" if parts.hostname != "localhost" else "127.0.0.1"
        authority = f"{other_host}:{parts.port}" if parts.port else other_host
        outer_url = urlunsplit((parts.scheme, authority, parts.path, "", ""))
    parent_html = (
        '<!doctype html><html><head><meta charset="utf-8"></head>'
        '<body style="margin:0;background:#e6edf0">'
        '<header style="height:100px">LOCAL embedding probe — not model inference</header>'
        f'<iframe id="app" src="{html.escape(args.url, quote=True)}" scrolling="no" '
        'style="width:100%;height:800px;border:0;overflow:hidden"></iframe>'
        '<script>window.heightReports=[];addEventListener("message", event => {'
        'if(event.source !== document.querySelector("iframe").contentWindow)return;'
        'if(event.data?.kind !== "fixture-height")return;'
        'const h=event.data.height;if(!Number.isFinite(h)||h<100||h>60000)return;'
        'heightReports.push(event.data);document.querySelector("iframe").style.height=h+"px";'
        '});</script></body></html>'
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            color_scheme=args.color_scheme,
            reduced_motion=args.reduced_motion,
        )
        if args.cross_origin:
            # Chromium otherwise treats the synthetic fulfilled parent as a
            # public address space and blocks its loopback iframe before load.
            context.grant_permissions(["local-network-access"], origin=outer_url)
        context.add_init_script(BRIDGE)
        context.route(
            outer_url, lambda route: route.fulfill(content_type="text/html", body=parent_html)
        )
        page = context.new_page()
        page.set_default_timeout(45000)
        page.goto(outer_url, wait_until="domcontentloaded")
        frame = page.frame_locator("#app")
        expect(frame.locator(".model-badge")).to_contain_text("UI fixture")
        expect(frame.locator(".model-badge")).to_contain_text("not model inference")
        frame.locator("#example-library label").first.evaluate("node => node.click()")
        expect(frame.locator("#question-input textarea")).not_to_have_value("")
        expect(frame.locator("#video-input video")).to_be_visible()
        frame.locator("#run").evaluate("node => node.click()")
        expect(frame.locator('#status .run-status[data-state="complete"]')).to_be_visible()
        app = next(f for f in page.frames if f.parent_frame)
        app.wait_for_timeout(300)
        measures, checks = [], []

        def measure(action):
            value = {"action": action, **app.evaluate(MEASURE)}
            measures.append(value)
            return value

        def check(name, passed):
            checks.append({"name": name, "passed": bool(passed)})

        def check_fit(action):
            value = measure(action)
            check(
                f"{action}: app footer fits iframe", value["footerBottom"] <= value["viewport"] + 2
            )
            return value

        def check_outer_footer(action):
            page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
            page.wait_for_timeout(150)
            box = app.locator(".app-footer").bounding_box()
            viewport = page.evaluate("innerHeight")
            check(
                f"{action}: footer reached by outer scrolling",
                box is not None and box["y"] >= 0 and box["y"] + box["height"] <= viewport + 2,
            )
            check(f"{action}: inner frame did not scroll", app.evaluate("scrollY") == 0)

        check_fit("completed")
        steps = app.locator("#pipeline-flow details[data-step-key]")
        for index in range(steps.count()):
            step = steps.nth(index)
            key = step.get_attribute("data-step-key")
            step.evaluate("node => { node.open = true; }")
            app.wait_for_timeout(300)
            check_fit(f"open {key}")
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        page.wait_for_timeout(200)
        check_outer_footer("desktop expanded")
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.output_dir / "embedded-bottom.png"))

        for cycle in range(6):
            app.locator('[data-flow-action="collapse"]').evaluate("node => node.click()")
            app.wait_for_timeout(300)
            value = check_fit(f"collapse all {cycle + 1}")
            check(f"collapse all {cycle + 1}: none open", not value["openSteps"])
            app.locator('[data-flow-action="expand"]').evaluate("node => node.click()")
            app.wait_for_timeout(300)
            value = check_fit(f"expand all {cycle + 1}")
            check(f"expand all {cycle + 1}: all open", len(value["openSteps"]) == steps.count())

        page.set_viewport_size({"width": 390, "height": 844})
        app.wait_for_timeout(500)
        check_fit("mobile expanded")
        check_outer_footer("mobile expanded")
        for index in range(app.locator("#evidence-section details").count()):
            app.locator("#evidence-section details").nth(index).evaluate(
                "node => { node.open=true; }"
            )
            app.wait_for_timeout(250)
            check_fit(f"mobile evidence caption {index + 1}")
        captions = app.locator("#evidence-section details")
        for cycle in range(6):
            captions.evaluate_all("nodes => nodes.forEach(node => { node.open=false; })")
            app.wait_for_timeout(250)
            check_fit(f"mobile close captions {cycle + 1}")
            captions.evaluate_all("nodes => nodes.forEach(node => { node.open=true; })")
            app.wait_for_timeout(250)
            check_fit(f"mobile open captions {cycle + 1}")
        check_outer_footer("mobile evidence captions expanded")
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        if args.output_dir:
            page.screenshot(path=str(args.output_dir / "embedded-mobile-bottom.png"))

        app.wait_for_timeout(500)
        reports_before = page.evaluate("window.heightReports.length")
        height_before = app.evaluate("innerHeight")
        app.wait_for_timeout(1500)
        reports_after = page.evaluate("window.heightReports.length")
        height_after = app.evaluate("innerHeight")
        check("idle: no additional resize reports", reports_before == reports_after)
        check("idle: stable iframe height", height_before == height_after)

        page.set_viewport_size({"width": args.width, "height": args.height})
        page.evaluate("window.scrollTo(0, 0)")
        expect(frame.locator(".model-badge")).to_contain_text("UI fixture")
        frame.locator("#run").evaluate("node => node.click()")
        expect(frame.locator("#run")).to_be_disabled()
        expect(frame.locator('#status .run-status[data-state="complete"]')).to_be_visible()
        app.wait_for_timeout(300)
        value = check_fit("rerun completed")
        check("rerun: previous expansion state cleared", not value["openSteps"])
        check_outer_footer("rerun completed")

        result = {
            "fixture_only": True,
            "bridge": "minimal parentIFrame size(height), scrolling=no",
            "viewport": {"width": args.width, "height": args.height},
            "color_scheme": args.color_scheme,
            "reduced_motion": args.reduced_motion,
            "cross_origin": args.cross_origin,
            "measurements": measures,
            "checks": checks,
            "height_reports": page.evaluate("window.heightReports"),
            "last_footer_clipped": measures[-1]["footerBottom"] > measures[-1]["viewport"] + 2,
        }
        if args.output_dir:
            (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        browser.close()
        if args.assert_fit:
            failed = [item["name"] for item in checks if not item["passed"]]
            assert not failed, f"Embedding regression checks failed: {failed}"


if __name__ == "__main__":
    main()
