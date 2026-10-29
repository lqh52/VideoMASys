"""Opt-in real hosted UI smoke: one library run inside the Hugging Face wrapper.

--run-analysis submits exactly one analysis and can incur hosted-provider API
charges even on a CPU Space. Visiting a dedicated GPU Space can also wake its
hardware. This script never changes Space settings or retries a failed analysis.
Optional HF authentication is sent only to this app origin; anonymous is default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit


def capture_raw_trace(app, output: Path, report: dict, timeout_ms: int = 5000) -> bool:
    """Copy the actual displayed JSON; never scrape a collapsed tree or resubmit QA.

    A page-local clipboard-write hook captures exactly what Gradio's Copy button
    writes, without reading/changing the machine clipboard or requiring iframe
    clipboard permissions. The original method is restored even on failure.
    """
    hooked = False
    opened = False
    try:
        raw = app.locator("#raw-trace")
        if not raw.is_visible():
            app.get_by_text("Raw prompts & responses", exact=True).click(timeout=timeout_ms)
            opened = True
        raw.wait_for(state="visible", timeout=timeout_ms)
        app.evaluate("""() => {
            const clipboard = navigator.clipboard;
            if (!clipboard || window.__videomasysSmokeClipboard) {
                throw new Error('Trace clipboard capture is unavailable');
            }
            const descriptor = Object.getOwnPropertyDescriptor(clipboard, 'writeText');
            const capture = {text: null, clipboard, descriptor};
            Object.defineProperty(clipboard, 'writeText', {
                configurable: true,
                value: async text => { capture.text = String(text); }
            });
            window.__videomasysSmokeClipboard = capture;
        }""")
        hooked = True
        raw.get_by_role("button", name=re.compile(r"^(Copy|Copied)$")).click(timeout=timeout_ms)
        app.wait_for_function(
            "typeof window.__videomasysSmokeClipboard?.text === 'string'",
            timeout=timeout_ms,
        )
        copied = app.evaluate("window.__videomasysSmokeClipboard.text")
        trace = json.loads(copied)
        if not isinstance(trace, list) or any(not isinstance(step, dict) for step in trace):
            raise ValueError("The displayed trace is not a JSON list of stage records.")
        (output / "raw-trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2))
        report.update(raw_trace_file="raw-trace.json", trace_captured=True)
        report.pop("trace_capture_error", None)
        pipeline_steps = [step for step in trace if step.get("stage") != "Hosted inference usage"]
        if pipeline_steps:
            report["last_pipeline_stage"] = pipeline_steps[-1].get("stage")
        verdicts = [
            step.get("output", {}) for step in pipeline_steps if step.get("stage") == "Verifier"
        ]
        if verdicts:
            report["verifier_accepted"] = verdicts[-1].get("accepted")
        return True
    except Exception as exc:
        report.update(trace_captured=False, trace_capture_error=str(exc))
        return False
    finally:
        if hooked:
            try:
                app.evaluate("""() => {
                    const capture = window.__videomasysSmokeClipboard;
                    if (!capture) return;
                    if (capture.descriptor) {
                        Object.defineProperty(capture.clipboard, 'writeText', capture.descriptor);
                    } else {
                        delete capture.clipboard.writeText;
                    }
                    delete window.__videomasysSmokeClipboard;
                }""")
            except Exception:
                pass  # The page may have closed; never retry analysis to recover a trace.
        if opened:
            try:
                app.get_by_text("Raw prompts & responses", exact=True).click(timeout=timeout_ms)
            except Exception:
                pass  # Restore layout when possible without masking the capture result.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--authenticated", action="store_true")
    parser.add_argument(
        "--example-index", type=int, default=0, help="Zero-based library card index; one run only"
    )
    parser.add_argument(
        "--expected-model", default="Qwen3-VL-8B-Instruct", help="Expected visible model badge text"
    )
    parser.add_argument("--question", help="Replace the selected library question for this one run")
    parser.add_argument(
        "--open-ended", action="store_true", help="Clear all library answer choices before running"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.example_index < 0:
        parser.error("--example-index must be nonnegative.")
    if len(args.repo_id.split("/")) != 2 or any(
        not part or not all(c.isalnum() or c in "_-" for c in part)
        for part in args.repo_id.split("/")
    ):
        parser.error("Provide an explicit owner/Space name.")
    args.expected_model = args.expected_model.strip()
    if not args.expected_model:
        parser.error("--expected-model must be nonempty.")
    if args.question is not None:
        args.question = args.question.strip()
        if not 2 <= len(args.question) <= 2000:
            parser.error("--question must contain 2–2000 characters.")
    if (args.question is not None or args.open_ended) and not args.run_analysis:
        parser.error("--question and --open-ended require --run-analysis.")
    return args


def configure_analysis_inputs(frame, args, report):
    """Apply explicit overrides without submitting QA; report the actual browser values."""
    question = frame.locator("#question-input textarea")
    choices = frame.locator("#choices-panel textarea")
    report["library_question"] = question.input_value()
    if args.question is not None:
        question.fill(args.question)
    if args.open_ended:
        opened = not choices.is_visible()
        if opened:
            frame.get_by_text("Answer choices · optional", exact=True).click()
        try:
            choices.fill("")
        finally:
            if opened:
                frame.get_by_text("Answer choices · optional", exact=True).click()
    report["question"] = question.input_value()
    report["options"] = [
        option.strip() for option in choices.input_value().splitlines() if option.strip()
    ]
    report["open_ended"] = not report["options"]


def main():
    args = parse_args()
    from playwright.sync_api import expect, sync_playwright

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    app_host = args.repo_id.replace("/", "-").replace("_", "-").lower() + ".hf.space"
    report = {
        "fixture_only": False,
        "inference_invoked": False,
        "repo": args.repo_id,
        "example_index": args.example_index,
        "expected_model": args.expected_model,
        "qa_completed": False,
        "ui_checks_passed": False,
        "trace_captured": False,
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1453, "height": 875},
            color_scheme="dark",
            reduced_motion="no-preference",
            service_workers="block",
        )
        if args.authenticated:
            from huggingface_hub import get_token

            token = get_token()
            if not token:
                raise ValueError("Authenticate locally first; do not pass a token on the CLI.")

            def authenticate(route):
                if urlsplit(route.request.url).hostname != app_host:
                    return route.continue_()
                route.continue_(
                    headers={**route.request.headers, "Authorization": f"Bearer {token}"}
                )

            context.route(f"https://{app_host}/**", authenticate)
        page = context.new_page()
        page.set_default_timeout(45000)
        expect.set_options(timeout=45000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        app = None
        try:
            page.goto(
                f"https://huggingface.co/spaces/{args.repo_id}",
                wait_until="domcontentloaded",
            )
            iframe = page.locator(f'iframe[src^="https://{app_host}/"]')
            expect(iframe).to_be_visible()
            frame = page.frame_locator(f'iframe[src^="https://{app_host}/"]')
            expect(frame.locator(".model-badge")).to_contain_text(args.expected_model)
            report["displayed_model"] = frame.locator(".model-badge").inner_text()
            assert "fixture" not in frame.locator(".model-badge").inner_text().lower()
            checks.append("real model app loaded inside the HF scrolling-disabled iframe")
            app = next(f for f in page.frames if urlsplit(f.url).hostname == app_host)
            assert iframe.get_attribute("scrolling") == "no"
            if args.run_analysis:
                examples = frame.locator("#example-library .wrap > label")
                # The model badge is SSR HTML; library inputs hydrate later.
                # Wait before counting so a cold page is not treated as empty.
                expect(examples.first).to_be_visible()
                count = examples.count()
                if not 0 <= args.example_index < count:
                    raise ValueError(
                        f"--example-index must be between 0 and {count - 1}; "
                        f"the app has {count} library cards."
                    )
                selected = examples.nth(args.example_index)
                report["selected_example_label"] = selected.inner_text()
                selected.click()
                expect(frame.locator("#question-input textarea")).not_to_have_value("")
                configure_analysis_inputs(frame, args, report)
                frame.locator("#run").click()
                report["inference_invoked"] = True
                expect(frame.locator("#run")).to_be_disabled()
                app.wait_for_function(
                    "document.querySelector('#status .run-status')?.dataset.state "
                    "=== 'complete' || document.querySelector('#status .run-status')"
                    "?.dataset.state === 'error'",
                    timeout=300000,
                )
                report["status"] = frame.locator("#status").inner_text()
                run_state = frame.locator("#status .run-status").get_attribute("data-state")
                report.update(qa_state=run_state, qa_completed=run_state == "complete")
                answer_panel = frame.locator("#answer")
                report["answer"] = answer_panel.inner_text() if answer_panel.is_visible() else None
                capture_raw_trace(app, output, report)
                assert run_state == "complete", report["status"]
                assert report["trace_captured"], (
                    "Analysis completed, but full trace capture failed: "
                    + report.get("trace_capture_error", "unknown capture error")
                )
                checks.append("one real library analysis completed")

                def assert_fit(label):
                    app.wait_for_function(
                        "document.querySelector('.app-footer').getBoundingClientRect().bottom "
                        "<= innerHeight + 2",
                        timeout=15000,
                    )
                    geometry = app.evaluate("""() => ({
                        viewport: innerHeight,
                        documentHeight: document.documentElement.scrollHeight,
                        footerBottom: document.querySelector('.app-footer')
                            .getBoundingClientRect().bottom,
                        innerScrollY: scrollY
                    })""")
                    assert geometry["innerScrollY"] == 0, "Inner scrolling hid an iframe cutoff."
                    report.setdefault("measurements", []).append({"action": label, **geometry})

                assert_fit("completed")
                for cycle in range(3):
                    frame.locator('[data-flow-action="expand"]').click()
                    assert_fit(f"expand all {cycle + 1}")
                    frame.locator('[data-flow-action="collapse"]').click()
                    assert_fit(f"collapse all {cycle + 1}")
                checks.append(
                    "completed results and repeated expansion fit without inner scrolling"
                )
                page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
                page.wait_for_timeout(400)
                expect(frame.locator(".app-footer")).to_be_in_viewport()
                footer = frame.locator(".app-footer").bounding_box()
                assert footer and footer["y"] >= -2
                assert footer["y"] + footer["height"] <= page.viewport_size["height"] + 2
                page.screenshot(path=str(output / "hosted-outer-scroll-bottom.png"))
                checks.append("app footer reachable at the bottom of the outer HF page")
            else:
                page.screenshot(path=str(output / "hosted-initial.png"))
            assert not errors, errors
            report.update(
                result="passed", ui_checks_passed=True, checks=checks, browser_errors=errors
            )
        except BaseException as exc:
            if app is not None and report["inference_invoked"] and not report["trace_captured"]:
                # Timeout/error capture never starts another analysis. The provider
                # may still finish a request after the browser stops waiting.
                try:
                    state = app.locator("#status .run-status").get_attribute(
                        "data-state", timeout=2000
                    )
                    report.update(qa_state=state, qa_completed=state == "complete")
                    report["status"] = app.locator("#status").inner_text(timeout=2000)
                except Exception:
                    pass
                capture_raw_trace(app, output, report, timeout_ms=2000)
            report.update(result="failed", error=str(exc), checks=checks, browser_errors=errors)
            page.screenshot(path=str(output / "hosted-failure.png"))
            raise
        finally:
            (output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            browser.close()


if __name__ == "__main__":
    main()
