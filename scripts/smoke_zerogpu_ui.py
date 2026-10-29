"""Browser regression checks for LOCAL, explicitly labeled UI fixtures only.

Start scripts/preview_zerogpu_ui.py --step-delay 1.5 first, so steps can be
inspected between updates. This script refuses non-loopback URLs
and checks the fixture badge before clicking Run. It never intentionally invokes
a hosted Space or real model inference.
Requires Playwright and its Chromium browser in the development environment.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


def is_loopback_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return False
        if parsed.hostname == "localhost":
            return True
        return bool(parsed.hostname and ipaddress.ip_address(parsed.hostname).is_loopback)
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8203")
    parser.add_argument(
        "--output-dir", type=Path, help="Screenshot directory; defaults to a temp folder"
    )
    parser.add_argument("--timeout-ms", type=int, default=45000)
    args = parser.parse_args()
    if not is_loopback_url(args.url):
        parser.error("Only localhost or loopback-IP HTTP(S) URLs are allowed.")
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive.")
    try:
        from playwright.sync_api import expect, sync_playwright
    except ImportError:
        parser.error("Install Playwright and Chromium in the development environment first.")
    expect.set_options(timeout=args.timeout_ms)

    output = args.output_dir or Path(tempfile.mkdtemp(prefix="videomasys-ui-smoke-"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks, page_errors = [], []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1366, "height": 1000},
            color_scheme="light",
            reduced_motion="reduce",
            service_workers="block",
        )

        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        def screenshot(name):
            page.screenshot(path=str(output / f"{name}.png"), full_page=True)

        def check_no_overflow():
            dimensions = page.evaluate(
                """() => ({
                    viewport: window.innerWidth,
                    document: document.documentElement.scrollWidth,
                    body: document.body.scrollWidth
                })"""
            )
            assert max(dimensions["document"], dimensions["body"]) <= dimensions["viewport"] + 1, (
                f"Horizontal overflow: {dimensions}"
            )

        try:
            # networkidle is unsuitable for Gradio's long-lived connections.
            page.goto(args.url, wait_until="domcontentloaded")
            assert is_loopback_url(page.url), "Refusing a non-loopback redirect."
            expect(page.locator(".model-badge")).to_contain_text("UI fixture")
            assert "not model inference" in page.locator(".model-badge").inner_text(), (
                "Only the explicitly marked fixture server may run this smoke test."
            )
            for selector in ("#results", "#evidence-section", "#run-details"):
                expect(page.locator(selector)).to_be_hidden()
            check_no_overflow()
            screenshot("01-initial-desktop")
            checks.append("initial results, evidence, and pipeline hidden")

            question = page.locator("#question-input textarea")
            run = page.locator("#run")
            examples = page.locator("#example-library label")
            expect(examples.first).to_be_visible()
            if page.locator("#example-library.with-thumbnails").count():
                previews = examples.evaluate_all(
                    """elements => elements.map(element => {
                        const image = getComputedStyle(element, '::before');
                        return image.backgroundImage.startsWith('url("data:image/')
                            && parseFloat(image.height) > 30;
                    })"""
                )
                assert all(previews), "Each curated example must display its thumbnail."
                checks.append("all example cards display real-frame thumbnail images")
            examples.first.click()
            expect(question).not_to_have_value("")
            original_question = question.input_value()
            expect(page.locator("#video-input video")).to_be_visible()
            choices = page.locator("#choices-panel textarea")
            expect(choices).to_be_visible()
            assert choices.input_value().strip(), "The first fixture example must include choices."
            checks.append("example prefilled its video, question, and expanded choices")

            def click_run():
                # Recheck before every run: a live model must never be exercised.
                expect(page.locator(".model-badge")).to_contain_text("UI fixture")
                run.click()
                expect(run).to_be_disabled(timeout=1500)
                expect(run).to_have_text("Running analysis…", timeout=1500)
                assert question.evaluate("node => node.disabled || node.readOnly"), (
                    "Question must be locked while analysis is running."
                )

            def expect_complete():
                expect(page.locator('#status .run-status[data-state="complete"]')).to_be_visible()
                expect(run).to_be_enabled()
                expect(run).to_have_text("Run analysis")
                expect(question).to_be_editable()
                expect(page.locator("#results")).to_be_visible()
                expect(page.locator("#answer")).to_contain_text("UI FIXTURE")
                expect(page.locator("#evidence-section")).to_be_visible()
                expect(page.locator("#run-details")).to_be_visible()
                order = page.evaluate('''() => {
                    const results = document.querySelector('#results').getBoundingClientRect();
                    const pipeline = document.querySelector('#run-details').getBoundingClientRect();
                    const evidence = document.querySelector('#evidence-section')
                        .getBoundingClientRect();
                    return results.bottom <= pipeline.top && pipeline.bottom <= evidence.top;
                }''')
                assert order, 'Expected answer/metrics, then pipeline, then evidence.'

            click_run()
            planner = page.locator('details.flow-step').filter(
                has=page.locator('summary strong', has_text='Planner')
            ).first
            tool = page.locator('details.flow-step').filter(
                has=page.locator('summary strong', has_text='Tool agent')
            ).first
            retrieval = page.locator('details.flow-step').filter(
                has=page.locator('summary strong', has_text='Retrieval')
            ).first
            planner.locator('summary').click()
            expect(page.locator('#evidence-section')).to_be_hidden()
            expect(planner).to_have_attribute('open', '')
            expect(tool).to_be_visible()
            expect(planner).to_have_attribute('open', '')
            planner.locator('summary').focus()
            page.keyboard.press('Enter')
            expect(planner).not_to_have_attribute('open', '')
            tool.locator('summary').focus()
            page.keyboard.press('Enter')
            expect(tool).to_have_attribute('open', '')
            expect(retrieval).to_be_visible()
            expect(planner).not_to_have_attribute('open', '')
            expect(tool).to_have_attribute('open', '')
            expect_complete()
            expect(tool).to_have_attribute('open', '')
            expect(planner).not_to_have_attribute('open', '')
            checks.append('mouse/keyboard expansion and collapse persist across streamed updates')
            checks.append('answer/metrics precede pipeline, which precedes evidence')

            def check_pipeline_scrolling():
                expand = page.locator('[data-flow-action="expand"]')
                collapse = page.locator('[data-flow-action="collapse"]')
                expand.click()
                steps = page.locator('details.flow-step')
                assert all(steps.evaluate_all('els => els.map(e => e.open)'))
                sizes = page.locator('.flow-body pre').evaluate_all(
                    'els => els.map(e => ({height:e.clientHeight, content:e.scrollHeight}))'
                )
                assert sizes and all(size['content'] <= size['height'] + 1 for size in sizes), (
                    f'Nested output scrollbar: {sizes}'
                )
                tool.locator('pre').hover()
                page.screenshot(
                    path=str(output / f'pipeline-{page.viewport_size["width"]}px.png'),
                    full_page=False,
                )
                before = page.evaluate('window.scrollY')
                page.mouse.wheel(0, 350)
                page.wait_for_function('(y) => window.scrollY > y + 100', arg=before)
                verifier = steps.filter(
                    has=page.locator('summary strong', has_text='Verifier')
                ).last
                verifier.locator('summary').focus()
                expect(verifier.locator('summary')).to_be_in_viewport()
                page.keyboard.press('End')
                expect(page.locator('#evidence-section')).to_be_in_viewport()
                collapse.click()
                assert not any(steps.evaluate_all('els => els.map(e => e.open)'))
                page.locator('#run-details').evaluate(
                    "e => e.scrollIntoView({block: 'start'})"
                )
                page.screenshot(
                    path=str(output / f'section-order-{page.viewport_size["width"]}px.png'),
                    full_page=False,
                )
                checks.append(
                    f'{page.viewport_size["width"]}px: expand/collapse all, page scrolling '
                    'over tool output, final step reachable'
                )
                check_no_overflow()

            check_pipeline_scrolling()
            page.locator('[data-flow-action="expand"]').click()
            screenshot("02-complete-desktop")
            checks.append("immediate run acknowledgement, input lock, completion, and unlock")

            # fill emits DOM input/change without keypress, reproducing the
            # Gradio 6 .input regression that ordinary typing can conceal.
            question.fill("UI fixture intentional error")
            expect(page.locator("#results")).to_be_hidden()
            expect(page.locator("#run-details")).to_be_hidden()
            checks.append("question fill clears stale results without needing a keypress")
            click_run()
            expect(page.locator('#status .run-status[data-state="error"]')).to_be_visible()
            expect(page.locator("#status")).to_contain_text("intentional test error")
            expect(run).to_be_enabled()
            expect(question).to_be_editable()
            expect(page.locator("#answer")).to_have_text("")
            expect(page.locator("#run-details")).to_be_visible()
            expect(page.locator("#run-details")).to_contain_text("intentional test error")
            expect(tool).not_to_have_attribute('open', '')
            expect(planner).not_to_have_attribute('open', '')
            failed = page.locator('details.flow-step').filter(
                has=page.get_by_text('Run', exact=True)
            )
            expect(failed).to_have_attribute('open', '')
            screenshot("03-error-desktop")
            checks.append("intentional failure preserves diagnostics and resets controls")

            question.fill(original_question)
            expect(page.locator("#run-details")).to_be_hidden()
            click_run()
            expect_complete()
            expect(tool).not_to_have_attribute('open', '')
            expect(planner).not_to_have_attribute('open', '')
            checks.append("successful retry after failure")
            checks.append('new runs reset previous step expansion state')

            page.set_viewport_size({"width": 390, "height": 844})
            expect_complete()
            check_pipeline_scrolling()
            page.locator('[data-flow-action="expand"]').click()
            check_no_overflow()
            screenshot("04-complete-mobile")
            page.locator('[data-flow-action="collapse"]').click()
            checks.append("390-pixel completed layout has no horizontal overflow")
            choices.fill("Changed first choice\nChanged second choice")
            expect(page.locator("#results")).to_be_hidden()
            expect(page.locator("#run-details")).to_be_hidden()
            expect(run).to_be_enabled()
            check_no_overflow()
            screenshot("05-cleared-mobile")
            checks.append("choice edit clears prior results and keeps mobile layout bounded")
            assert not page_errors, f"Browser errors: {page_errors}"
        except BaseException:
            print(
                json.dumps(
                    {
                        "question_at_failure": page.locator(
                            "#question-input textarea"
                        ).input_value(),
                        "status_at_failure": page.locator("#status").inner_text(),
                        "screenshots": str(output),
                    }
                )
            )
            screenshot("failure")
            raise
        finally:
            browser.close()

    print(
        json.dumps(
            {
                "result": "passed",
                "fixture_only": True,
                "checks": checks,
                "screenshots": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
