"""Offline CLI/browser fixtures; no remote Space, model requests, or video accuracy claims."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "hosted_smoke_overrides",
    Path(__file__).resolve().parents[1] / "scripts/smoke_hosted_zerogpu_embed.py",
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def parse(*args):
    return smoke.parse_args(["fixture/demo", "--output-dir", "/unused-fixture-output", *args])


def test_safe_defaults_and_explicit_overrides():
    args = parse()
    assert args.expected_model == "Qwen3-VL-8B-Instruct"
    assert args.example_index == 0
    assert args.question is None and not args.open_ended
    assert not args.authenticated and not args.run_analysis
    overridden = parse(
        "--run-analysis",
        "--expected-model",
        " test-model ",
        "--example-index",
        "2",
        "--question",
        " What happens? ",
        "--open-ended",
    )
    assert overridden.expected_model == "test-model"
    assert overridden.question == "What happens?"
    assert overridden.example_index == 2 and overridden.open_ended
    assert not overridden.authenticated


@pytest.mark.parametrize(
    "args",
    [
        ["--question", "What happens?"],
        ["--open-ended"],
        ["--expected-model", " "],
        ["--example-index", "-1"],
        ["--run-analysis", "--question", "?"],
        ["--run-analysis", "--question", "q" * 2001],
    ],
)
def test_invalid_configuration_fails_before_browser_or_api(args):
    with pytest.raises(SystemExit) as exc:
        parse(*args)
    assert exc.value.code == 2


@pytest.fixture
def page():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        try:
            browser = driver.chromium.launch(headless=True)
        except playwright.Error as error:
            pytest.skip(f"Local Chromium unavailable: {error}")
        context = browser.new_context(service_workers="block")
        requests = []

        def block(route):
            requests.append(route.request.url)
            route.abort()

        context.route("**/*", block)
        page = context.new_page()
        page.set_content("""
            <div id="question-input"><textarea>Original fixture question?</textarea></div>
            <div id="choices-panel">
              <button onclick="this.nextElementSibling.hidden = !this.nextElementSibling.hidden">
                Answer choices · optional
              </button>
              <textarea hidden>First fixture choice
Second fixture choice</textarea>
            </div>
            <button id="run"
              onclick="window.fixtureRuns = (window.fixtureRuns || 0) + 1">Run</button>
        """)
        yield page
        assert requests == []
        assert page.evaluate("window.fixtureRuns || 0") == 0
        browser.close()


def test_overrides_clear_choices_and_restore_collapsed_panel_without_submitting(page):
    args = parse("--run-analysis", "--question", "What is visible?", "--open-ended")
    report = {}
    smoke.configure_analysis_inputs(page, args, report)
    assert report == {
        "library_question": "Original fixture question?",
        "question": "What is visible?",
        "options": [],
        "open_ended": True,
    }
    assert not page.locator("#choices-panel textarea").is_visible()


def test_default_preserves_actual_question_choices_and_open_panel(page):
    page.get_by_text("Answer choices · optional", exact=True).click()
    report = {}
    smoke.configure_analysis_inputs(page, parse("--run-analysis"), report)
    assert report == {
        "library_question": "Original fixture question?",
        "question": "Original fixture question?",
        "options": ["First fixture choice", "Second fixture choice"],
        "open_ended": False,
    }
    assert page.locator("#choices-panel textarea").is_visible()
