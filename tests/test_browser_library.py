"""Browser regressions; install Playwright and Chromium to run these checks."""

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from videomasys.api.app import browser_demo_examples

playwright = pytest.importorskip("playwright.sync_api")
WEB_ROOT = Path(__file__).parents[1] / "src" / "videomasys" / "web"


@pytest.fixture(scope="module")
def browser():
    catalog = asyncio.run(browser_demo_examples())
    with playwright.sync_playwright() as driver:
        try:
            instance = driver.chromium.launch(args=["--no-sandbox"])
        except playwright.Error as error:
            pytest.skip(f"Chromium unavailable: {error}")
        yield instance, catalog
        instance.close()


@pytest.mark.parametrize("entry", ["/", "/?browser_demo=1"])
def test_library_survives_root_navigation_and_refresh_failure(browser, entry):
    instance, catalog = browser
    page = instance.new_page()
    requests = []
    errors = []
    fail_catalog = False
    page.on("pageerror", lambda error: errors.append(str(error)))

    def respond(route):
        path = urlsplit(route.request.url).path
        requests.append(path)
        if path == "/":
            route.fulfill(path=WEB_ROOT / "index.html", content_type="text/html")
        elif path.startswith("/static/"):
            route.fulfill(path=WEB_ROOT / path.removeprefix("/static/"))
        elif path == "/v1/browser-demo/examples":
            if fail_catalog:
                route.fulfill(status=500, body="Internal Server Error")
            else:
                route.fulfill(json=catalog)
        else:
            route.fulfill(status=404)

    page.route("http://videomasys.test/**", respond)
    try:
        page.goto(f"http://videomasys.test{entry}")
        page.locator("#libraryButton").click()
        playwright.expect(page.locator("#videoList .video-item")).to_have_count(2)
        fail_catalog = True
        page.locator("#refreshVideosButton").click()
        playwright.expect(page.locator("#libraryNotice")).to_contain_text(
            "Showing previously loaded videos"
        )
        playwright.expect(page.locator("#videoList .video-item")).to_have_count(2)
        fail_catalog = False
        page.locator("#refreshVideosButton").click()
        playwright.expect(page.locator("#libraryNotice")).to_be_hidden()
        page.keyboard.press("Escape")
        page.locator("a.brand").click()
        page.locator("#libraryButton").click()
        playwright.expect(page.locator("#videoList .video-item")).to_have_count(2)
        assert "/v1/videos" not in requests
        assert "/healthz" not in requests
        assert errors == []
    finally:
        page.close()
