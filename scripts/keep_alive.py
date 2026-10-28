"""
Keep the Streamlit Community Cloud deployment awake.

A sleeping app shows a "get this app back up" button instead of the app, and
a plain HTTP request does not wake it -- so this opens the app in a headless
browser, clicks the button if it is there, and waits for the app to render.

Usage: python scripts/keep_alive.py <app-url>
"""

import sys

from playwright.sync_api import sync_playwright

WAKE_BUTTON = "button:has-text('back up')"
APP_LOADED = "[data-testid='stApp'], [data-testid='stAppViewContainer']"


def main(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(10_000)

        wake = page.locator(WAKE_BUTTON)
        if wake.count() > 0:
            print("App was asleep -- clicking wake-up button")
            wake.first.click()
            page.wait_for_timeout(5_000)
        else:
            print("App is awake")

        # the app itself renders inside an iframe on *.streamlit.app
        for frame in page.frames:
            try:
                frame.wait_for_selector(APP_LOADED, timeout=180_000)
                print(f"App rendered ({frame.url})")
                break
            except Exception:
                continue
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1])
