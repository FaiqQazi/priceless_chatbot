import os
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.priceless.com/m/filter/options"
OUTPUT_FILE = "extracted_links.txt"

def extract_experience_links():
    """Extracts all experience links using Playwright and saves them to a file."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(BASE_URL)

        # Scroll to load all links
        prev_height = 0
        while True:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                break
            prev_height = new_height

        # Extract all unique links
        all_links = page.locator("a").evaluate_all("elements => elements.map(e => e.href)")
        experience_links = list(set([link for link in all_links if "/product/" in link]))

        print(f"Total extracted experience links: {len(experience_links)}")

        # Save to file
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for link in experience_links:
                f.write(link + "\n")

        browser.close()

if __name__ == "__main__":
    extract_experience_links()
