import scrapy
import os
import re
from urllib.parse import urlparse
from scrapy.crawler import CrawlerProcess

class PricelessScraper(scrapy.Spider):
    name = "priceless_scraper"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = "scraped_experiences"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Read extracted links from the Playwright output file
        with open("extracted_links.txt", "r", encoding="utf-8") as f:
            self.start_urls = [line.strip() for line in f.readlines()]

    def parse(self, response):
        """Extracts text from p, li, h1-h6 elements."""
        url = response.url

        paragraphs = response.xpath("//p//text()").getall()
        list_items = response.xpath("//li//text()").getall()
        headers = response.xpath("//h1//text() | //h2//text() | //h3//text() | //h4//text() | //h5//text() | //h6//text()").getall()

        all_text = "\n".join(text.strip() for text in paragraphs + list_items + headers if text.strip())

        if not all_text:
            all_text = "No content found."

        self.save_to_file(url, all_text)

    def save_to_file(self, url, text_content):
        """Saves extracted content to a file."""
        parsed_url = urlparse(url)
        sanitized_path = re.sub(r"[^a-zA-Z0-9_-]", "_", parsed_url.path)
        filename = os.path.join(self.output_dir, f"{sanitized_path}.txt")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\n{text_content}")

if __name__ == "__main__":
    process = CrawlerProcess(settings={
        "LOG_LEVEL": "INFO",
        "FEEDS": {"output.json": {"format": "json"}},
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 3,
        "AUTOTHROTTLE_MAX_DELAY": 10,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1,
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 2,
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    })
    process.crawl(PricelessScraper)
    process.start()
