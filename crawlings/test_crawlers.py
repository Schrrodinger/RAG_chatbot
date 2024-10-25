# back_end_service/crawlings/test_crawlers.py
from crawlers import crawl_products
import json
import os
import requests
import zipfile
import sys

# Ensure the script outputs in UTF-8
sys.stdout.reconfigure(encoding='utf-8')


def download_geckodriver():
    if not os.path.exists("geckodriver.exe"):
        print("Downloading GeckoDriver...")
        url = "https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-win64.zip"
        response = requests.get(url)
        with open("geckodriver.zip", "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile("geckodriver.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove("geckodriver.zip")
        print("GeckoDriver downloaded and extracted.")
    else:
        print("GeckoDriver already exists.")


def test_crawl():
    download_geckodriver()
    products = crawl_products()
    with open("test_products.json", "w", encoding='utf-8') as f:
        json.dump(products, f, indent=4)
    print("Scraping completed. Check test_products.json for results.")

# Run the test crawl function
test_crawl()