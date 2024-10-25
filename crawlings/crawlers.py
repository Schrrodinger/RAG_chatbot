# back_end_service/crawlings/crawlers.py
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib3 import Retry
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from requests.adapters import HTTPAdapter
import time


def fetch_product_links(url):
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        response = session.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching product links: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    product_links = [a['href'] for a in soup.select('div.product-list-filter.is-flex.is-flex-wrap-wrap a')]
    return product_links


# def fetch_product_details(url):
#     options = FirefoxOptions()
#     options.headless = True
#     service = FirefoxService("./geckodriver.exe")  # Ensure geckodriver is in the same directory
#     driver = webdriver.Firefox(service=service, options=options)
#
#     try:
#         driver.get(url)
#         name = driver.find_element(By.CSS_SELECTOR, 'div.box-product-name h1').text
#     except Exception as e:
#         print(f"Error fetching product details: {e}")
#         name = "N/A"
#     finally:
#         driver.quit()
#
#     return {
#         'name': name,
#         'url': url
#     }

# def fetch_product_details(url):
#     try:
#         response = requests.get(url)
#         response.encoding = 'utf-8'
#         response.raise_for_status()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching product details: {e}")
#         return {'name': 'N/A', 'url': url}
#
#     soup = BeautifulSoup(response.content, 'html.parser')
#     try:
#         name = soup.select_one('div.box-product-name h1').text.strip()
#         description = soup.select_one('#productDetailV2 .description').text.strip()
#         price = float(soup.select_one('.tpt---price').text.strip().replace('.', '').replace('₫', '').replace('đ', ''))
#         special_price_tags = soup.select('.tpt---sale-price')
#         if len(special_price_tags) >= 2:
#             # Pick the second one
#             special_price = float(special_price_tags[1].text.strip().replace('.', '').replace('₫', '').replace('đ', ''))
#         else:
#             special_price = None
#         # special_price = float(special_price.text.strip().replace(',', '').replace('₫', '')) if special_price else None
#         # image_url = soup.select_one('.product-image img')['src']
#             # Extract technical specifications
#         ul_elements = soup.select('ul.technical-content')
#         print(f"Number of <ul> elements with class 'technical-content': {len(ul_elements)}")
#         for ul in ul_elements:
#             print(ul.prettify())
#         technical_info = {}
#         technical_items = soup.select('ul.technical-content > li.technical-content-item')
#         print(f"Found {len(technical_items)} technical items.")
#         for item in technical_items:
#                 key = item.select_one('p').text.strip()
#                 value = item.select_one('div').text.strip()
#                 technical_info[key] = value
#                 print(f"Extracted technical info - {key}: {value}")
#     except AttributeError as e:
#         print(f"Error parsing product details: {e}")
#         return None
#
#     return {
#         'name': name,
#         'url': url,
#         'price': price,
#         'special_price': special_price,
#         'description': description.encode('utf-8').decode('unicode_escape'),
#         'technical_info': technical_info
#     }

def fetch_product_details(url):
    options = FirefoxOptions()
    options.headless = True
    service = FirefoxService(executable_path='./geckodriver')  # Ensure geckodriver is in the same directory
    driver = webdriver.Firefox(service=service, options=options)

    try:
        driver.get(url)

        name = driver.find_element(By.CSS_SELECTOR, 'div.box-product-name h1').text.strip()
        description = driver.find_element(By.CSS_SELECTOR, '#productDetailV2 .description').text.strip()
        price = float(
            driver.find_element(By.CSS_SELECTOR, '.tpt---price').text.strip().replace('.', '').replace('₫', '').replace(
                'đ', ''))

        special_price_tags = driver.find_elements(By.CSS_SELECTOR, '.tpt---sale-price')
        if len(special_price_tags) >= 2:
            # Pick the second one
            special_price = float(special_price_tags[1].text.strip().replace('.', '').replace('₫', '').replace('đ', ''))
        else:
            special_price = None

        driver.implicitly_wait(20)
        # Extract technical specifications
        technical_info = {}
        ul_elements = driver.find_elements(By.CSS_SELECTOR, 'ul.technical-content')
        print(f"Number of <ul> elements with class 'technical-content': {len(ul_elements)}")
        for ul in ul_elements:
            print(ul.get_attribute('outerHTML'))

        technical_items = driver.find_elements(By.CSS_SELECTOR, 'ul.technical-content > li.technical-content-item')
        print(f"Found {len(technical_items)} technical items.")
        for item in technical_items:
            key = item.find_element(By.CSS_SELECTOR, 'p').text.strip()
            value = item.find_element(By.CSS_SELECTOR, 'div').text.strip()
            technical_info[key] = value
            print(f"Extracted technical info - {key}: {value}")
    except Exception as e:
        print(f"Error parsing product details: {e}")
        return None
    finally:
        driver.quit()

    return {
        'name': name,
        'url': url,
        'price': price,
        'special_price': special_price,
        'description': description.encode('utf-8').decode('unicode_escape'),
        'technical_info': technical_info
    }


def crawl_products():
    base_url = "https://cellphones.com.vn/laptop.html"
    product_links = fetch_product_links(base_url)
    product_links = product_links[:1]
    print(product_links)
    products = [fetch_product_details(link) for link in product_links]
    return products

