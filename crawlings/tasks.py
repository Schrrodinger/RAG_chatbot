# back_end_service/crawlings/tasks.py
from __future__ import absolute_import, unicode_literals
from celery import shared_task, Celery
from celery.schedules import crontab
from .crawlers import crawl_products
from django.conf import settings
from pymongo import MongoClient

app = Celery('back_end_service')


@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Calls run_crawler() every day at midnight
    sender.add_periodic_task(
        crontab(hour=0, minute=0),
        run_crawler.s(),
    )


@shared_task
def run_crawler():
    products = crawl_products()
    client = MongoClient(settings.DATABASES['default']['CLIENT']['host'],
                         username=settings.DATABASES['default']['CLIENT']['username'],
                         password=settings.DATABASES['default']['CLIENT']['password'])
    db = client[settings.DATABASES['default']['NAME']]
    product_collection = db['product']
    product_collection.insert_many(products)
