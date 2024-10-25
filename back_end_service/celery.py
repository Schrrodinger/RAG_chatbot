# back_end_service/back_end_service/celery.py
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'back_end_service.settings')

app = Celery('back_end_service')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
