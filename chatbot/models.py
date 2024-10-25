from django.db import models


class Category(models.Model):
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)


class Product(models.Model):
    name = models.CharField(max_length=255)
    price = models.FloatField()
    special_price = models.FloatField(null=True, blank=True)
    url = models.URLField()
    image_url = models.URLField()
    description = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)


class Feedback(models.Model):
    content = models.TextField()
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    created_at = models.DateTimeField()
    rating = models.IntegerField()
    timestamp = models.DateTimeField()


class QnA(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
