from rest_framework import serializers
from .models import Product, Category, Review, QnA


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = '__all__'


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'


class ReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Review
        fields = '__all__'


class QnASerializer(serializers.ModelSerializer):
    class Meta:
        model = QnA
        fields = '__all__'
