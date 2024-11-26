from django.urls import path
from .views import BirdPredictionAPIView

urlpatterns = [
    path('predict/', BirdPredictionAPIView.as_view(), name='predict'),
]
