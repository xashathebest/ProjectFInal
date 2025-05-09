from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict, name='predict'),
    path('predict/', views.predict, name='predict_form'),
]