from django.urls import path
from apps import views


app_name = 'apps'
urlpatterns = [
    # path('cl/', views.form_index),
    path('classification/', views.classification),
    path('index/', views.index),
]
