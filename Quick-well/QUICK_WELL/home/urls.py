from django.urls import re_path, include, path
from . import views

app_name = 'home'

urlpatterns = [

    path('', views.home, name='home'),
]