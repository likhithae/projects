from django.urls import include, path
from . import views

app_name = 'funding'

urlpatterns = [

    path('', views.index, name='index'),
    path('startproject', views.startproject, name='startproject'),
    path('fullstory/<int:pk>/', views.fullstory, name='fullstory'),
    path('login/', views.loginfund, name='login'),
    path('donate/<int:pk>', views.donate, name='donate'),
]