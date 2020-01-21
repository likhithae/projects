from django.urls import re_path, include, path
from . import views

app_name = 'docapp'


urlpatterns = [

    path('index/', views.index, name='index'),
    path('<int:pk>/', views.appbooking, name='appbooking'),
    path('hospitalnearme/', views.maps, name='maps'),
    path('greet/', views.greet, name='greet'),
    path('profile/<int:pk>', views.profile, name='profile'),
    path('delete/<int:pk>',views.delete, name='delete'),
    path('login/', views.logindocapp, name='login'),

]