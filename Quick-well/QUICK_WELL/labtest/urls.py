from django.urls import re_path, include, path
from . import views

app_name = 'labtest'

urlpatterns = [

    path('index/', views.index, name='index'),
    path('<int:pk>', views.detail, name='details'),
    path('book/<int:pk>', views.labbook, name='labbook'),
    path('delete/<int:pk>',views.delete, name='delete'),
    path('login/', views.loginlabtest, name='login'),
]