"""QUICK_WELL URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from machina.app import board
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, \
    PasswordResetCompleteView

urlpatterns = [
    path('', include('home.urls')),
    path('admin/', admin.site.urls),
    path('appointment/', include('docapp.urls')),
    path('patient_profile/', include('patient_profile.urls')),
    path('', include('accounts.urls')),
    path('chat/', include('chat.urls')),
    path('forum/', include(board.urls)),
    path('ads/', include('ads.urls')),
    path('med/', include('med.urls')),
    #path('credits/', include('credits.urls')),
    path('lab/', include('labtest.urls')),
    path('funding/',include('funding.urls')),
    path('advertisements/', include('advertisements.urls')),
    path('video_conference/', include('video_conference.urls')),
    path('', include('django.contrib.auth.urls')),


]
urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)