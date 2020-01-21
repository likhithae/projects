from django.urls import path
from . import views
from django.contrib.auth.views import auth_login
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.requester, name='video-request'),
    path('p_videocall',views.p_home,name='p_videocall'),
    path('d_videocall',views.d_home,name='d_videocall'),
    path('mail', views.Mail),
    path('newrequest',views.new_request,name='new_request'),
    #path(''),
    path('call',views.login,name='call'),
    path('search/',views.search,name='search'),
    #path('login',login)
]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)