from django.conf.urls import url
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from django.views.generic import ListView, DetailView
#from .models import Post

app_name = 'accounts'

urlpatterns = [
    url(r'^login/$', views.login, name="login"),


    url(r'changepassword', views.change_password, name='changepassword'),

    # url(r'changepassword', views.change_password, name='changepassword'),

#    url(r'changepassword', views.change_password, name='changepassword'),
    # url(r'login/changepassword',views.change_password, name='changepassword'),


    url(r'login/changepassword',views.change_password, name='changepassword'),
    url(r'^logout/$', auth_views.LogoutView.as_view(template_name=''), name="logout"),
    url(r'^signup4/$', views.register, name="signup4"),
    url(r'^login/home/', views.index, name='index'),
    url(r'^login/contact', views.contact, name='contact'),
    url(r'^login/test', views.test, name='test'),
    url(r'^temp/verify/$', views.mail_conf, name="verify"),
    url(r'^login/doctor_update/$', views.doctor_update, name="doctor_update"),
  #  path('activate/<uidb64>/<token>', views.activate, name='activate'),
    # url(r'forgotpassword/', views.forgot_mail, name='new_pass'),
    #    url(r'^advt', views.advt, name='advt'),
#    url(r'^blog', ListView.as_view(queryset=Post.objects.all().order_by("-date")[:10], template_name="profile/includes/blog.html")),
#    url(r'^blog/blog/(?P<pk>\d+)$', DetailView.as_view(model=Post, template_name='profile/includes/post.html'))
]