from django.contrib import admin
from django.urls import path, re_path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    #re_path('^form/$', views.profile, name="patient_profile.profile"),
    path('signup/', views.signup_view, name="patient_profile.signup"),
    path('create/', views.create, name="patient_profile.create"),
    path('login/', views.login, name="patient_profile.login"),
    path('logout/', auth_views.LogoutView.as_view(template_name=''), name="patient_profile.logout"),
    path('edit/', views.patient_update, name="patient_update"),
    path('verify/', views.mail_conf, name="patient_verify"),
    # path('change-password/', views.change_password, name="change_password"),
    path('details/', views.view_profile, name="patient_profile.details"),
    path('reports/', views.reports, name="patient_profile.reports"),
    path('upload/', views.upload, name="patient_profile.upload"),
    #path('userprofile/', views.userprofile,name="patient_profile.userprofile"),
    path('test/', views.test, name="patient_profile.test"),
    #path('patient_update/', views.patient_update, name="patient_update"),
    path('changepassword', views.change_password, name='changepassword'),
    path('appointment/', views.appointment1, name="patient_profile.appointment"),
    path('lappointment/', views.lappointment1, name="patient_profile.lappointment"),
    path('lab_test/', views.labtest, name="patient_profile.lab_test"),
    #path('review/', views.review, name='patient_profile.review'),

]

