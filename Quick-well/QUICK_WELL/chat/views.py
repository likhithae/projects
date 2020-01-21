from django.contrib.auth.decorators import login_required
from django.shortcuts import render,redirect
from django.utils.safestring import mark_safe
import json
#from .forms import PatientdetailsForm
from django.db import models
from .models import Patientdetails
from django.http import HttpResponseRedirect
from advertisements.models import Advertisements
from home.models import Doctor,user_profile



# Create your views here.
@login_required(login_url='http://127.0.0.1:8000/login/')
def index(request):
    return render(request, 'chat/form.html', {})

@login_required(login_url='http://127.0.0.1:8000/login/')
def form(request):
    return render(request, 'chat/form.html')

@login_required
def consult(request):
    print("Request objects: {}".format(request.POST))
    #patient=request.POST['patient']
    patientname=request.POST['patientname']
    email=request.POST['email']
    symptoms = request.POST['symptoms']
    Department = request.POST['Department']
    all_Advertisements=Advertisements.objects.all()
    patientdetails = Patientdetails.objects.create(patientname=patientname,email=email,symptoms=symptoms,Department=Department)
    patientdetails.save()
    return render(request, 'chat/departments.html',{'all_Advertisements':all_Advertisements})


@login_required
def room(request, room_name):
    all_Doctors = Doctor.objects.all()
    all_user_profiles = user_profile.objects.all()
    return render(request, 'chat/room.html', {
        'room_name_json': mark_safe(json.dumps(room_name)),
        'username': mark_safe(json.dumps(request.user.username)),'all_Doctors':all_Doctors,'all_user_profiles':all_user_profiles
        },

    )

