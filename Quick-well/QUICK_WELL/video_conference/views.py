from django.core.mail import send_mail, EmailMessage
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.contrib.auth.models import User
from django.urls import reverse_lazy
import random
import webbrowser
import os
from .forms import *
from .models import *
from home.models import *
from django.contrib.auth.decorators import login_required
@login_required(login_url=reverse_lazy('accounts:login'))

def p_home(request):
    return render(request,'video_conference/videoconference_p.html')
def d_home(request):
    return render(request,'video_conference/videoconference_d.html')

def requester(request):
    try:
        if request.user.user:
            return redirect('d_videocall')
    except:
        return redirect('p_videocall')

    '''if request.method == 'POST':
        form = video_requestForm(request.POST)
        if form.is_valid():
            t = video_request(requester = form.cleaned_data['username'])
            t.save()
            return render(request, 'video_conference/conference.html' , {'form':form})

    else:
        form = video_requestForm()
    return render(request,'video_conference/request.html',{'form':form})'''

def login(request):
    cd = random.randint(100000, 9999999)
    os.system("start \"\" http://appr.tc/r/"+str(cd))

def Mail(request):
    x = User.objects.all()
    lis=[]
    for a in x:
        lis.append(a.email)
    for i in range(0,len(lis)):
        res = send_mail("Health Tips", "Today's health tip : drink 8 cups of water minimum a day.", "qucikwelldoctor@gmail.com", [lis[i]])
    return HttpResponse('tip sent')


def new_request(request):
    return render(request,'video_conference/newrequest_p.html',{})


def search(request):
    query = request.POST['query123']
    temp = str(query).upper()
    k=0
    for doc in Doctor.objects.all():
        if doc.firstname.upper() == temp:
            k = doc.id
            print()
            print("media/" + str(doc.doc_photo))
            print()
    if(k==0):
        context = {
            'found_status' : False,
        }
    else:
        context = {
            'found_status' : True,
            'doctor' : get_object_or_404(Doctor,id=k),
        }
    return render(request,'video_conference/result.html',context)
'''
def email_one(request):
    subject = "I am a text email"
    to = ['jaswanth.b17@iiits.in']
    from_email = 'b.jaswanth6@gmail.in'

    ctx = {
        'user': 'buddy',
        'purchase': 'Books'
    }

    message = render_to_string('blog/email.txt', ctx)

    EmailMessage(subject, message, to=to, from_email=from_email).send()

    return HttpResponse('email_one')'''