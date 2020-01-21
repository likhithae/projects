from django import forms
from django.contrib.auth.forms import User
from django.contrib.auth.forms import UserCreationForm
from .models import *
class video_requestForm(forms.Form):
    email=forms.EmailField(widget=forms.TextInput(attrs={'placeholder':'Requester Email'}))
    username=forms.CharField(max_length=50,widget=forms.TextInput(attrs={'placeholder':'Requester Name'}))
