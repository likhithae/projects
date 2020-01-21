# from attr.filters import exclude
from django import forms
from django.contrib.auth.forms import User, UserCreationForm, UserChangeForm, PasswordChangeForm
from home.models import user_profile, User_Review
from home.models import *
from django.forms import ModelForm,Textarea,DateInput
import datetime

class Signup_form(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        username = self.cleaned_data.get('username')
        if email and User.objects.filter(email=email).exclude(username=username).exists():
            raise forms.ValidationError(u'Email id already exists')
        return email


class profile(forms.ModelForm):
    class Meta:
        model = user_profile
        fields = ('name', 'age', 'dob', 'email', 'contact_number', 'address', 'city', 'district', 'state', 'country',
                  'zipcode', 'photo')


#
# class upload(forms.ModelForm):
#     class Meta:
#         model = user_reports
#         fields = ('username', 'file')
#

class profile_update(forms.ModelForm):
    class Meta:
        model = user_profile
        fields = ('username', 'age', 'dob', 'email', 'contact_number', 'address', 'city', 'district', 'state', 'country', 'zipcode',
                  'photo')

class Patient_Update_Form(UserChangeForm):
    class Meta:
        model = user_profile
        fields = ('name', 'email', 'contact_number', 'address')

class passwordchange(PasswordChangeForm):
    class Meta:
        model = User
        fields = ('old_password', 'new_password1', 'new_password2')

class DoctorForm(ModelForm):
    class meta:
        model = Doctor
        fields = ['pub_date','doctor_name']

