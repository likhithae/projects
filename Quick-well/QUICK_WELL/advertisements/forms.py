from django import forms
from .models import Advertisements

class AdvertisementsForm(forms.ModelForm):
    class Meta:
        model = Advertisements
        fields = ['Advertiser','email','image','link']


