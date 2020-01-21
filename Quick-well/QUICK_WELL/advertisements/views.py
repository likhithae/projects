from django.contrib.auth.decorators import login_required
from django.shortcuts import render,redirect
from .models import Advertisements
from .forms import AdvertisementsForm

# Create your views here.
@login_required
def advertisements(request):
    if request.method == 'POST':
        form1 = AdvertisementsForm(request.POST,request.FILES)
        if form1.is_valid():
            form1.save()
            return redirect('http://127.0.0.1:8000')
    else:
        form1 = AdvertisementsForm()

    return render(request, 'advertisements/advertisements.html',{'form': form1})
