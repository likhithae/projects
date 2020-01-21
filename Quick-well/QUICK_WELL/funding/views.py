from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.http import HttpResponse
from django.shortcuts import render,get_object_or_404,redirect
from django.core.mail import EmailMessage
from .forms import fundraiserForm
from home.models import fundraiser,user_profile


def index(request):
    fundraisers = fundraiser.objects.all()
    return render(request, 'funding/index.html',{'fundraiser': fundraisers})




def loginfund(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            else:
                return redirect('funding:index')
    else:
        form = AuthenticationForm()
    return render(request, 'funding/login.html', {'form': form})


@login_required(login_url='/funding/login/')
def startproject(request):
    user = get_object_or_404(user_profile, username=request.user)

    if request.method == 'POST':
        form = fundraiserForm(request.POST,request.FILES)

        if form.is_valid():

            category = form.cleaned_data['category']
            Title = form.cleaned_data['Title']
            goal_amount = form.cleaned_data['goal_amount']
            beneficiary_name = form.cleaned_data['beneficiary_name']
            beneficary_relation = form.cleaned_data['beneficiary_relation']
            Fundraiser_story = form.cleaned_data['Fundraiser_story']
            End_date = form.cleaned_data['End_date']
            photo = form.cleaned_data['photo']
            account_number = form.cleaned_data['account_number']
            accountholder_name = form.cleaned_data['accountholder_name']
            ifsc_code = form.cleaned_data['ifsc_code']
            temp = fundraiser.objects.create(user_name=user, category=category, Title=Title, goal_amount=goal_amount, beneficiary_name=beneficiary_name, beneficiary_relation=beneficary_relation, Fundraiser_story=Fundraiser_story, End_date=End_date, photo=photo,accountholder_name=accountholder_name, account_number=account_number,ifsc_code=ifsc_code)

            return render(request, 'funding/greet.html', {'form': form,'name': user.username})

        else:
            print(form.errors)

    else:
        form = fundraiserForm()

    return render(request, 'funding/startproject.html', {'form': form})


def fullstory(request,pk):
    fundraise = get_object_or_404(fundraiser, pk=pk)
    return render(request, 'funding/fullstory.html', {'fundraiser': fundraise})


def donate(request,pk):
    fundraise = get_object_or_404(fundraiser,pk=pk)
    return render(request, 'funding/donate.html', {'fundraiser':fundraise})