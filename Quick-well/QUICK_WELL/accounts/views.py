from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.http import HttpResponse
from .forms import *
from django.core.mail import send_mail, EmailMessage
from django.contrib.auth import get_user_model, login as auth_login, logout as out, update_session_auth_hash
from django.conf import settings
from home.models import Doctor, otp_verify
from django.contrib.auth import authenticate
from accounts.tokens import account_activation_token
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_text
from django.template.loader import render_to_string
from django.contrib.sites.shortcuts import get_current_site

import random

# def home(request):
#     return render(request, 'accounts/home.html')

def Mail(request,emailto):
   otp = random.randint(100000, 999999)
   res = send_mail("hello doctor", "Thanks for registering "+ str(otp) + " is your verification otp", "quickwelldoctor@gmail.com", [emailto], fail_silently=True)
   return HttpResponse('%s'%res)

def login(request):
    #
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            temp = 1
            doc = get_object_or_404(Doctor, user=request.user)
            print(request.user)
            print(doc)
            return redirect('http://127.0.0.1:8000/login/home/', {'doc': doc,'temp': temp})
        else:
            return HttpResponse("invalid details!")
    else:
        form = AuthenticationForm()
    return render(request, 'accounts/login.html', {'form': form,})

# def logout_view(request):
#     if request.method == 'POST':
#     out(request)
#     return render(request, 'accounts/index.html')
#
# def logout(request):
#     if request.method == "POST":
#         out(request)
#     return HttpResponse("logged out")

def register(request):
    registered = False
    if request.method == 'POST':
        user_form = Signup_user_form(request.POST)
        profile_form = Signup_profile_form(request.POST, request.FILES)
        if user_form.is_valid() and profile_form.is_valid():
            UserForm = user_form.save()
            UserForm.set_password(UserForm.password)
            UserForm.save()
            ProfileForm = profile_form.save(commit=False)
            ProfileForm.user = UserForm
            ProfileForm.save()
           # Mail(request, UserForm.email)
            otp = random.randint(100000, 999999)

            # current_site = get_current_site(request)
            # mail_subject = 'Verify your email.'
            # message = render_to_string('accounts/account_activation.html', {
            #     'user': ProfileForm.user,
            #     'domain': current_site.domain,
            #     'uid': urlsafe_base64_encode(force_bytes(ProfileForm.user.pk)).decode(),
            #     'token': account_activation_token.make_token(ProfileForm.user),
            # })
            # to_email = UserForm.email
            # email = EmailMessage(
            #     mail_subject, message, to=[to_email]
            # )
            # email.send()
            send_mail("hello doctor", "Thanks for registering " + str(otp) + " is your verification otp","quickwelldoctor@gmail.com", [UserForm.email])
            registered = True
            otpc = otp + 882413
        else:
            return HttpResponse("Invalid details!")
    else:
        user_form = Signup_user_form()
        profile_form = Signup_profile_form()
    if registered:
        request.session['username'] = user_form.cleaned_data.get('username')
        return render(request, 'accounts/mailconformation.html', {'otpc': otpc})
    # return HttpResponse("Successfully created your account")
    else:
        return render(request, 'accounts/signup4.html', {'user_form': user_form, 'profile_form': profile_form})

def index(request):
   # doc = Doctor.objects.all()
    query = request.GET.get('q')
    doc = get_object_or_404(Doctor, user=request.user)
    temp = 1
    return render(request, 'profile/includes/basic.html', {'temp': temp,'doc':doc})

def contact(request):
    temp = 1
    doc = get_object_or_404(Doctor, user=request.user)
    return render(request, 'profile/includes/basic.html', {'content': ['contact me at', 'sandeshjatla@gmail.com'], 'temp': temp, 'doc':doc})

def test(request):
    temp = 1
    doc = get_object_or_404(Doctor, user=request.user)
    return render(request, 'profile/includes/test.html', {'temp': temp, 'doc':doc})

    #     otp1 = request.POST.get('typed_otp')
    #     otp2 = otp_verify.objects.get(name=user).otp
    #     if otp1 == str(otp2):
    #         raw_password=request.POST.get('password1')
    #         new_user1=User.objects.create(username=username,email=email,first_name=firstname,last_name=lastname)
    #         new_user1.set_password(request.POST['password1'])
    #         new_teacher = Teacher(teacher=new_user1)
    #         new_user1.save()
    #         new_teacher.save()
    #         user = authenticate(username=username, password=raw_password)
    #         login(request, user)
    #         return redirect('result:choose_course_teacher')
    #     else:
    #         context={
    #             'firstname':firstname,
    #         }
    #         messages.error(request,'wrong otp')
    #         return render(request,'Result_Analysis/verify2.html',context)
    # else:
    #     return HttpResponse('wrong way')


def mail_conf(request):
    if request.method=='POST':
        otpc = int(request.POST['otpc'])
        otp1 = str(request.POST['otp1'])
        otpc = otpc - 882413
        otpc = str(otpc)
        if otpc == otp1:
            return redirect("http://127.0.0.1:8000/")
        else:
            username = request.session['username']
            dele = User.objects.get(username=username)
            dele.delete()
            return HttpResponse("mail unverified")
    else:
        return HttpResponse('404 error')

def doctor_update(request):
    temp = 1
    doc = get_object_or_404(Doctor, user=request.user)
    if request.method == 'POST':
        doc_form = Doctor_Update_Form(request.POST, instance=request.user.doctor or None)
        if doc_form.is_valid():
            doc_form.save()
            return redirect("http://127.0.0.1:8000/login/home")
    else:
        doc_form = Doctor_Update_Form(instance=request.user)
    return render(request, 'profile/includes/profile.html', {'doc_form':doc_form, 'temp':temp, 'doc':doc})

# def doc_acc_activation(request):
#     return render(request, 'accounts/acc_activate.html')

# def activate(request, uidb64, token):
#     try:
#         uid = force_text(urlsafe_base64_decode(uidb64))
#         user = User.objects.get(pk=uid)
#     except(TypeError, ValueError, OverflowError, User.DoesNotExist):
#         user = None
#     if user is not None and account_activation_token.check_token(user, token):
#         user.is_active = True
#         user.save()
#      #   login(request, user)
#         # return redirect('home')
#         return HttpResponse('Thank you for your email confirmation. Now you can login your account.')
#     else:
#         return HttpResponse('Activation link is invalid!')

def change_password(request):
    temp = 1
    doc = get_object_or_404(Doctor, user=request.user)
    if request.method== 'POST':
        change_form = passwordchange(data=request.POST, user=request.user)
        if change_form.is_valid():
            change_form.save()
            update_session_auth_hash(request,change_form.user)
            return redirect('http://127.0.0.1:8000/login/home/')
    else:
        change_form=passwordchange(user=request.user)
    return render(request, 'profile/includes/changepassword.html', {'change_form':change_form, 'temp':temp, 'doc':doc})

# def forgotemail(request):
#     if request.method == 'POST':
#         email=request.POST.get('email')
#         resetotp=send(email)
#         try:
#             username=User.objects.get(email=email).username
#         except:
#             messages.error(request,'no user exists with this email')
#             return render(request,'Result_Analysis/forget_email.html')
#
#         try:
#             user=otp_verify.objects.get(name=username)
#             user.otp=resetotp
#             user.save()
#         except:
#             user_otp=otp_verify.objects.create(name=username,otp=resetotp)
#             user_otp.save()
#         return render(request,'Result_Analysis/forget_otp.html',{'email':email})
#     else:
#         return render(request,'Result_Analysis/forget_email.html')

# def reset_password(request):
#     if request.method == 'POST':
#         email=request.POST.get('email')
#         username=User.objects.get(email=email).username
#         user=User.objects.get(username=username)
#         pass1=request.POST['resetpass1']
#         pass2=request.POST['resetpass2']
#         if pass1!=pass2:
#             return HttpResponse("wrong Details!")
#           #  messages.error(request,'passwords didnot match')
#          #   return render(request,'Result_Analysis/forget_password.html',{'email':email})
#         user.set_password(pass1)
#         user.save()
#         return HttpResponse('password reset successfully')
#     else:
#         return render(request,'accounts/forgot_password.html')