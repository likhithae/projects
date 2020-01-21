from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render,get_object_or_404,redirect
from home.models import LabTest, Tests_info, labAppointment, user_profile
from django.core.mail import EmailMessage
from .forms import labAppointmentForm
#from . import ref
from django.template.loader import render_to_string

# Create your views here.


def index(request):
    test = Tests_info.objects.all()
    query = request.GET.get('q')
    if query:
        test = Tests_info.objects.filter(test_name__icontains=query)
        if not test:
            test = Tests_info.objects.filter(cost__icontains=query)
    else:
        test = Tests_info.objects.all()

    return render(request, 'labtest/index.html', {'test':test})


def detail(request,pk):
    test = get_object_or_404(Tests_info, pk=pk)
    testname = test.test_name;
    print(testname)
    lab = LabTest.objects.filter(tests_available__test_name__contains=testname)
    return render(request, 'labtest/details.html', {'lab':lab,'test':test})

def loginlabtest(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            else:
                return redirect('labtest:index')
    else:
        form = AuthenticationForm()
    return render(request, 'labtest/login.html', {'form': form})



@login_required(login_url='/lab/login/')
def labbook(request,pk):
    test = get_object_or_404(Tests_info, pk=pk)
    testname = test.test_name
    lab = LabTest.objects.filter(tests_available__test_name__icontains=testname)
    testid = test.id
    error_msg=''
    user = get_object_or_404(user_profile, username=request.user)

    if request.method == 'POST':
        form = labAppointmentForm(request.POST)

        if form.is_valid():
            date = form.cleaned_data['date']
            time = form.cleaned_data['time']

            sametest = labAppointment.objects.filter(test_id__id__icontains=testid)
            #print(sametest)
            flag = 1

            for slot in sametest:
                if slot.date == date and slot.time == time:
                    error_msg = 'Sorry, this slot is not available please select a different slot!'
                    flag = 0
                    print(date)
                    break;


            if flag ==1:
                temp = labAppointment.objects.create(date=date, time=time, user_name=user, test_id=test)

                email_id = user.email
                user_name = user.username

                subject = "Appointment booked"
                to_email = email_id
                print(to_email)
                context = {
                    'name': user_name,
                    'time':time,
                    'date': date,
                    'test':test,
                    'appid': temp.id,
                    'lab': lab,
                }
                message = render_to_string('labtest/emailtxt.html', context)
                msg = EmailMessage(subject, message, to=[to_email])
                msg.content_subtype = 'html'

                try:
                    msg.send()
                    print('Successful')
                except:
                    print('Unsuccessful')

                return render(request, 'labtest/greet.html', context)

    else:
        form = labAppointmentForm()

    return render(request, 'labtest/booking.html', {'form': form,'test':test,'error':error_msg,'lab': lab})


def delete(request,pk):
    appointment = get_object_or_404(labAppointment, pk=pk)
    email_id = appointment.user_name.email;
    context={
        'test': appointment.test_id,
        'name': appointment.user_name.username,
        'time': appointment.time,
        'date': appointment.date,
    }
    appointment.delete()
    subject = "Appointment Cancelled"
    to_email = email_id
    print(to_email)
    message = render_to_string('labtest/cancelemail.html', context)
    msg = EmailMessage(subject, message, to=[to_email])
    msg.content_subtype = 'html'

    try:
        msg.send()
        print('Successful')
    except:
        print('Unsuccessful')

    return render(request, 'labtest/cancel.html', context)
