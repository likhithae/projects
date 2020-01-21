from django.views import generic
from home.models import Medicine, PurchaseItem, Order, user_profile
from django.contrib.auth import login, logout
from django.contrib import messages
from django.http import HttpResponse
from django.views.generic.edit import CreateView
from django.views.generic.list import ListView
from django.shortcuts import render, get_object_or_404, redirect, reverse
from django.db.models import Q
import datetime
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.contrib.auth.models import User
import random
import string
from datetime import date
import datetime
from django.core.mail import send_mail

def generate_order_id():
    date_str = date.today().strftime('%Y%m%d')[2:] + str(datetime.datetime.now().second)
    rand_str = "".join([random.choice(string.digits) for count in range(3)])
    return date_str + rand_str




def get_user_pending_order(request):
    # get order for the correct user
    user_profile1 = get_object_or_404(user_profile, username=request.user)
    order = Order.objects.filter(user=user_profile1, is_ordered=False)
    if order.exists():
        # get the only order in the list of filtered orders
        return order[0]
    return 0


def index(request):
    all_medicines = Medicine.objects.all()
    query = request.GET.get('q')
    if query:
        all_medicines = Medicine.objects.filter(name__icontains=query)
        if not all_medicines:
            all_medicines = Medicine.objects.filter(pharmacy__icontains=query)
    else:
        all_medicines = Medicine.objects.all()
    return render(request, 'med/index.html', {'all_medicines': all_medicines})


class IndexView(generic.ListView):
    template_name = 'med/index.html'
    context_object_name = 'all_medicines'

    def get_queryset(self):
        return Medicine.objects.all()

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            else:
                return redirect('med:index')
    else:
        form = AuthenticationForm()
    return render(request, 'med/login.html', {'form': form})
#@login_required(login_url='/med/login/')
#def detail(request, medicine_id):
 #   medicine = get_object_or_404(Medicine, pk=medicine_id)
  #  return render(request, 'med/detail.html', {'medicine': medicine})

def logout_view(request):
    #if request.method == 'POST':
    all_medicines = Medicine.objects.all()
    logout(request)
    return render(request, 'med/index.html', {'all_medicines': all_medicines})


@login_required(login_url='/med/login/')
def add_to_cart(request, **kwargs):
    user_profile1 = get_object_or_404(user_profile, username=request.user)
    medicine = Medicine.objects.filter(id=kwargs.get('medicine_id', "")).first()
    quantity = request.GET.get('quantity')
    if medicine in request.user.user_profile.medicine.all():
        return redirect(reverse('med:index'))

    purchase_item, status = PurchaseItem.objects.get_or_create(medicine=medicine, quantity=quantity)
    #purchase_item, _ = PurchaseItem.objects.get_or_create(medicine=medicine, quantity=quantity)
    user_order, status = Order.objects.get_or_create(user=user_profile1, is_ordered=False)
    user_order.items.add(purchase_item)
    if status:
        # generate a reference code
        user_order.ref_code = generate_order_id()
        purchase_item.ref_code = generate_order_id()
        user_order.save()
        purchase_item.save()

    return redirect(reverse('med:index'))


def delete_from_cart(request, item_id):
    item_to_delete = PurchaseItem.objects.filter(pk=item_id)
    if item_to_delete.exists():
        item_to_delete[0].delete()
    return redirect(reverse('med:order_summary'))

def delete_order(request, order_id):
    order_to_delete = Order.objects.filter(pk=order_id)
    if order_to_delete.exists():
        order_to_delete[0].delete()
    return redirect(reverse('med:checked'))

#@login_required()
#def CartView(request, **kwargs):
 #   user_profile = get_object_or_404(UserProfile, user=request.user)
  #  medicine =
    #model = Medicine
    #fields = ['name', 'pharmacy', 'description', 'mfg_date', 'mfg_date', 'exp_date', 'price']


def checkout(request, order_id):
    order_purchased = Order.objects.filter(pk=order_id)
    order_purchased.date_ordered = datetime.datetime.now()
    address = request.GET.get('address')
    email = request.GET.get('email')
    order_purchased.update(email=email)
    order_purchased.update(billing_add=address)
    # order_items = order_to_purchase.items.all()
    order_purchased.update(is_ordered=True)
    order_purchased.update(date_ordered=datetime.datetime.now())
    send_mail("Your quickwell.com order has been confirmed","Thankyou for shopping at Quickwell!!...you will receive your order in 3 days","quickwelldoctor@gmail.com", [order_purchased[0].email])

    context = {
        'order': order_purchased[0],
    }
    return render(request, 'med/checkout.html', context)


def finalPrice(request, order_id):
    order_to_purchase = Order.objects.filter(pk=order_id)
    # order_to_purchase.is_ordered = True

    context = {
        'order': order_to_purchase[0],
    }
    return render(request, 'med/order.html', context)


@login_required(login_url='/med/login')
def order_details(request, **kwargs):
    existing_order = get_user_pending_order(request)
    context = {
        'order': existing_order
    }
    return render(request, 'med/order_summary.html', context)
#
#
#
# def get_user_checked_order(request):
#     # get order for the correct user
#     user_profile = get_object_or_404(UserProfile, user=request.user)
#     order = Order.objects.filter(user=user_profile, is_ordered=True)
#     if order.exists():
#         # get the only order in the list of filtered orders
#         return order
#     return 0
#
#
# @login_required(login_url='/med/login')
# def checked(request, **kwargs):
#     checked_order = get_user_checked_order(request)
#     context = {
#         'order': checked_order
#     }
#     return render(request, 'med/order_checked.html', context)


@login_required(login_url='/med/login')
def checked(request, **kwargs):
    user_profile1 = get_object_or_404(user_profile, username=request.user)

    order = Order.objects.filter(user=user_profile1, is_ordered=True)
    context = {
        'order': order
    }

    return render(request, 'med/order_checked.html', context)



#def my_profile(request):
#	my_user_profile = UserProfile.objects.filter(user=request.user).first()
#	my_orders = Order.objects.filter(is_ordered=True, owner=my_user_profile)
#	context = {
#		'my_orders': my_orders
#	}

#	return render(request, "profile.html", context)


