# from django.db import models
# from django.conf import settings
# from django.contrib.auth.models import User
# from django.contrib.auth import get_user_model
# from django.db.models.signals import post_save
#
#
# class Medicine(models.Model):
#     name = models.CharField(max_length=100)
#     pharmacy = models.CharField(max_length=100)
#     about = models.CharField(max_length=1000, default='0000000')
#     description = models.CharField(max_length=1000)
#     mfg_date = models.DateField(null=True)
#     exp_date = models.DateField(null=True)
#     pres_req = models.CharField(max_length=100)
#     price = models.FloatField()
#
#     def __str__(self):
#         return self.name + ' - ' + self.pharmacy
#
#
# class PurchaseItem(models.Model):
#     ref_code = models.CharField(max_length=15, default='0000000')
#     medicine = models.ForeignKey(Medicine, on_delete=models.SET_NULL, null=True, unique=None)
#     quantity = models.IntegerField(null=True)
#     is_ordered = models.BooleanField(default=False)
#     date_added = models.DateTimeField(null=True)
#     date_ordered = models.DateTimeField(null=True)
#
#
#
#
# class UserProfile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     medicine = models.ManyToManyField(Medicine, blank=True)
#     first_name = models.CharField(max_length=100, default='0000000')
#     last_name = models.CharField(max_length=100, default='0000000')
#     bio = models.CharField(max_length=255, blank=True)
#     city = models.CharField(max_length=200, blank=False, default='')
#     interests = models.CharField(max_length=255, blank=True)
#
#
# def create_profile(sender, **kwargs):
#     if kwargs['created']:
#         user_profile = UserProfile.objects.create(user=kwargs['instance'])
#
#
# post_save.connect(create_profile, sender=User)
#
#
# class Order(models.Model):
#     ref_code = models.CharField(max_length=15)
#     user = models.ForeignKey(UserProfile, on_delete=models.SET_NULL, null=True)
#     items = models.ManyToManyField(PurchaseItem)
#     is_ordered = models.BooleanField(default=False)
#     date_ordered = models.DateTimeField(null=True)
#
#     def get_cart_items(self):
#         return self.items.all()
#
#     def get_cart_total(self):
#         sum = 0 ;
#         for item in self.items.all():
#             sum = sum + ((item.medicine.price)*(item.quantity))
#         return sum
#
#
#
#
#
#
#
#
