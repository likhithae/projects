from django.contrib import admin
from .models import *

admin.site.register(user_profile)
#admin.site.register(user_reports)
admin.site.register(Doctor)
# admin.site.register(Qualification)
# admin.site.register(Hospital_Affiliation)
# admin.site.register(Office)
# admin.site.register(Office_Docavailability)
admin.site.register(User_Review)
admin.site.register(Appointment)
admin.site.register(Appointment_Status)
admin.site.register(LabTest)
admin.site.register(Tests_info)
admin.site.register(Medicine)
admin.site.register(PurchaseItem)
admin.site.register(Order)
# Register your models here.
