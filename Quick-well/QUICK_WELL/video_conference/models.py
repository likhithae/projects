from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class video_request(models.Model):
    #doctor = models.ForeignKey(User,on_delete=models.CASCADE)
    date_requested = models.DateTimeField(auto_now_add= True)
    request_id = models.AutoField(primary_key=True)
    requester = models.CharField(max_length=120)

class video_con_pass(models.Model):
    patient = models.CharField(max_length=120)
    date_conference = models.DateTimeField(auto_now_add=True)
    # doctor = models.ForeignKey(User,on_delete=models.CASCADE)
    conference_link = models.CharField(max_length=120)

