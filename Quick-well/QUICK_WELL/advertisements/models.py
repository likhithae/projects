from django.contrib.auth import get_user_model
from django.db import models

# Create your models here.
User = get_user_model()

class Advertisements(models.Model):
    Advertiser = models.CharField(max_length=100)
    email = models.EmailField()
    image = models.ImageField(upload_to='AdvertisementsImages/',default='AdvertisementImages/None/no-img.jpg')
    link = models.URLField()
