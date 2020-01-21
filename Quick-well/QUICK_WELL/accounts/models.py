
# from django.db import models
# from django.contrib.auth.models import User
#
# class Profile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     address = models.CharField(null=False, max_length=100)
#     experience = models.FloatField(null=True)
#     doc_photo = moels.FileField(null=True)
#     phone_num = models.BigIntegerField(null=True, blank=True)
#     # previous_hospitals = models.CharField(max_length=300,null=True)
#     spec = models.CharField(max_length=150)
#
#     def __str__(self):
#         return self.user.username

# class Post (models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE, null=True)
#     title = models.CharField(max_length=50)
#     body =models.TextField()
#     date =models.DateTimeField()
#
#     def __str__(self):
#         return self.title
