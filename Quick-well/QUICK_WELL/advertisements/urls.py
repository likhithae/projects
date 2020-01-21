from django.conf.urls import url
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'advertisements'

urlpatterns = [
    url(r'^$', views.advertisements, name='advertisements'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
