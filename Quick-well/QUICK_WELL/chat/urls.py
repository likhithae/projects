from django.conf.urls import url
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'chat'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^form/$', views.form, name='form'),
    url(r'^consult/$', views.consult, name='consult'),
    url(r'^(?P<room_name>[^/]+)/$', views.room, name='room'),


] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
