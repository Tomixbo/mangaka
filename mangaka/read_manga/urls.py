from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('add-manga/', views.add_manga, name='add_manga'),
    path('', views.manga_list, name='manga_list'),
    path('read-manga/<uuid:manga_id>/', views.reader, name='read_manga'),
    path('update-text/', views.update_text, name='update_text'),
    path('ai-correction/', views.ai_correction, name='ai_correction'),

    # Static media files
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
