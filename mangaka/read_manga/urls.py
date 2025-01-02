from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Add Manga view
    path('add-manga/', views.add_manga, name='add_manga'),

    # Manga List view
    path('', views.manga_list, name='manga_list'),

    # Manga Reader view
    path('read-manga/<uuid:manga_id>/', views.reader, name='read_manga'),

    path('update-text/', views.update_text, name='update_text'),

    # Static media files
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
