from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import generate_video_api, check_video_exists, download_video

urlpatterns = [
    path("generate-video/<uuid:manga_id>/", generate_video_api, name="generate_video"),
    path("check-video/<uuid:manga_id>/", check_video_exists, name="check_video"),
    path('download_video/<str:manga_id>/', download_video, name='download_video'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
