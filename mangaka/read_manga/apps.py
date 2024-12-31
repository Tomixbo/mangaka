from django.apps import AppConfig

class ReadMangaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "read_manga"

    def ready(self):
        print("ReadManga app is ready")
