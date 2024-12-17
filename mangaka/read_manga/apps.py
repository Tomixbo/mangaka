from django.apps import AppConfig
from TTS.api import TTS
import torch


class ReadMangaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "read_manga"

    def ready(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing TTS on device: {device}")

        # Ajouter TTS en tant qu'attribut de l'instance ReadMangaConfig
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
