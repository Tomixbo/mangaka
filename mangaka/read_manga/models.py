from django.db import models
import os
import uuid
from django.db.models.signals import post_delete
import shutil 
from django.dispatch import receiver

def manga_directory_path(instance, filename):
    """
    Returns the file path for a manga's file.
    Files are stored in a folder named after the manga's ID.
    """
    return f'manga/{instance.manga.id}/{filename}'

class Manga(models.Model):
    """
    Represents a manga with its metadata and processing status.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField()
    status = models.CharField(
        max_length=50,
        choices=[('processing', 'Processing'), ('ready', 'Ready')],
        default='processing'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
    
   

class MangaPage(models.Model):
    """
    Represents a single page of a manga with its original image.
    """
    manga = models.ForeignKey(Manga, on_delete=models.CASCADE, related_name='pages')
    original_image = models.ImageField(upload_to=manga_directory_path)

    def __str__(self):
        return f"Page {self.id} for {self.manga.title}"

class Panel(models.Model):
    """
    Represents a specific panel within a manga page.
    """
    manga_page = models.ForeignKey(MangaPage, on_delete=models.CASCADE, related_name='panels')
    image = models.ImageField(upload_to=manga_directory_path)
    recognized_text = models.TextField(blank=True, null=True)
    audio_file = models.FileField(upload_to=manga_directory_path, blank=True, null=True)
    order = models.PositiveIntegerField(default=0)  # Field to store panel order

    class Meta:
        ordering = ['order']  # Ensure panels are always sorted by order

    def __str__(self):
        return f"Panel {self.order} for {self.manga_page.manga.title} - Page {self.manga_page.id}"

# Signal to handle file deletion when a Manga instance is deleted
@receiver(post_delete, sender=Manga)
def delete_manga_files(sender, instance, **kwargs):
    """
    Deletes the folder associated with the manga when the Manga instance is deleted.
    """
    manga_folder = os.path.join('media/manga', str(instance.id))
    if os.path.exists(manga_folder):
        shutil.rmtree(manga_folder)
        print(f"Deleted folder: {manga_folder}")

# Signal to handle file deletion when a MangaPage instance is deleted
@receiver(post_delete, sender=MangaPage)
def delete_manga_page_file(sender, instance, **kwargs):
    """
    Deletes the file associated with a MangaPage when the instance is deleted.
    """
    if instance.original_image:
        if os.path.exists(instance.original_image.path):
            os.remove(instance.original_image.path)
            print(f"Deleted file: {instance.original_image.path}")

# Signal to handle file deletion when a Panel instance is deleted
@receiver(post_delete, sender=Panel)
def delete_panel_files(sender, instance, **kwargs):
    """
    Deletes the files (image and audio) associated with a Panel when the instance is deleted.
    """
    if instance.image:
        if os.path.exists(instance.image.path):
            os.remove(instance.image.path)
            print(f"Deleted panel image: {instance.image.path}")
    if instance.audio_file:
        if os.path.exists(instance.audio_file.path):
            os.remove(instance.audio_file.path)
            print(f"Deleted audio file: {instance.audio_file.path}")