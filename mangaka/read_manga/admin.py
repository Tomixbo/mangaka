from django.contrib import admin
from .models import Manga, MangaPage, Panel

class PanelInline(admin.TabularInline):
    """
    Inline to display panels within a manga page.
    """
    model = Panel
    fields = ('order', 'image', 'recognized_text', 'audio_file')
    readonly_fields = ('recognized_text', 'audio_file')
    extra = 0  # No extra empty rows

class MangaPageInline(admin.TabularInline):
    """
    Inline to display manga pages within a manga.
    """
    model = MangaPage
    fields = ('original_image',)
    extra = 0  # No extra empty rows

class MangaAdmin(admin.ModelAdmin):
    """
    Custom admin view for Manga model.
    """
    list_display = ('title', 'status', 'created_at', 'updated_at')
    list_filter = ('status', 'created_at', 'updated_at')
    search_fields = ('title',)
    inlines = [MangaPageInline]

class MangaPageAdmin(admin.ModelAdmin):
    """
    Custom admin view for MangaPage model.
    """
    list_display = ('manga', 'original_image')
    search_fields = ('manga__title',)
    inlines = [PanelInline]

class PanelAdmin(admin.ModelAdmin):
    """
    Custom admin view for Panel model.
    """
    list_display = ('manga_page', 'order', 'image', 'recognized_text', 'audio_file')
    readonly_fields = ('recognized_text', 'audio_file')
    search_fields = ('manga_page__manga__title',)

# Register the models and their custom admin views
admin.site.register(Manga, MangaAdmin)
admin.site.register(MangaPage, MangaPageAdmin)
admin.site.register(Panel, PanelAdmin)
