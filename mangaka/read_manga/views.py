from django.shortcuts import render, redirect,get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from aiohttp import ClientSession
import asyncio
import os
import cv2
from asgiref.sync import sync_to_async
from .models import Manga, MangaPage, Panel
from .utils import (
    preprocess_image,
    remove_duplicate_boxes,
    filter_nested_panels,
    sort_boxes_manga_style,
    assign_text_to_panels,
    sort_texts_in_panel,
    format_text_to_sentence_case,
)
import io
import json
import base64
import threading

def manga_list(request):
    """
    View to display the list of mangas added by the user.
    """
    mangas = Manga.objects.all().order_by('-created_at')
    # Ajouter une image de couverture pour chaque manga
    for manga in mangas:
        first_page = manga.pages.first()  # Récupère la première page associée au manga
        manga.cover_image_url = first_page.original_image.url if first_page else None

    return render(request, 'read_manga/manga_list.html', {'mangas': mangas})



def add_manga(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description')
        uploaded_files = request.FILES.getlist('images')

        if not title or not description or not uploaded_files:
            return render(request, 'read_manga/add_manga.html', {
                'error_message': "Tous les champs sont obligatoires."
            })

        manga = Manga.objects.create(title=title, description=description, status='processing')

        for uploaded_file in uploaded_files:
            MangaPage.objects.create(manga=manga, original_image=uploaded_file)

        # Lancer le traitement dans un thread
        threading.Thread(target=asyncio.run, args=(process_manga_images(manga.id),)).start()

        return redirect('manga_list')

    return render(request, 'read_manga/add_manga.html')

async def process_manga_images(manga_id):
    """
    Background processing for manga images.
    """
    manga = await sync_to_async(Manga.objects.get)(id=manga_id)
    manga_pages = await sync_to_async(list)(MangaPage.objects.filter(manga=manga))

    async with ClientSession() as session:
        tasks = [process_image(session, manga_id, manga_page) for manga_page in manga_pages]
        await asyncio.gather(*tasks)

    # Update manga status to READY after processing
    manga.status = 'ready'
    await sync_to_async(manga.save)()

async def process_image(session, manga_id, manga_page):
    """
    Process a single manga page: detect panels, recognize text, and generate audio.
    """
    # Étape 1: Vérification du chemin de l'image
    image_path = manga_page.original_image.path
    manga_page_id = manga_page.id  # Précharger l'identifiant
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    print(f"Processing image at: {image_path}")
    image = cv2.imread(image_path)

    # Étape 2: Appel à l'API pour détecter les panneaux et textes
    with open(image_path, 'rb') as img_file:
        async with session.post("http://127.0.0.1:5000/detect_panels_texts", data={"file": img_file}) as response:
            if response.status != 200:
                raise RuntimeError(f"API call failed for {image_path} with status {response.status}: {await response.text()}")

            detection_results = await response.json()
            panels = detection_results.get("panels", [])
            texts = detection_results.get("texts", [])
            print(f"Panels: {panels}, Texts: {texts}")

    # Étape 3: Validation et nettoyage des panneaux
    unique_panels = remove_duplicate_boxes(panels)
    filtered_panels = filter_nested_panels(unique_panels)
    sorted_panels = sort_boxes_manga_style(filtered_panels)

    # Assigner les textes aux panneaux
    panel_texts = assign_text_to_panels(sorted_panels, texts)

    print(f"Sorted panels: {sorted_panels}")
    print(f"Assigned texts to panels: {panel_texts}")

    # Étape 4: Traitement des panneaux
    for panel_id, texts_in_panel in panel_texts.items():
        x_min_p, y_min_p, x_max_p, y_max_p = sorted_panels[panel_id][:4]
        panel_image = image[y_min_p:y_max_p, x_min_p:x_max_p]
        _, buffer = cv2.imencode('.jpg', panel_image)

        # Sauvegarder l'image du panneau
        panel_image_path = f'manga/{manga_id}/panel_{manga_page_id}_{panel_id}.jpg'
        full_panel_image_path = os.path.join(settings.MEDIA_ROOT, panel_image_path)
        os.makedirs(os.path.dirname(full_panel_image_path), exist_ok=True)
        cv2.imwrite(full_panel_image_path, panel_image)

        print(f"Panel image saved at: {full_panel_image_path}")

        # Étape 5: Reconnaissance de texte pour chaque panneau
        sorted_texts = sort_texts_in_panel(texts_in_panel)
        processed_texts = []

        for text_box in sorted_texts:
            x_min_t, y_min_t, x_max_t, y_max_t = text_box[:4]
            cropped_text = image[y_min_t:y_max_t, x_min_t:x_max_t]
            preprocessed_text = preprocess_image(cropped_text)

            if preprocessed_text is not None:
                _, buffer = cv2.imencode('.jpg', preprocessed_text)
                async with session.post(
                    "http://127.0.0.1:5000/text_recognition",
                    data={"file": io.BytesIO(buffer)}
                ) as response:
                    if response.status == 200:
                        recognized_text = (await response.json()).get("recognized_text", "")
                        corrected_text = format_text_to_sentence_case(recognized_text)
                        processed_texts.append(corrected_text)

        # Combiner les textes traités
        recognized_text = "\n".join(processed_texts)
        print(f"Recognized text for panel {panel_id}: {recognized_text}")

        # Étape 6: Génération de l'audio
        audio_file_path = None
        if recognized_text:
            tts_payload = {"text": recognized_text.replace(".", ","), "speaker": "Damien Black", "language": "fr"}
            audio_filename = f"audio_{manga_page_id}_{panel_id}.wav"
            audio_file_full_path = os.path.join(settings.MEDIA_ROOT, f"manga/{manga_id}/{audio_filename}")

            async with session.post("http://127.0.0.1:5000/text_to_speech", data=tts_payload) as tts_response:
                if tts_response.status == 200:
                    os.makedirs(os.path.dirname(audio_file_full_path), exist_ok=True)
                    with open(audio_file_full_path, 'wb') as audio_file:
                        audio_file.write(await tts_response.read())
                    audio_file_path = f"manga/{manga_id}/{audio_filename}"
                    print(f"Audio file saved at: {audio_file_path}")
                else:
                    print(f"Failed to generate audio for panel {panel_id} with status {tts_response.status}")

        # Étape 7: Enregistrement dans la base de données
        await sync_to_async(Panel.objects.create)(
            manga_page=manga_page,
            image=panel_image_path,
            recognized_text=recognized_text,
            audio_file=audio_file_path,
            order=panel_id
        )
        print(f"Panel {panel_id} saved to database.")



def reader(request, manga_id):
    # Récupérer le manga correspondant
    manga = get_object_or_404(Manga, id=manga_id)

    # Récupérer toutes les pages associées au manga, triées par ID ou autre champ pertinent
    pages = manga.pages.order_by('id')  # Assurez-vous que 'id' représente l'ordre correct des pages

    # Récupérer les panneaux associés, triés par page et par ordre des panneaux
    panels = Panel.objects.filter(manga_page__in=pages).order_by('manga_page_id', 'order')

    # Génération des données Base64 pour les images de panels
    panels_data_all = []
    for panel in panels:
        with panel.image.open('rb') as image_file:
            panels_data_all.append(base64.b64encode(image_file.read()).decode('utf-8'))

    # Texte reconnu
    text_data_all = [
        panel.recognized_text.split('\n') if panel.recognized_text else [] for panel in panels
    ]

    # Audios associés (urls ou null si inexistant)
    audio_data_all = [panel.audio_file.url if panel.audio_file else None for panel in panels]

    # Convertir les données en JSON pour le frontend
    panels_json = json.dumps(panels_data_all)
    texts_json = json.dumps(text_data_all)
    audios_json = json.dumps(audio_data_all)

    # Thumbnails à partir des images originales des pages (optionnel si était présent dans le view original)
    thumbnails = [page.original_image.name for page in manga.pages.all()]

    return render(request, 'read_manga/reader.html', {
        'panels': panels_json,
        'texts': texts_json,
        'audios': audios_json,
        'thumbnails': thumbnails,
    })
