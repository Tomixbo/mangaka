from django.shortcuts import render, redirect,get_object_or_404
from django.views.decorators.csrf import csrf_exempt
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
from groq import Groq
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

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

async def generate_audio(session, recognized_text, manga_id, manga_page_id, panel_id):
    """
    Generates audio for a given text using the TTS API.

    Args:
        session (aiohttp.ClientSession): The session to use for API requests.
        recognized_text (str): The text to convert to audio.
        manga_id (UUID): The ID of the manga.
        manga_page_id (int): The ID of the manga page.
        panel_id (int): The ID of the panel.

    Returns:
        str or None: The relative path to the generated audio file, or None if failed.
    """
    if not recognized_text:
        print("No recognized text provided. Skipping audio generation.")
        return None

    # Prepare TTS payload and file paths
    tts_payload = {"text": recognized_text.replace(".", ","), "speaker": "Damien Black", "language": "fr"}
    audio_filename = f"audio_{manga_page_id}_{panel_id}.wav"
    audio_file_path = f"manga/{manga_id}/{audio_filename}"
    full_audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file_path)

    try:
        # Send TTS request
        async with session.post("http://127.0.0.1:5000/text_to_speech", data=tts_payload) as tts_response:
            if tts_response.status == 200:
                # Ensure directory exists and save the audio file
                os.makedirs(os.path.dirname(full_audio_file_path), exist_ok=True)
                with open(full_audio_file_path, 'wb') as audio_file:
                    audio_file.write(await tts_response.read())
                print(f"Audio file saved at: {audio_file_path}")
                return audio_file_path
            else:
                print(f"Failed to generate audio with status {tts_response.status}")
                return None
    except Exception as e:
        print(f"Error during audio generation: {e}")
        return None

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
        audio_file_path = await generate_audio(session, recognized_text, manga_id, manga_page_id, panel_id)


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
    panels_metadata = []  # Tableau pour stocker les métadonnées des panneaux
    for panel in panels:
        with panel.image.open('rb') as image_file:
            panels_data_all.append(base64.b64encode(image_file.read()).decode('utf-8'))
        panels_metadata.append({
            "panel_id": str(panel.id),  # Convertir UUID en chaîne
            "panel_order" : str(panel.order),
            "manga_page_id": str(panel.manga_page.id),  # Convertir UUID en chaîne
            "manga_id": str(manga.id)  # Convertir UUID en chaîne
        })

    # Texte reconnu
    text_data_all = [
        panel.recognized_text.split('\n') if panel.recognized_text else [] for panel in panels
    ]

    # Audios associés (urls ou null si inexistant)
    audio_data_all = [panel.audio_file.url if panel.audio_file else None for panel in panels]

    # Convertir les données en JSON pour le frontend
    panels_json = json.dumps(panels_data_all)
    panels_metadata_json = json.dumps(panels_metadata)  # Aucun problème ici maintenant
    texts_json = json.dumps(text_data_all)
    audios_json = json.dumps(audio_data_all)

    # Thumbnails à partir des images originales des pages (optionnel si était présent dans le view original)
    thumbnails = [page.original_image.name for page in manga.pages.all()]

    return render(request, 'read_manga/reader.html', {
        'panels': panels_json,
        'panels_metadata': panels_metadata_json,  # Ajouter les métadonnées au contexte
        'texts': texts_json,
        'audios': audios_json,
        'thumbnails': thumbnails,
    })



@csrf_exempt  # Nécessaire si CSRF est activé pour les requêtes AJAX
def update_text(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            panel_id = data.get("panel_id")
            panel_order = data.get("panel_order")
            manga_id = data.get("manga_id")
            manga_page_id = data.get("manga_page_id")
            corrected_texts = data.get("corrected_texts")

            # Log des données reçues
            print(f"Requête reçue : panel_id={panel_id}, panel_order={panel_order}, manga_id={manga_id}, manga_page_id={manga_page_id}")

            # Rechercher le panneau correspondant
            panel = Panel.objects.filter(
                id=panel_id,
                manga_page__id=manga_page_id,
                manga_page__manga__id=manga_id
            ).first()

            if not panel:
                return JsonResponse({"success": False, "message": "Panneau introuvable"})

            # Mettre à jour le texte du panneau
            panel.recognized_text = "\n".join(corrected_texts)
            panel.save()

            # Lancer la génération d'audio dans un thread
            def audio_generation_thread(panel, manga_id, manga_page_id, panel_order):
                async def generate_audio_task():
                    async with ClientSession() as session:
                        await generate_audio(session, panel.recognized_text, manga_id, manga_page_id, panel_order)

                # Créez un nouvel event loop pour le thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Exécutez la tâche asynchrone dans ce nouvel event loop
                    loop.run_until_complete(generate_audio_task())
                finally:
                    loop.close()

            # Lancer le traitement dans un thread
            threading.Thread(
                target=audio_generation_thread,
                args=(panel, manga_id, manga_page_id, panel_order),
                name="audio_generation_thread"
            ).start()

            return JsonResponse({"success": True, "message": "Texte mis à jour et audio en cours de génération."})
        except Exception as e:
            return JsonResponse({"success": False, "message": str(e)})

    return JsonResponse({"success": False, "message": "Invalid request method"})

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def validate_api_response(response_data):
    try:
        # Ensure the response is a dictionary with only keys like "text1", "text2", etc.
        if not isinstance(response_data, dict):
            return False
        for key, value in response_data.items():
            if not key.startswith("text") or not isinstance(value, str):
                return False
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def ai_correction(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            panel_id = data["panel_id"]
            manga_id = data["manga_id"]
            manga_page_id = data["manga_page_id"]

            # Fetch the panel and its text from the database
            # Rechercher le panneau correspondant
            panel = Panel.objects.filter(
                id=panel_id,
                manga_page__id=manga_page_id,
                manga_page__manga__id=manga_id
            ).first()
            panel_text = panel.recognized_text or ""  # Get the text or empty string if none
            panel_image_path = panel.image.path

            # Encode the image to base64
            base64_image = encode_image(panel_image_path)

            # Initialize the Groq client
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            retry_count = 3  # Set a retry limit
            response_content = None

            for attempt in range(retry_count):
                # Call the multimodal API
                completion = client.chat.completions.create(
                    model="llama-3.2-90b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"This is the recognized text in the image:\n {panel_text}\nYour task is to correct text recognition from Japanese manga images.\nText is read right to left, starting from the top-right. Read all text on the right column from top to bottom before moving progressively to the left. Treat linked bubbles as one.\nFix errors, including French spelling and grammar mistakes, count but do not paraphrase or alter the original meaning of the recognized text. Reorder text if mixed up and remove duplicates.\nPreserve French text, character names, manga-specific terms, and Japanese words.\nRespond **only** in JSON format, respect strictly the format as follows:\n```json\n{{\n  \"text1\": \"Corrected sentence 1\",\n  \"text2\": \"Corrected sentence 2\",\n  ...\n}}\n",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                    temperature=0,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"},
                    stop=None,
                )

                # Extract the response content
                response_content = completion.choices[0].message.content
                corrected_texts = json.loads(response_content)

                # Validate the response
                if validate_api_response(corrected_texts):
                    break  # Exit the loop if the response is valid
                else:
                    print(f"Attempt {attempt + 1}: Invalid response format. Retrying...", response_content)

            if not validate_api_response(corrected_texts):
                return JsonResponse({"success": False, "message": "Failed to get a valid response from the API after retries."})

            return JsonResponse({"success": True, "corrected_texts": corrected_texts})
        except Exception as e:
            return JsonResponse({"success": False, "message": str(e)})
    return JsonResponse({"success": False, "message": "Invalid request method"})