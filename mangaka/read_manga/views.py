from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import cv2
import base64
import json
from ultralytics import YOLO
from .utils import *
from django.apps import apps 
from concurrent.futures import ThreadPoolExecutor, as_completed


# Vue principale
def index(request):
    return render(request, 'read_manga/index.html')

def sort_boxes_manga_style(boxes, vertical_threshold=5):
    """
    Sort bounding boxes for manga-style reading: top to bottom, then right to left.
    The vertical threshold is used to group boxes into the same line.
    """
    if not boxes:
        return []

    # Step 1: Group boxes by their top position (y_min)
    boxes.sort(key=lambda b: b[1])  # Sort by y_min (top position)

    grouped_lines = []
    current_line = [boxes[0]]

    for i in range(1, len(boxes)):
        prev_box = current_line[-1]
        current_box = boxes[i]

        # Check if the current box is on the same vertical line
        if abs(current_box[1] - prev_box[1]) <= vertical_threshold:
            current_line.append(current_box)
        else:
            # Start a new line
            grouped_lines.append(current_line)
            current_line = [current_box]

    # Add the last line
    if current_line:
        grouped_lines.append(current_line)

    # Step 2: Sort each line horizontally (right to left)
    for line in grouped_lines:
        line.sort(key=lambda b: -b[0])  # Sort by x_min descending (right to left)

    # Step 3: Flatten the grouped lines back into a single sorted list
    sorted_boxes = [box for line in grouped_lines for box in line]

    return sorted_boxes

# Vue pour uploader et traiter les images
def process_uploaded_images_old(request):
    if request.method == 'POST':
        # Obtenir TTS à partir de la configuration de l'application
        tts = apps.get_app_config('read_manga').tts

        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploaded_images'))
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
        audio_output_dir = os.path.join(settings.MEDIA_ROOT, 'audio_files')

        # Assurer que les répertoires nécessaires existent
        os.makedirs(audio_output_dir, exist_ok=True)

        # Vider les dossiers uploaded_images et audio_files
        for folder in [upload_dir, audio_output_dir]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        uploaded_files = request.FILES.getlist('images')
        image_paths = [fs.save(uploaded_file.name, uploaded_file) for uploaded_file in uploaded_files]
        panels_data, text_data, audio_data = [], [], []

        # Charger le modèle YOLO une seule fois
        model_path = os.path.join(settings.BASE_DIR, 'models', 'panel_text_detection', 'panel_text_detection.pt')
        model = YOLO(model_path)

        for image_path in sorted(image_paths):
            full_image_path = os.path.join(fs.location, image_path)
            image = cv2.imread(full_image_path)
            if image is None:
                print(f"Image {image_path} could not be loaded.")
                continue

            height, width, _ = image.shape

            # Détection des panneaux et textes avec YOLO
            detected_boxes = model.predict(source=full_image_path, conf=0.25, save=False)

            panels = []
            texts = []
            for result in detected_boxes:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])  # Score de confiance
                    if class_id == 0:  # Panneau
                        panels.append((x_min, y_min, x_max, y_max, confidence))
                    elif class_id == 1:  # Texte
                        texts.append((x_min, y_min, x_max, y_max, confidence))

            # Supprimer les doublons de textes détectés
            unique_texts = remove_duplicate_boxes(texts, iou_threshold=0.5)

            # Supprimer les panneaux imbriqués
            filtered_panels = filter_nested_panels(panels)

            # Trier les panneaux selon le style manga
            sorted_panels = sort_boxes_manga_style(filtered_panels, vertical_threshold=100)

            # Associer les textes aux panneaux détectés
            panel_texts = assign_text_to_panels(sorted_panels, unique_texts)

            # Traiter chaque panneau et ses textes associés
            for panel_id, texts_in_panel in panel_texts.items():
                x_min_p, y_min_p, x_max_p, y_max_p = sorted_panels[panel_id][:4]
                panel_image = image[y_min_p:y_max_p, x_min_p:x_max_p]
                _, buffer = cv2.imencode('.jpg', panel_image)
                panels_data.append(buffer.tobytes())

                # Trier les textes dans le panneau pour respecter l'ordre de lecture
                sorted_texts = sort_texts_in_panel(texts_in_panel)

                # Reconnaissance OCR et correction des textes dans le panneau
                processed_texts = []
                for text_box in sorted_texts:
                    x_min_t, y_min_t, x_max_t, y_max_t = text_box[:4]  # Ignorer le score de confiance
                    cropped_text = image[y_min_t:y_max_t, x_min_t:x_max_t]
                    preprocessed_text = preprocess_image(cropped_text)
                    if preprocessed_text is not None:
                        ocr_result = ocr.ocr(preprocessed_text, cls=False)
                        if ocr_result and isinstance(ocr_result[0], list):
                            recognized_text = " ".join([re.sub(r'^\s*[-_]+\s*|\s*[-_]+\s*$', '', str(line[1][0])) for line in ocr_result[0]])
                            corrected_text = format_text_to_sentence_case(recognized_text)
                            processed_texts.append(corrected_text)

                # Générer l'audio pour le panneau
                page_index = image_paths.index(image_path)  # Obtenir l'indice de la page
                audio_filename = f"page_{page_index}_panel_{panel_id}.wav"
                audio_file_path = os.path.join(audio_output_dir, audio_filename)
                if processed_texts:
                    print('Start text to speech...')
                    modified_texts = " ".join(processed_texts).replace(".", ",")
                    print(modified_texts)
                    tts.tts_to_file(
                        text=modified_texts,
                        speaker="Damien Black",
                        language="fr",
                        file_path=audio_file_path
                    )
                    print(f'Text to speech : {audio_filename} - Done!')
                    audio_data.append(audio_filename)  # Enregistrer le chemin du fichier audio
                else:
                    audio_data.append(None)  # Pas d'audio pour ce panneau

                text_data.append(processed_texts)

        # Convertir les données des panneaux en base64 pour le frontend
        panels_json = json.dumps([base64.b64encode(panel).decode('utf-8') for panel in panels_data])

        # Convertir les textes et audios en JSON
        texts_json = json.dumps(text_data)
        audio_json = json.dumps(audio_data)

        return render(request, 'read_manga/reader.html', {
            'panels': panels_json,
            'texts': texts_json,
            'audios': audio_json,
            'thumbnails': [os.path.basename(path) for path in image_paths],
        })

    return render(request, 'read_manga/index.html')

def process_uploaded_images(request):
    if request.method == 'POST':
        # 1) Récupération du TTS + initialisation chemins
        tts = apps.get_app_config('read_manga').tts

        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploaded_images'))
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
        audio_output_dir = os.path.join(settings.MEDIA_ROOT, 'audio_files')

        # Assure la présence des dossiers
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(audio_output_dir, exist_ok=True)

        # Nettoie les anciens fichiers
        for folder in [upload_dir, audio_output_dir]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Récupère les fichiers uploadés
        uploaded_files = request.FILES.getlist('images')
        image_paths = [fs.save(uploaded_file.name, uploaded_file) for uploaded_file in uploaded_files]

        # 2) Charge le modèle YOLO et effectue le "warm-up"
        model_path = os.path.join(settings.BASE_DIR, 'models', 'panel_text_detection', 'panel_text_detection.pt')
        model = YOLO(model_path)

        # -- Warm-up sur une image bidon ou la première image, pour éviter les soucis de fusion BN en multi-thread --
        # => Utilisez un vrai petit fichier init.jpg si vous le souhaitez.
        dummy_image_path = os.path.join(settings.BASE_DIR, 'theme', 'static', 'img', 'init.jpg')
        if os.path.exists(dummy_image_path):
            try:
                _ = model.predict(source=dummy_image_path, conf=0.25, save=False)
            except Exception as e:
                print("Warm-up YOLO a rencontré un souci (modèle sans BN ?).", e)
        else:
            # si vous n'avez pas d'image bidon, vous pouvez ignorer
            pass

        # 3) Définir une fonction qui effectue la détection + OCR SANS TTS
        def detect_and_ocr_single_image(image_path, index):
            full_image_path = os.path.join(fs.location, image_path)
            image = cv2.imread(full_image_path)
            if image is None:
                print(f"Image {image_path} could not be loaded.")
                return [], [], []

            # Détection YOLO
            detected_boxes = model.predict(source=full_image_path, conf=0.25, save=False)

            # On sépare panneaux et textes
            panels = []
            texts = []
            for result in detected_boxes:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if class_id == 0:  # panneau
                        panels.append((x_min, y_min, x_max, y_max, confidence))
                    elif class_id == 1:  # texte
                        texts.append((x_min, y_min, x_max, y_max, confidence))

            # Nettoyage duplicatas et panneaux imbriqués
            unique_texts = remove_duplicate_boxes(texts, iou_threshold=0.5)
            filtered_panels = filter_nested_panels(panels)

            # Tri style manga
            sorted_panels = sort_boxes_manga_style(filtered_panels, vertical_threshold=100)

            # Associer textes aux panneaux
            panel_texts = assign_text_to_panels(sorted_panels, unique_texts)

            # Reconstruire les données : on stocke l'image des panneaux + le texte OCR (mais SANS TTS)
            local_panels_data = []
            local_text_data = []
            # on laisse l'audio_data vide ici, on fera la TTS plus tard
            local_audio_data = []

            for panel_id, texts_in_panel in panel_texts.items():
                x_min_p, y_min_p, x_max_p, y_max_p = sorted_panels[panel_id][:4]
                panel_image = image[y_min_p:y_max_p, x_min_p:x_max_p]
                _, buffer = cv2.imencode('.jpg', panel_image)
                local_panels_data.append(buffer.tobytes())

                # Tri des textes dans le panneau
                sorted_texts = sort_texts_in_panel(texts_in_panel)

                # OCR pur
                processed_texts = []
                for text_box in sorted_texts:
                    x_min_t, y_min_t, x_max_t, y_max_t = text_box[:4]
                    cropped_text = image[y_min_t:y_max_t, x_min_t:x_max_t]
                    preprocessed_text = preprocess_image(cropped_text)
                    if preprocessed_text is not None:
                        ocr_result = ocr.ocr(preprocessed_text, cls=False)
                        if ocr_result and isinstance(ocr_result[0], list):
                            #print(ocr_result)
                            lines = []
                            for line in ocr_result[0]:
                                original_line = str(line[1][0])  # Conservez l'original pour vérification
                                cleaned_line = re.sub(r'^\s*[-_]+\s*|\s*[-_]+\s*$', '', original_line)  # Nettoyez la ligne

                                if original_line.strip().endswith('-'):  # Vérifiez si l'original se termine par un tiret
                                    lines.append(cleaned_line)  # Pas d'espace ajouté
                                else:
                                    lines.append(cleaned_line + ' ')  # Ajoutez un espace pour les lignes normales

                            recognized_text = "".join(lines).strip()  # Joindre sans espace supplémentaire au début/fin
                            corrected_text = format_text_to_sentence_case(recognized_text)
                            processed_texts.append(corrected_text)

                local_text_data.append(processed_texts)

            return local_panels_data, local_text_data, local_audio_data

        # 4) On parallélise la DETECTION + OCR pour toutes les images
        results_sorted_by_index = [None] * len(image_paths)

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {}
            for i, img_path in enumerate(sorted(image_paths)):
                future = executor.submit(detect_and_ocr_single_image, img_path, i)
                future_to_index[future] = i

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                panels_data, text_data, audio_data = future.result()
                results_sorted_by_index[idx] = (panels_data, text_data, audio_data)

        # 5) On exécute maintenant la TTS de manière SEQUENTIELLE
        # => Pour chaque image, pour chaque "panneau", on génère l'audio si du texte existe.
        # On complète la liste audio_data_all
        panels_data_all = []
        text_data_all = []
        audio_data_all = []

        for i, (panels_data, text_data, audio_data) in enumerate(results_sorted_by_index):
            # panels_data = liste de bytes
            # text_data = liste de listes (textes OCR)
            # audio_data = liste (toujours vide jusqu'ici)
            updated_audio_data = []
            # pour chaque panneau
            for panel_id, panel_texts in enumerate(text_data):
                audio_filename = f"page_{i}_panel_{panel_id}.wav"
                audio_file_path = os.path.join(audio_output_dir, audio_filename)
                if panel_texts:
                    # TTS => on ne le fait pas en thread
                    modified_texts = " ".join(panel_texts).replace(".", ",")
                    tts.tts_to_file(
                        text=modified_texts,
                        speaker="Damien Black",
                        language="fr",
                        file_path=audio_file_path
                    )
                    updated_audio_data.append(audio_filename)
                else:
                    updated_audio_data.append(None)

            # on met à jour l'audio_data dans results
            # (audio_data était vide, on le remplace par updated_audio_data)
            audio_data = updated_audio_data

            # pour l'assemblage final
            panels_data_all.extend(panels_data)
            text_data_all.extend(text_data)
            audio_data_all.extend(audio_data)

            # On peut stocker au même index si besoin,
            # mais ici on concatène tout dans de grandes listes.

        # 6) On génère le JSON pour le template
        panels_json = json.dumps([base64.b64encode(p).decode('utf-8') for p in panels_data_all])
        texts_json = json.dumps(text_data_all)
        audios_json = json.dumps(audio_data_all)

        # 7) On retourne la page 'reader.html' avec le format EXACT
        return render(request, 'read_manga/reader.html', {
            'panels': panels_json,
            'texts': texts_json,
            'audios': audios_json,
            'thumbnails': [os.path.basename(path) for path in image_paths],
        })

    # Si la requête n'est pas POST, on affiche l'index
    return render(request, 'read_manga/index.html')