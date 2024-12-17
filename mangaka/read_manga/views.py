from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import cv2
import base64
import json
from ultralytics import YOLO
from .utils import *


def detect_panels(image_path, model, size_threshold=50, overlap_threshold=0.8):
    """
    Détection des panels parents dans une image à l'aide de YOLO avec priorisation des parents.
    :param image_path: Chemin de l'image à traiter.
    :param model: Modèle YOLO préchargé.
    :param size_threshold: Taille minimale (largeur ou hauteur) pour qu'un panel soit considéré.
    :param overlap_threshold: Seuil de recouvrement pour considérer un panel comme enfant.
    :return: Liste des bounding boxes [(x_min, y_min, x_max, y_max)] des parents uniquement.
    """
    # Effectuer la détection avec le modèle YOLO préchargé
    results = model.predict(source=image_path, conf=0.25, save=False)

    # Extraire les bounding boxes
    panels = []
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()  # Convertir en liste
            width = x_max - x_min
            height = y_max - y_min

            # Appliquer un seuil de taille minimale
            if width >= size_threshold and height >= size_threshold:
                panels.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    # Identifier les parents
    parents = []
    for i, box1 in enumerate(panels):
        x1_min, y1_min, x1_max, y1_max = box1
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        is_parent = True  # Par défaut, suppose que c'est un parent

        for j, box2 in enumerate(panels):
            if i != j:
                x2_min, y2_min, x2_max, y2_max = box2
                area2 = (x2_max - x2_min) * (y2_max - y2_min)

                # Vérifier le chevauchement
                overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                overlap_area = overlap_x * overlap_y

                # Calculer le ratio de recouvrement relatif à box1
                overlap_ratio1 = overlap_area / area1
                # Calculer le ratio de recouvrement relatif à box2
                overlap_ratio2 = overlap_area / area2

                # Prioriser le parent : garder la boîte avec la plus grande aire
                if overlap_ratio1 > overlap_threshold and overlap_ratio2 > overlap_threshold:
                    if area1 < area2:  # Si box1 est plus petite, elle devient enfant
                        is_parent = False
                        break

        if is_parent:
            parents.append(box1)

    return parents

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
def process_uploaded_images(request):
    if request.method == 'POST':
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploaded_images'))
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
        audio_output_dir = os.path.join(settings.MEDIA_ROOT, 'audio_files')

        # Ensure the audio output directory exists
        os.makedirs(audio_output_dir, exist_ok=True)

        # Clear both uploaded_images and audio_files folders
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
                    if class_id == 0:  # Panneau
                        panels.append((x_min, y_min, x_max, y_max))
                    elif class_id == 1:  # Texte
                        texts.append((x_min, y_min, x_max, y_max))

            # Trier les panneaux selon le style manga
            sorted_panels = sort_boxes_manga_style(panels, vertical_threshold=100)

            # Associer les textes aux panneaux détectés
            panel_texts = assign_text_to_panels(sorted_panels, texts)

            # Traiter chaque panneau et ses textes associés
            for panel_id, texts_in_panel in panel_texts.items():
                x_min_p, y_min_p, x_max_p, y_max_p = sorted_panels[panel_id]
                panel_image = image[y_min_p:y_max_p, x_min_p:x_max_p]
                _, buffer = cv2.imencode('.jpg', panel_image)
                panels_data.append(buffer.tobytes())

                # **Sort texts in the panel for proper reading order**
                sorted_texts = sort_texts_in_panel(texts_in_panel)

                # Reconnaissance OCR et correction des textes dans le panneau
                processed_texts = []
                for text_box in sorted_texts:
                    x_min_t, y_min_t, x_max_t, y_max_t = text_box
                    cropped_text = image[y_min_t:y_max_t, x_min_t:x_max_t]
                    preprocessed_text = preprocess_image(cropped_text)
                    if preprocessed_text is not None:
                        ocr_result = ocr.ocr(preprocessed_text, cls=False)
                        if ocr_result and isinstance(ocr_result[0], list):
                            recognized_text = " ".join([line[1][0] for line in ocr_result[0]])
                            #corrected_text = correct_text_language_tool(format_text_to_sentence_case(recognized_text))
                            corrected_text = format_text_to_sentence_case(recognized_text)
                            processed_texts.append(corrected_text)
                # Generate audio for the panel
                page_index = image_paths.index(image_path)  # Get the index of the current page
                audio_filename = f"page_{page_index}_panel_{panel_id}.wav"
                audio_file_path = os.path.join(audio_output_dir, audio_filename)
                if processed_texts:
                    print('Start text to speech...')
                    tts.tts_to_file(
                        text=" ".join(processed_texts),
                        speaker="Damien Black",
                        language="fr",
                        file_path=audio_file_path
                    )
                    print(f'Text to speech : {audio_filename} - Done!')
                    audio_data.append(audio_filename)  # Save audio file path
                else:
                    audio_data.append(None)  # No audio for this panel

                text_data.append(processed_texts)

        # Convertir les données des panneaux en base64 pour le frontend
        panels_json = json.dumps([base64.b64encode(panel).decode('utf-8') for panel in panels_data])

        # Convertir les textes en JSON
        texts_json = json.dumps(text_data)
        audio_json = json.dumps(audio_data)

        return render(request, 'read_manga/reader.html', {
            'panels': panels_json,
            'texts': texts_json,
            'audios': audio_json,
            'thumbnails': [os.path.basename(path) for path in image_paths],
        })

    return render(request, 'read_manga/index.html')


