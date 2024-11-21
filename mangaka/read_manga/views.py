from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import cv2
import base64
import json
from ultralytics import YOLO

def detect_panels2(image_path, size_threshold=50, overlap_threshold=0.8):
    """
    Détection des panels parents dans une image à l'aide de YOLO avec priorisation des parents.
    :param image_path: Chemin de l'image à traiter.
    :param size_threshold: Taille minimale (largeur ou hauteur) pour qu'un panel soit considéré.
    :param overlap_threshold: Seuil de recouvrement pour considérer un panel comme enfant.
    :return: Liste des bounding boxes [(x_min, y_min, x_max, y_max)] des parents uniquement.
    """
    # Charger le modèle YOLO
    model_path = os.path.join(settings.BASE_DIR, 'models', 'best_s.pt')  # Chemin vers votre modèle YOLO
    model = YOLO(model_path)

    # Effectuer la détection
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




# Détection des panels
def detect_panels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    for i, contour1 in enumerate(contours):
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        is_internal = False
        for j, contour2 in enumerate(contours):
            if i != j:
                x2, y2, w2, h2 = cv2.boundingRect(contour2)
                if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                    is_internal = True
                    break
        if not is_internal and w1 > 50 and h1 > 50:
            panels.append((x1, y1, x1 + w1, y1 + h1))
    return panels

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
def upload_images(request):
    if request.method == 'POST':
        print("Received POST request for file upload.")
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploaded_images'))
                # Vider le dossier 'uploaded_images' avant d'importer les nouveaux fichiers
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
        try:
            print(f"Clearing folder: {upload_dir}")
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error clearing folder {upload_dir}: {e}")

        uploaded_files = request.FILES.getlist('images')

        # Vérifier les fichiers reçus
        if not uploaded_files:
            print("No files received in the POST request.")
        else:
            print(f"Number of files received: {len(uploaded_files)}")

        # Stocker les fichiers importés
        image_paths = []
        for uploaded_file in uploaded_files:
            try:
                print(f"Saving file: {uploaded_file.name}")
                filename = fs.save(uploaded_file.name, uploaded_file)
                file_path = os.path.join(fs.location, filename)
                image_paths.append(file_path)
                print(f"File saved at: {file_path}")
            except Exception as e:
                print(f"Error saving file {uploaded_file.name}: {e}")

        panels_data = []

        # Process each image
        for image_path in sorted(image_paths):  # Sort images
            print(f"Processing image: {image_path}")
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_path}. Skipping.")
                continue

            # Detect panels in the image
            detected_boxes = detect_panels2(image)
            print(f"Number of panels detected in {image_path}: {len(detected_boxes)}")

            # **Order the detected boxes using sort_boxes_manga_style**
            sorted_boxes = sort_boxes_manga_style(detected_boxes, vertical_threshold=50)
            print(f"Panels sorted for {image_path}")

            # Now, extract the panels in the ordered sequence
            for box in sorted_boxes:
                x_min, y_min, x_max, y_max = box
                panel = image[y_min:y_max, x_min:x_max]
                # Resize panel and add white background here (we'll handle this in step 2)
                _, buffer = cv2.imencode('.jpg', panel)
                panels_data.append(buffer.tobytes())
                print(f"Panel extracted from {image_path}: {box}")

        # Convertir les images en base64 pour le template
        try:
            panels_data = [base64.b64encode(panel).decode('utf-8') for panel in panels_data]
            print(f"Total panels processed: {len(panels_data)}")
        except Exception as e:
            print(f"Error converting panels to base64: {e}")
        
        # Serialize panels_data to JSON
        try:
            panels_json = json.dumps(panels_data)
        except Exception as e:
            print(f"Error serializing panels data to JSON: {e}")
            panels_json = '[]'

        try:
            image_names = sorted(os.listdir(os.path.join(settings.MEDIA_ROOT, 'uploaded_images')))
            print(f"Total thumbnails generated: {len(image_names)}")
        except Exception as e:
            print(f"Error listing uploaded images: {e}")
            image_names = []

        return render(request, 'read_manga/reader.html', {
            'panels': panels_json,
            'thumbnails': image_names,
        })

    print("GET request received. Rendering index page.")
    return render(request, 'read_manga/index.html')
