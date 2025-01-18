import cv2
import re
import numpy as np
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')


def preprocess_image(image):
    """
    Preprocess an image for OCR:
    - Convert to grayscale
    - Resize for better OCR accuracy
    """
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return resized


def remove_duplicate_boxes(boxes, iou_threshold=0.5):
    """
    Remove duplicate bounding boxes based on Intersection over Union (IoU).
    """
    def iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1[:4]
        x2_min, y2_min, x2_max, y2_max = box2[:4]

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    unique_boxes = []
    for box in boxes:
        is_duplicate = any(iou(box, unique_box) > iou_threshold for unique_box in unique_boxes)
        if not is_duplicate:
            unique_boxes.append(box)

    return unique_boxes


def filter_nested_panels(panels, overlap_threshold=0.8):
    """
    Filter out nested panels by keeping only parent panels.
    """
    def calculate_overlap(panel1, panel2):
        x1_min, y1_min, x1_max, y1_max = panel1[:4]
        x2_min, y2_min, x2_max, y2_max = panel2[:4]

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Calculate the area of panel1
        panel1_area = (x1_max - x1_min) * (y1_max - y1_min)

        return inter_area / panel1_area if panel1_area > 0 else 0

    filtered_panels = []
    for panel in panels:
        is_nested = any(calculate_overlap(panel, other_panel) > overlap_threshold for other_panel in panels if panel != other_panel)
        if not is_nested:
            filtered_panels.append(panel)

    return filtered_panels


def sort_boxes_manga_style(boxes, vertical_threshold=5):
    """
    Sort bounding boxes for manga-style reading: top to bottom, then right to left.
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


def assign_text_to_panels(panels, texts):
    """
    Assign texts to the panels they belong to.
    """
    panel_texts = {i: [] for i in range(len(panels))}
    for text in texts:
        x_min_t, y_min_t, x_max_t, y_max_t = text[:4]
        for i, panel in enumerate(panels):
            x_min_p, y_min_p, x_max_p, y_max_p = panel[:4]
            if (x_min_t >= x_min_p and y_min_t >= y_min_p and
                x_max_t <= x_max_p and y_max_t <= y_max_p):
                panel_texts[i].append(text)
                break
    return panel_texts


def sort_texts_in_panel(texts, horizontal_threshold=50):
    """
    Sort texts within a panel for manga-style reading:
    - Group texts by columns (sorted from right to left).
    - Sort texts within each column from top to bottom.
    """
    if not texts:
        return []

    # Sort texts by x_min (from right to left)
    texts.sort(key=lambda t: t[0], reverse=True)

    grouped_columns = []
    current_column = [texts[0]]

    # Group texts into columns
    for i in range(1, len(texts)):
        prev_text = current_column[-1]
        current_text = texts[i]

        # Check if the current text belongs to the same column
        if abs(current_text[0] - prev_text[0]) <= horizontal_threshold:
            current_column.append(current_text)
        else:
            grouped_columns.append(current_column)
            current_column = [current_text]

    # Add the last column
    if current_column:
        grouped_columns.append(current_column)

    # Sort each column top to bottom by y_min
    for column in grouped_columns:
        column.sort(key=lambda t: t[1])

    # Flatten the columns into a single sorted list
    sorted_texts = [text for column in grouped_columns for text in column]

    return sorted_texts

def prioritize_punctuation(text):
    def replace_match(match):
        # Extraire les caractères uniques de ponctuation dans l'ordre d'apparition
        punctuations = set(match.group(0))
        # Définir la priorité des ponctuations
        priority = ['?', '!', ',', '.']
        # Trouver la première ponctuation correspondant à la priorité
        for p in priority:
            if p in punctuations:
                return p
        return '.'  # Par défaut, retourner un point si aucune priorité n'est trouvée

    # Remplacer selon la logique de priorité
    return re.sub(r'[.,!?]+', replace_match, text)

def format_text_to_sentence_case(text):
    """
    Format text to sentence case.
    """
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = prioritize_punctuation(text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Maj and Min
    sentences = nltk.sent_tokenize(text, language='french')
    formatted_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            formatted_sentence = sentence[0].upper() + sentence[1:].lower()
            formatted_sentences.append(formatted_sentence)
    return ' '.join(formatted_sentences)

def format_text_after_AI(text):
    """
    Format text to sentence case.
    """
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = prioritize_punctuation(text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text