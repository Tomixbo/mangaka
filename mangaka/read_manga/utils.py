import nltk
import language_tool_python
from paddleocr import PaddleOCR
import cv2
import os
from django.conf import settings
from TTS.api import TTS
import torch
import re

# Initialize TTS Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Télécharger les modèles nécessaires pour nltk
#nltk.download('punkt')

# Initialisation de PaddleOCR et LanguageTool
ocr = PaddleOCR(
    rec_model_dir=os.path.join(settings.BASE_DIR, 'models','rec_french_manga_latin_2'),
    use_angle_cls=True,
    show_log=False,
    lang='latin'
)
language_tool = language_tool_python.LanguageTool('fr')


def assign_text_to_panels(panels, texts):
    panel_texts = {i: [] for i in range(len(panels))}
    for text in texts:
        print(f"Debugging text: {text}")  # Debug statement to confirm the structure
        try:
            x_min_t, y_min_t, x_max_t, y_max_t = text  # Unpack directly, no slicing
        except ValueError as e:
            print(f"Error unpacking text {text}: {e}")
            continue
        for i, panel in enumerate(panels):
            x_min_p, y_min_p, x_max_p, y_max_p = panel
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
        print("No texts to sort in panel.")
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

    # Debugging: print each column
    for idx, column in enumerate(grouped_columns):
        print(f"Column {idx + 1}: {column}")

    # Flatten the columns into a single sorted list
    sorted_texts = [text for column in grouped_columns for text in column]

    print(f"Final sorted texts: {sorted_texts}")
    return sorted_texts




def preprocess_image(image):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return resized


def format_text_to_sentence_case(text):
    # Supprimer les séries de points '..', '...', '....', '.....'
    text = re.sub(r'\.{2,}', '', text)
    
    sentences = nltk.sent_tokenize(text, language='french')
    formatted_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            formatted_sentence = sentence[0].upper() + sentence[1:].lower()
            formatted_sentences.append(formatted_sentence)
    return ' '.join(formatted_sentences)


def correct_text_language_tool(text):
    corrected_text = language_tool.correct(text)
    return corrected_text
