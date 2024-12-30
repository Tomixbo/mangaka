from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import io
import base64
import requests
import json
import cv2
import asyncio
from aiohttp import ClientSession
from .utils import (
    preprocess_image,
    remove_duplicate_boxes,
    filter_nested_panels,
    sort_boxes_manga_style,
    assign_text_to_panels,
    sort_texts_in_panel,
    format_text_to_sentence_case,
)

def index(request):
    return render(request, 'read_manga/index.html')


async def process_image(session, image_path, image, audio_output_dir, fs, panels_data_all, text_data_all, audio_data_all):
    # Step 1: Detect panels and text using API
    with open(image_path, 'rb') as img_file:
        async with session.post("http://127.0.0.1:5000/detect_panels_texts", data={"file": img_file}) as response:
            if response.status != 200:
                print(f"Detection API failed for {image_path}: {await response.text()}.")
                return

            detection_results = await response.json()
            panels = detection_results.get("panels", [])
            texts = detection_results.get("texts", [])

    # Step 2: Clean and sort panels and text
    unique_texts = remove_duplicate_boxes(texts, iou_threshold=0.5)
    filtered_panels = filter_nested_panels(panels)
    sorted_panels = sort_boxes_manga_style(filtered_panels, vertical_threshold=100)
    panel_texts = assign_text_to_panels(sorted_panels, unique_texts)

    local_panels_data = []
    local_text_data = []
    local_audio_data = []

    for panel_id, texts_in_panel in panel_texts.items():
        x_min_p, y_min_p, x_max_p, y_max_p = sorted_panels[panel_id][:4]
        panel_image = image[y_min_p:y_max_p, x_min_p:x_max_p]
        _, buffer = cv2.imencode('.jpg', panel_image)
        local_panels_data.append(buffer.tobytes())

        # Step 3: OCR for texts in panel
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

        local_text_data.append(processed_texts)

        # Step 4: Generate TTS audio
        page_index = os.path.basename(image_path)
        audio_filename = f"{page_index}_panel_{panel_id}.wav"
        audio_file_path = os.path.join(audio_output_dir, audio_filename)

        if processed_texts:
            modified_texts = " ".join(processed_texts).replace(".", ",")
            tts_payload = {"text": modified_texts, "speaker": "Damien Black", "language": "fr"}
            async with session.post("http://127.0.0.1:5000/text_to_speech", data=tts_payload) as response:
                if response.status == 200:
                    with open(audio_file_path, 'wb') as audio_file:
                        audio_file.write(await response.read())
                    local_audio_data.append(audio_filename)
                else:
                    local_audio_data.append(None)
        else:
            local_audio_data.append(None)

    # Accumulate results for all panels
    panels_data_all.extend(local_panels_data)
    text_data_all.extend(local_text_data)
    audio_data_all.extend(local_audio_data)


async def handle_uploaded_images(image_paths, fs, audio_output_dir):
    panels_data_all = []
    text_data_all = []
    audio_data_all = []

    async with ClientSession() as session:
        tasks = []
        for image_path in sorted(image_paths):
            full_image_path = os.path.join(fs.location, image_path)
            image = cv2.imread(full_image_path)
            if image is not None:
                tasks.append(
                    process_image(
                        session, full_image_path, image, audio_output_dir,
                        fs, panels_data_all, text_data_all, audio_data_all
                    )
                )

        # Run tasks in parallel
        await asyncio.gather(*tasks)

    return panels_data_all, text_data_all, audio_data_all


def process_uploaded_images(request):
    if request.method == 'POST':
        # Initialize directories
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploaded_images'))
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
        audio_output_dir = os.path.join(settings.MEDIA_ROOT, 'audio_files')

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(audio_output_dir, exist_ok=True)

        # Clear previous data
        for folder in [upload_dir, audio_output_dir]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Handle uploaded images
        uploaded_files = request.FILES.getlist('images')
        image_paths = [fs.save(uploaded_file.name, uploaded_file) for uploaded_file in uploaded_files]

        # Process images asynchronously
        panels_data_all, text_data_all, audio_data_all = asyncio.run(
            handle_uploaded_images(image_paths, fs, audio_output_dir)
        )

        # Convert results to JSON for frontend
        panels_json = json.dumps([base64.b64encode(panel).decode('utf-8') for panel in panels_data_all])
        texts_json = json.dumps(text_data_all)
        audios_json = json.dumps(audio_data_all)

        return render(request, 'read_manga/reader.html', {
            'panels': panels_json,
            'texts': texts_json,
            'audios': audios_json,
            'thumbnails': [os.path.basename(path) for path in image_paths],
        })

    return render(request, 'read_manga/index.html')
