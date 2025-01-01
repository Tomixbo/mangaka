from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
import asyncio
import torch
from paddleocr import PaddleOCR
from TTS.api import TTS
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from scipy.io.wavfile import write
import os
from PIL import Image
import re

app = FastAPI()

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_semaphore = asyncio.Semaphore(1)  # Limit GPU to 1 concurrent task
cpu_executor = ThreadPoolExecutor(max_workers=4)  # Worker pool for CPU tasks

# YOLO model (CPU)
YOLO_MODEL_PATH = "models/panel_text_detection/panel_text_detection.pt"
yolo_model = YOLO(YOLO_MODEL_PATH) if os.path.exists(YOLO_MODEL_PATH) else None

# OCR model (GPU)
OCR_MODEL_DIR = "models/rec_french_manga_latin_4"
ocr = PaddleOCR(
    rec_model_dir=OCR_MODEL_DIR,
    use_angle_cls=True,
    lang="latin",
    show_log=False
)

# TTS model (GPU)
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_model = TTS(TTS_MODEL_NAME).to(device)

# File preprocessing
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return resized

# Detect panels and texts (CPU)
@app.post("/detect_panels_texts")
async def detect_panels_texts(file: UploadFile = File(...)):
    if not yolo_model:
        raise HTTPException(status_code=500, detail="YOLO model not initialized.")
    try:
        image = np.frombuffer(await file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image.")

        def detect_task():
            results = yolo_model.predict(source=image, conf=0.25, save=False)
            panels, texts = [], []
            for result in results[0].boxes.data:
                x_min, y_min, x_max, y_max, confidence, cls = result.tolist()
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                class_id = int(cls)
                if class_id == 0:
                    panels.append((x_min, y_min, x_max, y_max, confidence))
                elif class_id == 1:
                    texts.append((x_min, y_min, x_max, y_max, confidence))
            return {"panels": panels, "texts": texts}

        # Execute detection in CPU worker pool
        return await asyncio.get_event_loop().run_in_executor(cpu_executor, detect_task)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

# Text recognition (GPU)
@app.post("/text_recognition")
async def text_recognition(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)
        preprocessed_image = preprocess_image(image_np)

        if preprocessed_image is None:
            raise HTTPException(status_code=400, detail="Invalid preprocessed image.")

        async with gpu_semaphore:  # Restrict GPU access
            ocr_results = await asyncio.to_thread(ocr.ocr, preprocessed_image, cls=False)
            if not ocr_results or not isinstance(ocr_results[0], list):
                raise ValueError("No text detected.")

            lines = []
            for line in ocr_results[0]:
                original_line = str(line[1][0])
                cleaned_line = re.sub(r'^\s*[-_]+\s*|\s*[-_]+\s*$', '', original_line)
                if original_line.strip().endswith('-'):
                    lines.append(cleaned_line)
                else:
                    lines.append(cleaned_line + ' ')

            recognized_text = "".join(lines).strip()
            return {"recognized_text": recognized_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {e}")

# Text-to-speech (GPU)
@app.post("/text_to_speech")
async def text_to_speech(
    text: str = Form(...),
    speaker: str = Form("Damien Black"),
    language: str = Form("en")
):
    async with gpu_semaphore:  # Restrict GPU access
        try:
            # Si le texte est vide, générer un fichier audio avec du silence
            if not text.strip():
                # Générer 1 seconde de silence
                silence_duration = 2  # durée en secondes
                sample_rate = 22050  # fréquence d'échantillonnage
                silence = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)

                # Créer le buffer audio avec le silence
                audio_buffer = BytesIO()
                write(audio_buffer, sample_rate, silence)
                audio_buffer.seek(0)

                return StreamingResponse(
                    audio_buffer,
                    media_type="audio/wav",
                    headers={"Content-Disposition": "inline; filename=output.wav"}
                )

            # Si le texte n'est pas vide, générer l'audio avec le modèle TTS
            wav = await asyncio.to_thread(tts_model.tts, text=text, speaker=speaker, language=language)
            audio_buffer = BytesIO()
            write(audio_buffer, 22050, np.array(wav, dtype=np.float32))
            audio_buffer.seek(0)
            
            return StreamingResponse(
                audio_buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=output.wav"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS error: {e}")


# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}
