from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
import torch
from paddleocr import PaddleOCR
from TTS.api import TTS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import re
from scipy.io.wavfile import write
import asyncio
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO initialization
YOLO_MODEL_PATH = "models/panel_text_detection/panel_text_detection.pt"
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
else:
    yolo_model = None
    print("YOLO model not found. Ensure the model file is available.")

# PaddleOCR initialization
OCR_MODEL_DIR = "models/rec_french_manga_latin_4"
ocr = PaddleOCR(
    rec_model_dir=OCR_MODEL_DIR,
    use_angle_cls=True,
    lang="latin",
    show_log=False
)

# TTS initialization
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_model = TTS(TTS_MODEL_NAME).to(device)

# Preprocessing function for OCR
def preprocess_image(image):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return resized

@app.post("/detect_panels_texts")
async def detect_panels_texts(file: UploadFile = File(...)):
    if not yolo_model:
        raise HTTPException(status_code=500, detail="YOLO model not initialized.")

    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to load image.")

    results = yolo_model.predict(source=image, conf=0.25, save=False)

    panels = []
    texts = []
    for result in results[0].boxes.data:
        x_min, y_min, x_max, y_max, confidence, cls = result.tolist()
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        class_id = int(cls)
        if class_id == 0:  # panneau
            panels.append((x_min, y_min, x_max, y_max, confidence))
        elif class_id == 1:  # texte
            texts.append((x_min, y_min, x_max, y_max, confidence))

    return {"panels": panels, "texts": texts}

@app.post("/text_recognition")
async def text_recognition(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)

        preprocessed_text = preprocess_image(image_np)
        if preprocessed_text is not None:
            ocr_results = ocr.ocr(preprocessed_text, cls=False)
            if ocr_results and isinstance(ocr_results[0], list):
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

        raise HTTPException(status_code=400, detail="No text detected.")
    except Exception as e:
        print("Erreur pendant le traitement OCR :", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text_to_speech")
async def text_to_speech(
    text: str = Form(...),
    speaker: str = Form("Damien Black"),
    language: str = Form("en")
):
    try:
        # Generate audio amplitudes
        wav = await asyncio.to_thread(tts_model.tts, text=text, speaker=speaker, language=language)

        # Convert amplitudes to WAV file in memory
        sample_rate = 22050
        audio_buffer = BytesIO()
        write(audio_buffer, sample_rate, np.array(wav, dtype=np.float32))
        audio_buffer.seek(0)

        # Return audio as a streaming response
        return StreamingResponse(audio_buffer, media_type="audio/wav", headers={"Content-Disposition": "inline; filename=output.wav"})

    except Exception as e:
        print("Error during TTS processing:", e)
        raise HTTPException(status_code=500, detail="TTS processing error: " + str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)