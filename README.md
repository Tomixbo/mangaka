# FastAPI Application for YOLO, PaddleOCR, and TTS

This application provides APIs for detecting panels and texts in images, recognizing text, and generating speech audio. It is built using FastAPI and supports asynchronous operations.

## Endpoints

### 1. Detect Panels and Texts
**Endpoint**: `/detect_panels_texts`  
**Method**: POST  
**Description**: Detects panels and texts in an uploaded image using YOLO.

**Request Parameters**:
- `file` (required): Image file to be processed. (Content-Type: `multipart/form-data`)

**Response**:
```json
{
  "panels": [
    [x_min, y_min, x_max, y_max, confidence],
    ...
  ],
  "texts": [
    [x_min, y_min, x_max, y_max, confidence],
    ...
  ]
}
```

### 2. Text Recognition
**Endpoint**: `/text_recognition`  
**Method**: POST  
**Description**: Recognizes text from an uploaded image using PaddleOCR.

**Request Parameters**:
- `file` (required): Image file to be processed. (Content-Type: `multipart/form-data`)

**Response**:
```json
{
  "recognized_text": "Extracted text from the image."
}
```

### 3. Text to Speech
**Endpoint**: `/text_to_speech`  
**Method**: POST  
**Description**: Converts text to speech audio using a TTS model.

**Request Parameters**:
- `text` (required): The text to be converted to speech. (Content-Type: `application/x-www-form-urlencoded`)
- `speaker` (optional): The speaker to use for TTS. Default is `Damien Black`.
- `language` (optional): The language to use for TTS. Default is `en`.

**Response**: Audio file in WAV format as a streaming response.

### 4. Health Check
**Endpoint**: `/health`  
**Method**: GET  
**Description**: Returns the health status of the application.

**Response**:
```json
{
  "status": "ok"
}
```

---

## Deployment Instructions

### Prerequisites
1. **Python 3.8+**: Ensure Python is installed on your system.
2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   pip install ./paddlepaddle-gpu/paddlepaddle_gpu-3.0.0b2-cp312-cp312-win_amd64.whl
   ```
3. **Model Files**:
   - Place the YOLO model at `models/panel_text_detection/panel_text_detection.pt`.
   - Place the PaddleOCR model at `models/rec_french_manga_latin_4`.

### Running the Application

1. Save the application code to a file, e.g., `app.py`.
2. Start the server using Gunicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5000 --workers 1 --reload
   ```

### Accessing the API
- The application will be available at `http://<host>:5000`.
- Test the endpoints using tools like Postman, cURL, or any HTTP client.

### Example Requests

#### Detect Panels and Texts
```bash
curl -X POST "http://127.0.0.1:8000/detect_panels_texts" -F "file=@path_to_image.jpg"
```

#### Text Recognition
```bash
curl -X POST "http://127.0.0.1:8000/text_recognition" -F "file=@path_to_image.jpg"
```

#### Text to Speech
```bash
curl -X POST "http://127.0.0.1:8000/text_to_speech" \
     -d "text=Bonjour, ceci est un test de synthèse vocale." \
     -d "speaker=Damien Black" \
     -d "language=fr"
```

---

## Notes
- Ensure CUDA is available if running on a GPU for better performance.
- Logs and error messages will provide detailed information for debugging if any endpoint fails.

Feel free to modify or extend the endpoints as needed for your application.

