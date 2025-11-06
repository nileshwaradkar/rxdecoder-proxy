import os
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rxdecoder")

AZURE_ENDPOINT_BASE = os.environ.get("AZURE_READ_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.environ.get("AZURE_READ_KEY", "")

app = FastAPI(title="RxDecoder OCR Proxy")


# Allow calls from your Flutter web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"ok": True, "service": "rxdecoder-proxy"}


@app.post("/ocr/azure")
async def ocr_azure(file: UploadFile = File(...)):
    """
    Receives an image file from the app and forwards the raw bytes
    to Azure Computer Vision Image Analysis API.
    """
    if not AZURE_ENDPOINT_BASE or not AZURE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Azure OCR is not configured on the server.",
        )

    # Read the uploaded file as raw bytes
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image uploaded")

    logger.info("Received image from client: %s (%d bytes)", file.filename, len(image_bytes))

    # Build Azure URL and query params
    url = f"{AZURE_ENDPOINT_BASE}/computervision/imageanalysis:analyze"
    params = {
        "api-version": "2023-02-01-preview",
        "features": "read",     # use the Read OCR
        "language": "en",
    }

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        # IMPORTANT: send raw bytes, not multipart/form-data
        "Content-Type": "application/octet-stream",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                params=params,
                headers=headers,
                content=image_bytes,  # raw bytes, not "files="
            )
    except httpx.RequestError as e:
        logger.exception("Network error talking to Azure")
        raise HTTPException(status_code=502, detail=f"Azure request error: {e}") from e

    logger.info("Azure OCR status=%s", resp.status_code)

    # Log a small part of the response for debugging
    snippet = resp.text[:500]
    logger.info("Azure OCR response snippet: %s", snippet)

    # If Azure returned an error, bubble it up so the app can show something useful
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    # On success, just relay Azure's JSON back to the Flutter app
    try:
        data = resp.json()
    except ValueError:
        # Fallback if response is not JSON
        data = {"raw": resp.text}

    return data
