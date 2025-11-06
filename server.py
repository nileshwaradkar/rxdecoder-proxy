from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import requests
import logging

app = FastAPI()

# already there in your file, but make sure it's this:
AZURE_ENDPOINT = os.environ.get("AZURE_READ_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.environ.get("AZURE_READ_KEY", "")

logger = logging.getLogger("uvicorn.error")


@app.post("/ocr/azure")
async def ocr_azure(file: UploadFile = File(...)):
    # 1) Basic config checks
    if not AZURE_ENDPOINT or not AZURE_KEY:
        logger.error(f"Azure configuration missing. "
                     f"AZURE_ENDPOINT='{AZURE_ENDPOINT}', AZURE_KEY set={bool(AZURE_KEY)}")
        raise HTTPException(status_code=500, detail="Azure OCR is not configured on the server")

    # 2) Read image bytes from uploaded file
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file received")

    # 3) Build Azure URL
    url = f"{AZURE_ENDPOINT}?language=unk&detectOrientation=true"

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream",
    }

    try:
        # 4) Call Azure OCR
        resp = requests.post(url, headers=headers, data=data, timeout=30)
        logger.info(f"Azure OCR status={resp.status_code}, body={resp.text[:400]}")
    except Exception as e:
        logger.exception("Azure OCR HTTP call failed")
        # this 502 is what you were seeing before
        raise HTTPException(status_code=502, detail="Azure OCR call failed")

    # 5) If Azure returned an error (e.g. 401, 400), forward it
    if resp.status_code >= 400:
        logger.error(f"Azure OCR error {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    # 6) Success: return Azure JSON directly to the app
    try:
        return resp.json()
    except Exception:
        logger.error(f"Azure OCR returned non-JSON body: {resp.text[:400]}")
        raise HTTPException(status_code=500, detail="Azure OCR returned invalid JSON")
