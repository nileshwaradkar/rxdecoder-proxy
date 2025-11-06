from fastapi import FastAPI, UploadFile, HTTPException
import httpx
import os

AZURE_ENDPOINT = os.environ.get("AZURE_READ_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.environ.get("AZURE_READ_KEY", "")
API_VER = "2024-02-29-preview"  # Azure Image Analysis (Read)

app = FastAPI(title="RxDecoder OCR Proxy")


@app.get("/")
def root():
    return {"ok": True, "service": "rxdecoder-proxy"}


@app.post("/ocr/azure")
async def ocr_azure(file: UploadFile):
    if not AZURE_ENDPOINT or not AZURE_KEY:
        raise HTTPException(status_code=500, detail="Azure config not set")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        url = f"{AZURE_ENDPOINT}/computervision/imageanalysis:analyze"
        params = {"api-version": API_VER, "features": "read"}
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_KEY,
            "Content-Type": "application/octet-stream",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url, params=params, headers=headers, content=img_bytes
            )
            resp.raise_for_status()
            data = resp.json()

        lines: list[str] = []
        read_result = data.get("readResult") or {}
        for block in read_result.get("blocks", []):
            for ln in block.get("lines", []):
                text = (ln.get("text") or "").strip()
                if text:
                    lines.append(text)

        return {"text": "\n".join(lines), "lineCount": len(lines)}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Azure HTTP error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
