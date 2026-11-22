# Prescription Studio

A mobile-friendly FastAPI website for fictional prescription OCR and medicine database lookup. The page at `/` and Swagger at `/docs` run in one service.

## Run from this workspace

Use your selected Python environment:

```powershell
python -m pip install -r requirements-web.txt
python -m unittest discover -s tests -v
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ and http://127.0.0.1:8000/docs.

`requirements.txt` is the original research dependency list. `requirements-web.txt` contains only the dependencies used by this website. The web matcher uses the original brand database and fuzzy matching without loading the unused LangChain/BM25 retriever.

## Render deployment

Deploy from the user-controlled fork at https://github.com/salahibrahiem12/Prescription-API, branch `codex/prescription-website`. The repository includes a `render.yaml` Blueprint for a dedicated **Free Python Web Service**.

- Build: `pip install -r requirements-web.txt && python -m unittest discover -s tests -v`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health` (returns HTTP 200 only after the index has loaded)
- Python: 3.13.12
- Model: `gemini-3.1-flash-lite`, the stable replacement for the retired `gemini-3.1-flash-lite-preview`
- Set `GEMINI_API_KEY` directly in the Render service's Environment dashboard, then redeploy. Never put the key in frontend code or Git.

The service starts without a Gemini key. Database search remains available; `/health` reports `ocr_configured: false`, and image analysis returns an explicit OCR-pending response. Do not add a required secret placeholder to the Blueprint because that would prevent the keyless demo from deploying.

## Verify the deployed website

1. Open `/health` and confirm `index_ready: true`.
2. Open `/`, search for `Panadol`, and confirm product suggestions appear.
3. Select **Try fictional sample**, then **Analyze prescription**. With a configured key, verify the original Panadol reading, dose, frequency, route, duration, and database suggestions appear. Without a key, verify the explicit live-OCR-pending message.
4. Check `/docs` and the Render build and runtime logs.
5. Check the page at a mobile viewport and ensure controls and results fit.

The source and API checks do not establish live deployment or real provider OCR success. Those must be verified separately after Render access is connected.

## Privacy and limits

Use fictional prescriptions only. The page discloses Gemini processing before any image is selected or submitted. Patient fields are excluded from the OCR prompt; every database suggestion remains separate from the original OCR reading. Uncertain and unmatched entries remain visible.

Uploads are limited to 5 MiB and 20 million pixels, checked as actual supported image formats, and converted to PNG with metadata stripped. Request size is bounded before multipart parsing. Temporary normalized images are deleted on success and failure. OCR JSON results are not written to disk, response caching is disabled, and processing modules do not print prescription details or provider exceptions. Existing historical log/result files are retained locally but excluded from future commits.

The single-worker demo permits five lookup/analysis requests per minute per observed client and ten actual Gemini calls per minute globally (including extra OCR passes and matching fallbacks). Automatic SDK retries are disabled. Limits are in memory and reset on restart; keep one worker/instance or use shared storage before scaling. Free Render services may sleep when idle.

## Regression tests

Tests cover brand cleaning (`Nexium 40 mg` and `Lasix`), decimal strength ranking before truncation on exact/fuzzy/AI paths including `k=1`, pack counts, real Panadol lookup without Gemini, repeated brands at different doses, uncertain lines, upload bounds, missing-key behavior, temporary-file cleanup, safe errors, rate limits, website assets, readiness, and Swagger.

Tests mock OCR provider responses; they never submit real patient data or call Gemini. No Gemini key is needed for the test suite.
