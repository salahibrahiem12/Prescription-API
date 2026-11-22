"""Single-service prescription demo: website, OCR and database lookup."""
import io
import logging
import os
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated

from dotenv import load_dotenv
load_dotenv()
# Access URLs can contain private query text; exclude them from runtime logs.
logging.getLogger('uvicorn.access').disabled = True
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

import drug_pipeline
from pipeline import run

BASE = Path(__file__).resolve().parent
XLSX_PATH = os.getenv('XLSX_PATH', str(BASE / 'Book1.xlsx'))
MAX_UPLOAD = 5 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
_client_calls = {}
_rate_lock = Lock()


@asynccontextmanager
async def lifespan(app):
    drug_pipeline.build_index(XLSX_PATH)
    drug_pipeline._load()  # Readiness means the index was actually loaded.
    app.state.index_ready = True
    yield


app = FastAPI(title='Prescription API', version='1.1.0', lifespan=lifespan,
              description='Research demo. Verify every result with a licensed pharmacist.')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')


class UploadLimitMiddleware:
    """Bound the request before multipart parsing can spool an oversized upload."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/analyze':
            return await self.app(scope, receive, send)
        size = 0
        async def limited_receive():
            nonlocal size
            message = await receive()
            size += len(message.get('body', b''))
            if size > MAX_UPLOAD + 64 * 1024:
                raise HTTPException(413, 'Image must be 5 MB or smaller.')
            return message
        headers = dict(scope.get('headers', []))
        try:
            oversized = int(headers.get(b'content-length', b'0')) > MAX_UPLOAD + 64 * 1024
        except ValueError:
            oversized = True
        if oversized:
            return await JSONResponse({'detail': 'Image must be 5 MB or smaller.'}, 413)(scope, receive, send)
        await self.app(scope, limited_receive, send)


app.add_middleware(UploadLimitMiddleware)


@app.middleware('http')
async def privacy_headers(request, call_next):
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    if request.url.path == '/':
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    return response


def throttle(request):
    now = time.monotonic()
    # Do not trust user-supplied forwarding headers. Single Render worker plus
    # the global provider-call budget bounds total usage even behind a proxy.
    key = request.client.host if request.client else 'unknown'
    with _rate_lock:
        for old_key in list(_client_calls):
            if not _client_calls[old_key] or now - _client_calls[old_key][-1] >= 60:
                del _client_calls[old_key]
        calls = _client_calls.setdefault(key, deque())
        while calls and now - calls[0] >= 60:
            calls.popleft()
        if len(calls) >= 5:
            raise HTTPException(429, 'Demo request limit reached. Retry in one minute.', headers={'Retry-After': '60'})
        calls.append(now)


@app.get('/', include_in_schema=False)
def website():
    return FileResponse(BASE / 'static' / 'index.html')


@app.get('/health')
def health():
    ready = getattr(app.state, 'index_ready', False)
    return JSONResponse({'status': 'ok' if ready else 'starting', 'index_ready': ready,
                         'ocr_configured': bool(os.getenv('GEMINI_API_KEY'))}, 200 if ready else 503)


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=150)]
class MatchRequest(BaseModel):
    names: list[Name] = Field(min_length=1, max_length=10)
    k: int = Field(default=5, ge=1, le=20)
    meta: dict[str, dict] | None = None


@app.post('/match')
def match_drugs(body: MatchRequest, request: Request):
    throttle(request)
    try:
        return {'results': drug_pipeline.match_many(body.names, k=body.k, meta=body.meta)}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, 'Database lookup is temporarily unavailable.') from None


@app.post('/analyze')
async def analyze(request: Request, file: UploadFile = File(...), k: int = Query(5, ge=1, le=20)):
    throttle(request)
    tmp_path = None
    try:
        if file.content_type not in {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}:
            raise HTTPException(415, 'Choose a JPEG, PNG, WebP or GIF image.')
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, 'Image must be 5 MB or smaller.')
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {'JPEG', 'PNG', 'WEBP', 'GIF'} or image.width * image.height > 20_000_000:
                    raise ValueError()
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(422, 'This file is not a valid image, or its dimensions are too large.') from None
        if not os.getenv('GEMINI_API_KEY'):
            raise HTTPException(503, 'Live OCR is pending: the server owner must configure GEMINI_API_KEY. Text lookup is available.')
        # Normalize format and strip EXIF metadata before sending the image to Gemini.
        with Image.open(io.BytesIO(data)) as image:
            normalized = image.convert('RGB')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp_path = Path(tmp.name)
                normalized.save(tmp, format='PNG')
        return await run_in_threadpool(run, str(tmp_path), XLSX_PATH, k, True)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(502, 'Image analysis failed. Please retry or use text lookup.') from None
    finally:
        await file.close()
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
