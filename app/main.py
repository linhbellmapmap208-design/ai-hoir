"""HTTP API around the TransKun piano-transcription model.

A bot (or any client) posts a performance recording and receives a MIDI file
back, along with the MIDI channel the notes were written to.
"""

import base64
import os
import secrets
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from . import transcriber

PUBLIC_SUFFIX = os.environ.get("BASE44_PUBLIC_HOST_SUFFIX", "")
PUBLIC_BASE = f"https://3000-{PUBLIC_SUFFIX}" if PUBLIC_SUFFIX else "http://localhost:3000"

# When set, every transcribe/transcriber route requires this key. Clients may send it
# as `X-API-Key`, `Authorization: Bearer <key>`, or `?api_key=`. Empty = open (dev).
API_KEY = os.environ.get("API_KEY", "").strip()
PROTECTED_PREFIXES = ("/transcribe", "/transcriber")

INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TransKun Piano Transcription API</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 40px 20px; background: #0f1115; color: #e6e8ee;
         font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 24px; margin: 0 0 6px; letter-spacing: -0.01em; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em;
       color: #8b93a7; margin: 32px 0 10px; }
  p.sub { margin: 0; color: #9aa3b8; }
  .status { display: inline-block; margin-top: 18px; padding: 6px 12px; border-radius: 999px;
            font-size: 13px; background: #1b1f2a; border: 1px solid #2a3040; color: #9aa3b8; }
  .status.ready { background: #10291d; border-color: #1f5c3d; color: #6ee7a8; }
  .status.error { background: #2c1418; border-color: #5c2129; color: #f8a0ab; }
  table { width: 100%; border-collapse: collapse; }
  td { padding: 10px 12px; border-top: 1px solid #212736; vertical-align: top; }
  td:first-child { width: 34%; white-space: nowrap; }
  tr:first-child td { border-top: none; }
  code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
  code { background: #1b1f2a; padding: 2px 6px; border-radius: 5px; color: #cdd3e1; }
  pre { background: #1b1f2a; border: 1px solid #2a3040; border-radius: 8px;
        padding: 14px; overflow-x: auto; margin: 0; color: #cdd3e1; }
  .hint { color: #8b93a7; font-size: 13px; margin-top: 12px; }
</style>
</head>
<body>
<main>
  <h1>TransKun Piano Transcription API</h1>
  <p class="sub">Piano audio in, MIDI out. Model <strong>transkun 2.0</strong> running on CPU.</p>
  <div class="status" id="status">checking&hellip;</div>

  <h2>Endpoints</h2>
  <table>
    <tr><td><code>POST /transcribe</code><br><code>POST /transcriber</code></td>
        <td>Send <strong>either</strong> an audio/video file (multipart field <code>file</code> &mdash;
            <code>mp3</code>, <code>wav</code>, <code>m4a</code>, <code>mp4</code>, <code>mov</code>, &hellip;;
            the audio track is extracted with ffmpeg)
            <strong>or</strong> a link (field <code>url</code> / <code>link</code> &mdash; e.g. a
            SoundCloud track, fetched with yt-dlp) &rarr; MIDI file. The MIDI channel comes back
            in the <code>X-Midi-Channel</code> header, the note count in
            <code>X-Note-Count</code>.</td></tr>
    <tr><td><code>...?format=json</code></td>
        <td>JSON response with <code>midi_base64</code>, <code>channel</code> and
            <code>note_count</code>. Add <code>&amp;include_notes=true</code> for every note,
            each tagged with its channel.</td></tr>
    <tr><td><code>GET /health</code></td><td>Service and model status.</td></tr>
    <tr><td><code>GET /docs</code></td><td>Interactive API documentation.</td></tr>
  </table>

  <h2>API key</h2>
  <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
    <input id="apiKey" readonly value="__API_KEY_VALUE__"
      style="flex:1 1 240px; min-width:0; background:#1b1f2a; border:1px solid #2a3040; border-radius:8px;
             padding:10px 12px; color:#e6e8ee; font:13px ui-monospace,SFMono-Regular,Menlo,monospace;">
    <button id="copyKey" type="button"
      style="padding:10px 14px; border:0; border-radius:8px; background:#3b82f6; color:#fff;
             font:13px ui-sans-serif,system-ui,sans-serif; cursor:pointer;">Copy</button>
  </div>
  <p class="hint">Gửi key này ở header <code>X-API-Key</code> (hoặc <code>Authorization: Bearer</code> / <code>?api_key=</code>) với mỗi request tới <code>/transcribe</code> hoặc <code>/transcriber</code>.</p>

  <h2>Model link for your bot</h2>
  <pre>__PUBLIC_BASE__/transcribe</pre>
  <p class="hint">__API_KEY_HINT__File: <code>curl -H "X-API-Key: YOUR_KEY" -F "file=@song.mp3" __PUBLIC_BASE__/transcriber -o out.mid</code><br>
     Link: <code>curl -H "X-API-Key: YOUR_KEY" -F "link=https://soundcloud.com/..." __PUBLIC_BASE__/transcriber -o out.mid</code></p>
</main>
<script>
  async function poll() {
    try {
      const res = await fetch('/health');
      const data = await res.json();
      const el = document.getElementById('status');
      if (data.status === 'ready') {
        el.textContent = 'model ready \\u00b7 device ' + data.device;
        el.className = 'status ready';
      } else if (data.status === 'loading') {
        el.textContent = 'loading model\\u2026';
        el.className = 'status';
        setTimeout(poll, 2000);
      } else {
        el.textContent = 'error: ' + data.error;
        el.className = 'status error';
      }
    } catch (e) {
      setTimeout(poll, 2000);
    }
  }
  poll();

  document.getElementById('copyKey')?.addEventListener('click', async () => {
    const el = document.getElementById('apiKey');
    try { await navigator.clipboard.writeText(el.value); }
    catch (e) { el.select(); document.execCommand('copy'); }
    const b = document.getElementById('copyKey'); const t = b.textContent;
    b.textContent = 'Copied!'; setTimeout(() => { b.textContent = t; }, 1500);
  });
</script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(_app):
    transcriber.warm_up()
    yield


app = FastAPI(title="TransKun Piano Transcription API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Midi-Channel", "X-Note-Count", "Content-Disposition"],
)


@app.middleware("http")
async def _api_key_guard(request: Request, call_next):
    # Let CORS handle preflight without a key check.
    if request.method != "OPTIONS" and API_KEY:
        path = request.url.path
        if path.startswith(PROTECTED_PREFIXES):
            provided = (
                request.headers.get("x-api-key", "").strip()
                or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
                or request.query_params.get("api_key", "").strip()
            )
            if not provided or not secrets.compare_digest(provided, API_KEY):
                return JSONResponse(
                    status_code=401, content={"detail": "invalid or missing API key"}
                )
    return await call_next(request)


@app.get("/health")
def health():
    error = transcriber.load_error()
    if error:
        status = "error"
    elif transcriber.is_ready():
        status = "ready"
    else:
        status = "loading"

    return {
        "status": status,
        "model": "transkun-2.0",
        "device": transcriber.DEVICE,
        "model_loaded": transcriber.is_ready(),
        "error": error,
    }


def _safe_stem(name):
    cleaned = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in (name or "")).strip()
    return cleaned[:80].strip() or "transcription"


async def _request_input(request):
    """Pull an uploaded file and/or a link out of a multipart, form or JSON body."""
    upload = None
    fields = {}

    if request.headers.get("content-type", "").startswith("application/json"):
        fields = await request.json()
    else:
        form = await request.form()
        candidate = form.get("file")
        if isinstance(candidate, UploadFile):
            upload = candidate
        fields = form

    url = None
    for key in ("url", "link", "soundcloud"):
        value = fields.get(key) or request.query_params.get(key)
        if value:
            url = str(value).strip()
            break

    return upload, url


def _process(upload_path, upload_stem, url, work_dir):
    """Blocking part: resolve the input to a local audio file, then transcribe it."""
    if upload_path is not None:
        audio_path, stem = upload_path, upload_stem
    else:
        audio_path, title = transcriber.download_audio(url, work_dir)
        stem = _safe_stem(title)
    # A content/link key lets a bot's retries of the same audio reuse a finished result
    # instead of re-running the (slow, CPU-bound) model every time.
    key = transcriber.cache_key(audio_path, link=url)
    return transcriber.transcribe_file(audio_path, key=key), stem


@app.post("/transcribe")
@app.get("/transcribe")
@app.post("/transcribe/predict")
@app.get("/transcribe/predict")
@app.post("/transcriber")
@app.get("/transcriber")
@app.post("/transcriber/predict")
@app.get("/transcriber/predict")
async def transcribe(
    request: Request,
    format: str = Query("midi", pattern="^(midi|json)$"),
    include_notes: bool = Query(False),
):
    # Wake the model on demand so a request to a cold process still triggers the load.
    transcriber.wake()

    upload, url = await _request_input(request)
    if upload is None and not url:
        raise HTTPException(
            status_code=422,
            detail="send an audio file (field 'file') or a link (field 'url' or 'link')",
        )

    work_dir = tempfile.mkdtemp(prefix="transkun-")
    try:
        upload_path = upload_stem = None
        if upload is not None:
            suffix = Path(upload.filename or "audio").suffix or ".mp3"
            upload_path = os.path.join(work_dir, f"input{suffix}")
            with open(upload_path, "wb") as handle:
                handle.write(await upload.read())
            upload_stem = _safe_stem(Path(upload.filename or "").stem)

        try:
            # Downloading and transcribing are blocking and CPU heavy - keep them
            # off the event loop so /health and other requests stay responsive.
            result, stem = await run_in_threadpool(
                _process, upload_path, upload_stem, url, work_dir
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503, detail=str(exc), headers={"Retry-After": "5"}
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not transcribe audio: {exc}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if format == "json":
        payload = {
            "filename": f"{stem}.mid",
            "channel": result["channel"],
            "note_count": len(result["notes"]),
            "midi_base64": base64.b64encode(result["midi_bytes"]).decode("ascii"),
        }
        if include_notes:
            payload["notes"] = result["notes"]
        return JSONResponse(payload)

    headers = {
        "X-Midi-Channel": str(result["channel"]),
        "X-Note-Count": str(len(result["notes"])),
        "Content-Disposition": f'attachment; filename="{stem}.mid"',
    }
    return Response(content=result["midi_bytes"], media_type="audio/midi", headers=headers)


@app.get("/", response_class=HTMLResponse)
def index():
    hint = "<strong>API key required.</strong> Send it via the <code>X-API-Key</code> header, <code>Authorization: Bearer &lt;key&gt;</code>, or <code>?api_key=</code>.<br>" if API_KEY else ""
    return (
        INDEX_TEMPLATE.replace("__PUBLIC_BASE__", PUBLIC_BASE)
        .replace("__API_KEY_HINT__", hint)
        .replace("__API_KEY_VALUE__", API_KEY or "")
    )
