"""HTTP API around the TransKun piano-transcription model.

A bot (or any client) posts a performance recording and receives a MIDI file
back, along with the MIDI channel the notes were written to.
"""

import base64
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import transcriber

PUBLIC_SUFFIX = os.environ.get("BASE44_PUBLIC_HOST_SUFFIX", "")
PUBLIC_BASE = f"https://3000-{PUBLIC_SUFFIX}" if PUBLIC_SUFFIX else "http://localhost:3000"

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
    <tr><td><code>POST /transcribe</code></td>
        <td>Multipart upload of an audio file (field <code>file</code>) &rarr; MIDI file.
            The MIDI channel is returned in the <code>X-Midi-Channel</code> header and the
            note count in <code>X-Note-Count</code>.</td></tr>
    <tr><td><code>POST /transcribe?format=json</code></td>
        <td>JSON response with <code>midi_base64</code>, <code>channel</code> and
            <code>note_count</code>. Add <code>&amp;include_notes=true</code> for every note,
            each tagged with its channel.</td></tr>
    <tr><td><code>GET /health</code></td><td>Service and model status.</td></tr>
    <tr><td><code>GET /docs</code></td><td>Interactive API documentation.</td></tr>
  </table>

  <h2>Model link for your bot</h2>
  <pre>__PUBLIC_BASE__/transcribe</pre>
  <p class="hint">Example: <code>curl -F "file=@song.mp3" __PUBLIC_BASE__/transcribe -o out.mid</code></p>
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


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    format: str = Query("midi", pattern="^(midi|json)$"),
    include_notes: bool = Query(False),
):
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = transcriber.transcribe_file(tmp_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not transcribe audio: {exc}")
    finally:
        os.remove(tmp_path)

    stem = Path(file.filename or "transcription").stem or "transcription"

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
    return INDEX_TEMPLATE.replace("__PUBLIC_BASE__", PUBLIC_BASE)
