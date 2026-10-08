"""TransKun piano transcription, loaded once per process on CPU.

The TransKun 2.0 checkpoint and its model config ship inside the `transkun`
pip package, so nothing has to be downloaded at runtime.
"""

import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

DEVICE = os.environ.get("DEVICE", "cpu")

# Threads the model may use for CPU inference. The machine's own core count is the
# real ceiling - asking for more than the box has only adds scheduling overhead.
CPU_THREADS = max(1, int(os.environ.get("CPU_THREADS", "4")))
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(CPU_THREADS)
except RuntimeError:
    pass  # interop pool already initialised - intra-op threads are the ones that matter here

# Tempo written into every output MIDI. Notes keep their real (second-based) timing -
# `_set_default_bpm` rescales ticks so the declared BPM changes but playback does not.
DEFAULT_BPM = max(1, int(os.environ.get("DEFAULT_BPM", "120")))

# Only one transcription runs at a time. The model is CPU-bound, so letting a bot's
# concurrent retries run together oversubscribes the cores and makes every request time
# out. A waiter that can't get the model within this window is told to retry shortly.
_transcribe_lock = threading.Lock()
BUSY_WAIT_SECONDS = float(os.environ.get("BUSY_WAIT_SECONDS", "3"))

_model = None
_load_error = None
_ready = threading.Event()
_load_lock = threading.Lock()
_load_started = threading.Event()  # set once a load thread has been kicked off

MODEL_LOAD_TIMEOUT_SECONDS = 600


def _load_blocking():
    """Load the checkpoint exactly once; record any failure instead of crashing."""
    global _model, _load_error
    with _load_lock:
        if _model is not None or _load_error is not None:
            return
        try:
            import moduleconf
            import transkun
            from transkun.ModelTransformer import TransKun

            package_dir = Path(transkun.__file__).parent
            conf_path = package_dir / "pretrained" / "2.0.conf"
            weight_path = package_dir / "pretrained" / "2.0.pt"

            manager = moduleconf.parseFromFile(str(conf_path))
            model = TransKun(conf=manager["Model"].config).to(DEVICE)

            checkpoint = torch.load(str(weight_path), map_location=DEVICE)
            state = (
                checkpoint["best_state_dict"]
                if "best_state_dict" in checkpoint
                else checkpoint["state_dict"]
            )
            model.load_state_dict(state, strict=False)
            model.eval()

            _model = model
            print(f"[transkun] model ready on device '{DEVICE}'", flush=True)
            # Keep the model awake: prime the inference kernels before the first request.
            _warmup_inference(model)
        except Exception as exc:  # keep the HTTP server alive so the error is visible
            _load_error = f"{type(exc).__name__}: {exc}"
            print(f"[transkun] model load failed: {_load_error}", flush=True)
        finally:
            _ready.set()


def _warmup_inference(model):
    """Prime the CPU/MKL kernels with one throwaway forward so the first real request
    has no wake-up cost - the model is kept 'awake'. Non-fatal: a failure here never
    blocks serving, the model is usable either way."""
    try:
        with torch.inference_mode():
            dummy = torch.zeros(int(model.fs * 1.0), 1, device=DEVICE)
            model.transcribe(dummy, discardSecondHalf=False)
        print("[transkun] warmup inference done - model awake", flush=True)
    except Exception as exc:
        print(f"[transkun] warmup inference skipped: {type(exc).__name__}: {exc}", flush=True)


def warm_up():
    """Start loading the model in the background so the server can bind immediately.

    Idempotent: only the first call actually spawns the load thread - `_load_blocking`
    re-checks under the lock, so later calls are no-ops even if they sneak through.
    """
    if _load_started.is_set():
        return
    _load_started.set()
    threading.Thread(target=_load_blocking, name="transkun-load", daemon=True).start()


def wake():
    """Ensure the model is loaded (or loading). Called on every transcribe request so a
    cold process - e.g. one whose startup warm-up never fired - still wakes on demand."""
    if _model is None and _load_error is None:
        warm_up()


def is_ready():
    return _model is not None


def load_error():
    return _load_error


def _require_model():
    _ready.wait(MODEL_LOAD_TIMEOUT_SECONDS)
    if _model is None:
        raise RuntimeError(load_error() or "model is still loading")
    return _model


def _read_mono_audio(audio_path, target_fs):
    """Decode any ffmpeg-readable audio file to mono float32 at the model sample rate."""
    from pydub import AudioSegment

    segment = AudioSegment.from_file(str(audio_path)).set_channels(1)
    source_fs = segment.frame_rate

    samples = np.array(segment.get_array_of_samples()).astype(np.float32) / (2 ** 15)
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    if source_fs != target_fs:
        import soxr

        samples = np.ascontiguousarray(soxr.resample(samples, source_fs, target_fs), dtype=np.float32)

    return samples


def download_audio(url, dest_dir):
    """Download a SoundCloud (or any yt-dlp supported) link as an mp3.

    Returns the path of the downloaded file and the track title.
    """
    import yt_dlp

    options = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(dest_dir, "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)

    downloaded = sorted(Path(dest_dir).glob("source.mp3"))
    if not downloaded:
        downloaded = sorted(p for p in Path(dest_dir).iterdir() if p.suffix != ".part")
    if not downloaded:
        raise RuntimeError(f"yt-dlp produced no audio file for {url}")

    return str(downloaded[0]), (info or {}).get("title") or ""


def _midi_channels(midi_bytes):
    """Read back the MIDI channel(s) that note events were written to."""
    import io

    import mido

    midi = mido.MidiFile(file=io.BytesIO(midi_bytes))
    channels = set()
    for track in midi.tracks:
        for message in track:
            if message.type in ("note_on", "note_off"):
                channels.add(message.channel)
    return sorted(channels)


def _set_default_bpm(midi_bytes, bpm):
    """Declare a fixed default tempo in the MIDI without changing any note's timing.

    transkun's `writeMidi` emits a single constant tempo (pretty_midi's 120 BPM), so we
    can keep playback identical by rescaling every delta-time by `orig_tempo / new_tempo`
    and rewriting the tempo meta. Seconds-per-tick stays the same; only the BPM number
    shown by players/editors changes.
    """
    import io

    import mido

    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    new_tempo = mido.bpm2tempo(bpm)

    orig_tempo = 500000  # 120 BPM, pretty_midi's default
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                orig_tempo = msg.tempo
                break
        else:
            continue
        break
    scale = orig_tempo / new_tempo

    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        out.tracks.append(new_track)
        for msg in track:
            msg = msg.copy()
            if msg.type == "set_tempo":
                msg.tempo = new_tempo
            msg.time = max(0, round(msg.time * scale))
            new_track.append(msg)

    if out.tracks and not any(m.type == "set_tempo" for tr in out.tracks for m in tr):
        out.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=new_tempo, time=0))

    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


def transcribe_file(audio_path):
    """Transcribe one audio file.

    Returns the MIDI bytes, the MIDI channel the notes are written to, and the
    note list (each note carries its channel).
    """
    model = _require_model()

    # Serialize: run one transcription at a time so concurrent bot retries don't
    # oversubscribe the CPU and time out together. Waiters get a fast 503 (Retry-After).
    if not _transcribe_lock.acquire(timeout=BUSY_WAIT_SECONDS):
        raise RuntimeError("model is busy transcribing another request; retry shortly")
    try:
        from transkun.Data import writeMidi

        samples = _read_mono_audio(audio_path, model.fs)
        # transkun expects (frames, channels); mono audio therefore needs a trailing axis.
        x = torch.from_numpy(samples[:, None]).to(DEVICE)

        with torch.inference_mode():
            notes_est = model.transcribe(x, discardSecondHalf=False)

        midi = writeMidi(notes_est)

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            midi.write(tmp_path)
            midi_bytes = Path(tmp_path).read_bytes()
        finally:
            os.remove(tmp_path)

        # Declare the default BPM without altering real note timing (seconds stay the same).
        midi_bytes = _set_default_bpm(midi_bytes, DEFAULT_BPM)

        channels = _midi_channels(midi_bytes)
        channel = channels[0] if channels else 0
        instrument = midi.instruments[0]
        notes = [
            {
                "pitch": int(note.pitch),
                "start": round(float(note.start), 4),
                "end": round(float(note.end), 4),
                "velocity": int(note.velocity),
                "channel": channel,
            }
            for note in instrument.notes
        ]

        return {"midi_bytes": midi_bytes, "channel": channel, "notes": notes}
    finally:
        _transcribe_lock.release()
