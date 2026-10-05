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

_model = None
_load_error = None
_ready = threading.Event()
_load_lock = threading.Lock()

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
        except Exception as exc:  # keep the HTTP server alive so the error is visible
            _load_error = f"{type(exc).__name__}: {exc}"
            print(f"[transkun] model load failed: {_load_error}", flush=True)
        finally:
            _ready.set()


def warm_up():
    """Start loading the model in the background so the server can bind immediately."""
    threading.Thread(target=_load_blocking, name="transkun-load", daemon=True).start()


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


def transcribe_file(audio_path):
    """Transcribe one audio file.

    Returns the MIDI bytes, the MIDI channel the notes are written to, and the
    note list (each note carries its channel).
    """
    model = _require_model()

    from transkun.Data import writeMidi

    samples = _read_mono_audio(audio_path, model.fs)
    # transkun expects (frames, channels); mono audio therefore needs a trailing axis.
    x = torch.from_numpy(samples[:, None]).to(DEVICE)

    with torch.no_grad():
        notes_est = model.transcribe(x, discardSecondHalf=False)

    midi = writeMidi(notes_est)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        midi.write(tmp_path)
        midi_bytes = Path(tmp_path).read_bytes()
    finally:
        os.remove(tmp_path)

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
