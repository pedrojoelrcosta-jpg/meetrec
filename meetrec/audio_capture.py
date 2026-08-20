"""Dual-track capture: microphone + WASAPI loopback of the default output.

Uses PyAudioWPatch (native WASAPI loopback, no virtual audio cables). Each
track is written incrementally to WAV in chunks — nothing accumulates in
memory.

Mid-meeting device changes: if the stream dies, the current default device
is reopened, the lost interval is filled with silence (to preserve the
timeline) and naive resampling is applied if the new rate differs from the
initial one.
"""

import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

log = logging.getLogger(__name__)

CHUNK_FRAMES = 4096
REOPEN_RETRY_S = 1.0


def _default_mic(p: pyaudio.PyAudio) -> dict:
    return p.get_default_input_device_info()


def _default_loopback(p: pyaudio.PyAudio) -> dict:
    return p.get_default_wasapi_loopback()


class _TrackRecorder(threading.Thread):
    """Records one device to a WAV file, tolerating device changes."""

    def __init__(self, name: str, wav_path: Path, get_device, stop_event):
        super().__init__(name=f"rec-{name}", daemon=True)
        self._wav_path = wav_path
        self._get_device = get_device
        self._stop = stop_event
        self._pa: pyaudio.PyAudio | None = None
        self.error: Exception | None = None
        self.rate = 0
        self.channels = 0
        self.frames_written = 0

    def _open_stream(self):
        device = self._get_device(self._pa)
        rate = int(device["defaultSampleRate"])
        channels = min(2, max(1, int(device["maxInputChannels"])))
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=CHUNK_FRAMES,
        )
        return stream, rate, channels

    def run(self) -> None:
        self._pa = pyaudio.PyAudio()
        wav = None
        stream = None
        try:
            stream, self.rate, self.channels = self._open_stream()
            wav = wave.open(str(self._wav_path), "wb")
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.rate)
            src_rate, src_channels = self.rate, self.channels
            last_ok = time.monotonic()

            while not self._stop.is_set():
                try:
                    data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                except OSError:
                    # device changed/disappeared: reopen the current default
                    try:
                        stream.close()
                    except Exception:
                        pass
                    stream = None
                    while stream is None and not self._stop.is_set():
                        try:
                            stream, src_rate, src_channels = self._open_stream()
                        except OSError:
                            time.sleep(REOPEN_RETRY_S)
                    if stream is None:
                        break
                    gap_s = time.monotonic() - last_ok
                    silence = b"\x00" * int(gap_s * self.rate) * 2 * self.channels
                    wav.writeframes(silence)
                    self.frames_written += len(silence) // (2 * self.channels)
                    log.warning("%s: device reopened, %.1fs of silence inserted",
                                self.name, gap_s)
                    continue

                last_ok = time.monotonic()
                samples = np.frombuffer(data, dtype=np.int16)
                if src_channels != self.channels:
                    samples = samples.reshape(-1, src_channels)
                    if self.channels == 1:
                        samples = samples.mean(axis=1).astype(np.int16)
                    else:  # mono -> stereo
                        samples = np.repeat(samples, 2, axis=1)[:, :2]
                    samples = samples.reshape(-1)
                if src_rate != self.rate:
                    frames = samples.reshape(-1, self.channels)
                    n_out = int(len(frames) * self.rate / src_rate)
                    idx = np.linspace(0, len(frames) - 1, n_out)
                    frames = frames[idx.round().astype(int)]
                    samples = frames.reshape(-1)
                wav.writeframes(samples.tobytes())
                self.frames_written += len(samples) // self.channels
        except Exception as exc:  # noqa: BLE001 — reported to the caller
            self.error = exc
            log.exception("%s: fatal capture error", self.name)
        finally:
            for closer in (
                lambda: stream and stream.close(),
                lambda: wav and wav.close(),
                lambda: self._pa and self._pa.terminate(),
            ):
                try:
                    closer()
                except Exception:
                    pass

    @property
    def seconds_written(self) -> float:
        return self.frames_written / self.rate if self.rate else 0.0


class DualTrackRecorder:
    """track_mic.wav (microphone) + track_sys.wav (default output loopback)."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        self.started_at: float | None = None
        self._stop = threading.Event()
        self.mic = _TrackRecorder("mic", session_dir / "track_mic.wav",
                                  _default_mic, self._stop)
        self.sys = _TrackRecorder("sys", session_dir / "track_sys.wav",
                                  _default_loopback, self._stop)

    def start(self) -> None:
        self.started_at = time.time()
        self.mic.start()
        self.sys.start()

    def stop(self) -> dict:
        self._stop.set()
        self.mic.join(timeout=10)
        self.sys.join(timeout=10)
        return {
            "duration_s": round(max(self.mic.seconds_written,
                                    self.sys.seconds_written), 1),
            "mic": {"rate": self.mic.rate, "channels": self.mic.channels,
                    "seconds": round(self.mic.seconds_written, 1),
                    "error": str(self.mic.error) if self.mic.error else None},
            "sys": {"rate": self.sys.rate, "channels": self.sys.channels,
                    "seconds": round(self.sys.seconds_written, 1),
                    "error": str(self.sys.error) if self.sys.error else None},
        }
