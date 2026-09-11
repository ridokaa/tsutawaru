"""Windows WASAPI loopback capture backend using PyAudioWPatch."""
from __future__ import annotations

from typing import Any, Callable, Tuple


def open_loopback(blocksize: int, callback: Callable) -> Tuple[Any, Any, dict]:
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    # Resolves the loopback twin of the current default output device.
    loop = pa.get_default_wasapi_loopback()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=loop["maxInputChannels"],
        rate=int(loop["defaultSampleRate"]),
        frames_per_buffer=blocksize,
        input=True,
        input_device_index=loop["index"],
        stream_callback=callback,
    )
    return pa, stream, loop
