"""
Microphone device enumeration and input-stream factory.

Usage:
    from voice_os.core.mic_manager import MicManager
    mgr = MicManager()
    devices = mgr.list_input_devices()
    stream = mgr.open_input_stream(callback=my_cb)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

from voice_os.config.settings import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000   # Hz — required by Whisper and Silero VAD
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 512        # frames per callback (~32 ms at 16 kHz)


@dataclass
class MicDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


class MicManager:
    """Wraps sounddevice to enumerate and open microphone input streams."""

    def __init__(self) -> None:
        self._device_id: Optional[int] = settings.mic_device_id

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------

    def list_input_devices(self) -> List[MicDevice]:
        """Return all devices that have at least one input channel."""
        devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append(
                    MicDevice(
                        index=idx,
                        name=dev["name"],
                        max_input_channels=dev["max_input_channels"],
                        default_samplerate=dev["default_samplerate"],
                    )
                )
        return devices

    def get_default_input_device(self) -> Optional[MicDevice]:
        """Return the system-default input device info, or None."""
        try:
            idx = sd.default.device[0]
            if idx < 0:
                return None
            dev = sd.query_devices(idx)
            return MicDevice(
                index=idx,
                name=dev["name"],
                max_input_channels=dev["max_input_channels"],
                default_samplerate=dev["default_samplerate"],
            )
        except Exception as exc:
            logger.warning("Could not query default input device: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Stream factory
    # ------------------------------------------------------------------

    def open_input_stream(
        self,
        callback: Callable[[np.ndarray, int, object, object], None],
        *,
        sample_rate: int = SAMPLE_RATE,
        blocksize: int = BLOCKSIZE,
    ) -> sd.InputStream:
        """
        Open a non-blocking sounddevice InputStream.

        ``callback`` receives (indata: np.ndarray[int16], frames, time, status).
        The stream is *not* started here — caller must call stream.start().
        """
        device = self._device_id  # None → use system default
        logger.info(
            "Opening mic stream: device=%s, rate=%d, blocksize=%d",
            device if device is not None else "default",
            sample_rate,
            blocksize,
        )

        def _wrapped_callback(indata, frames, time, status):
            if status:
                logger.debug("Mic stream status: %s", status)
            callback(indata.copy(), frames, time, status)

        stream = sd.InputStream(
            device=device,
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=blocksize,
            callback=_wrapped_callback,
        )
        return stream

    def set_device(self, device_id: Optional[int]) -> None:
        """Hot-swap the mic device; takes effect on the next open_input_stream call."""
        self._device_id = device_id
        settings.update(mic_device_id=device_id)
        logger.info("Mic device set to: %s", device_id)
