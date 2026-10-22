"""Device lookup on Windows / macOS / Linux by name + host API.

Indices change when a USB device is replugged, so everything is resolved by
substring. On Windows the ReSpeaker only delivers its 6 channels once the
"audio enhancements" (Voice Clarity effect pack) are disabled for it: with
them on, every host API returns 2 identical channels at 48 kHz (doc/01).
"""

from __future__ import annotations

import sounddevice as sd


def host_name(dev: dict) -> str:
    return sd.query_hostapis(dev["hostapi"])["name"]


def find(name: str, kind: str, channels: int = 1, hostapis=("Windows WASAPI", "MME", "Core Audio", "ALSA")) -> int:
    """First device whose name contains ``name`` with >= ``channels`` of ``kind``
    ('input' | 'output'), preferring host APIs in the given order."""
    key = f"max_{kind}_channels"
    devs = [(i, d) for i, d in enumerate(sd.query_devices())
            if name.lower() in d["name"].lower() and d[key] >= channels]
    for api in hostapis:
        for i, d in devs:
            if host_name(d) == api:
                return i
    if devs:
        return devs[0][0]
    raise SystemExit(f"no {kind} device matching {name!r} with >= {channels} ch; "
                     f"see python -m sounddevice")


def describe(i: int) -> str:
    d = sd.query_devices(i)
    return f"#{i} {d['name']} [{host_name(d)}] in={d['max_input_channels']} out={d['max_output_channels']} fs={d['default_samplerate']:.0f}"
