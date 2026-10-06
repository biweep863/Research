"""Binaural room impulse responses (BRIRs): image sources + HRIRs + stochastic tail.

Each image source from pyroomacoustics' shoebox ISM (up to ``ism_order``)
is rendered through the HRIR of its arrival direction in the listener's head
frame. Beyond the ISM horizon a binaural diffuse tail (independent noise
through many random-direction HRIRs, so the interaural coherence is that of a
diffuse field) decays at the room's RT60 and is level-matched to the ISM.

Returned per source: ``full`` (what the ears get) and ``early`` (direct path +
reflections within ``early_ms`` of it) -- the training target.
"""

from __future__ import annotations

import numpy as np
import scipy.signal

from .hrtf import HrtfSet

C = 343.0


def _render_images(delays: np.ndarray, amps: np.ndarray, idx: np.ndarray, hrtf: HrtfSet, n: int):
    """Sum of amps * HRIR[idx] delayed by integer ``delays`` -> (2, n)."""
    L = hrtf.hrir.shape[-1]
    out = np.zeros((2, n + L - 1))
    keep = delays < n
    delays, amps, idx = delays[keep], amps[keep], idx[keep]
    for u in np.unique(idx):
        sel = idx == u
        train = np.zeros(n)
        np.add.at(train, delays[sel], amps[sel])
        out += scipy.signal.fftconvolve(train[None], hrtf.hrir[u], axes=-1)
    return out[:, :n]


def diffuse_tail(n: int, hrtf: HrtfSet, rng, n_dirs: int = 32) -> np.ndarray:
    """(2, n) unit-power binaural diffuse noise."""
    pick = rng.choice(len(hrtf.dirs), size=n_dirs, replace=False)
    out = np.zeros((2, n))
    for p in pick:
        out += scipy.signal.fftconvolve(rng.standard_normal(n)[None], hrtf.hrir[p], axes=-1)[:, :n]
    return out / np.sqrt(np.mean(out**2))


def make_brir(
    room_dim,
    rt60: float,
    listener: np.ndarray,
    source: np.ndarray,
    hrtf: HrtfSet,
    fs: int,
    rng,
    ism_order: int = 6,
    early_ms: float = 50.0,
    max_len_s: float = 0.8,
    pre_delay_ms: float = 1.0,
):
    import pyroomacoustics as pra

    room_dim = np.asarray(room_dim, float)
    e_abs, _ = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_abs), max_order=ism_order, air_absorption=False)
    room.add_source(np.asarray(source, float))
    room.add_microphone(np.asarray(listener, float))
    room.image_source_model()
    src = room.sources[0]
    vec = src.images.T - listener[None]  # head faces +x, no rotation
    dist = np.linalg.norm(vec, axis=1)
    amps = src.damping[0] / (4 * np.pi * dist)
    t0 = dist.min() / C
    shift = t0 - pre_delay_ms / 1000
    delays = np.round((dist / C - shift) * fs).astype(int)
    n = int(max_len_s * fs)
    idx = hrtf.nearest(vec)
    # ISM is complete only until the first missing (order+1) image could arrive
    t_horizon = (ism_order + 1) * room_dim.min() / C - shift
    n_h = min(n, max(int(t_horizon * fs), int((early_ms / 1000 + 0.01) * fs)))
    ism = delays < n_h
    full = _render_images(delays[ism], amps[ism], idx[ism], hrtf, n)
    early_cut = int((pre_delay_ms + early_ms) / 1000 * fs)
    sel = delays < early_cut
    early = _render_images(delays[sel], amps[sel], idx[sel], hrtf, n)

    if n_h < n:  # stochastic late tail, level-matched to the last 10 ms of the ISM
        win = int(0.01 * fs)
        e_ism = np.mean(full[:, n_h - win : n_h] ** 2)
        decay = np.exp(-6.9 * np.arange(n) / fs / rt60)  # amplitude: -60 dB at RT60
        tail = diffuse_tail(n, hrtf, rng) * decay / decay[n_h - win // 2] * np.sqrt(e_ism)
        ramp = np.clip((np.arange(n) - (n_h - win)) / win, 0, 1)
        full = full * (1 - ramp) + tail * ramp
    return full.astype(np.float32), early.astype(np.float32)


def sample_room(cfg_rooms: dict, rng):
    d = cfg_rooms
    dim = np.array([rng.uniform(*d["length_m"]), rng.uniform(*d["width_m"]), rng.uniform(*d["height_m"])])
    rt60 = rng.uniform(*d["rt60_s"])
    m = d["wall_margin_m"]
    listener = np.array([rng.uniform(m, dim[0] - m), rng.uniform(m, dim[1] - m), rng.uniform(*d["listener_height_m"])])
    return dim, rt60, listener


def place_source(dim, listener, rng, dist_range, el_range_deg, margin: float = 0.3, tries: int = 200):
    """Random azimuth (full circle), elevation and distance that fits in the room."""
    for _ in range(tries):
        az = rng.uniform(-np.pi, np.pi)
        el = np.deg2rad(rng.uniform(*el_range_deg))
        r = rng.uniform(*dist_range)
        p = listener + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        if np.all(p > margin) and np.all(p < dim - margin):
            return p, float(np.degrees(az)), float(np.degrees(el)), float(r)
    raise RuntimeError("could not place a source in the room")


def make_room(i: int, cfg_rooms: dict, hrtf: HrtfSet, fs: int, seed: int):
    """Room ``i`` of a bank: its own RNG stream, so banks can be built in parallel."""
    rng = np.random.default_rng([seed, i])
    d = cfg_rooms
    dim, rt60, lis = sample_room(d, rng)
    p, az, el, r = place_source(dim, lis, rng, d["target_dist_m"], d["target_el_deg"])
    kw = dict(ism_order=d["ism_order"], early_ms=d["early_ms"], max_len_s=d["brir_len_s"])
    full, early = make_brir(dim, rt60, lis, p, hrtf, fs, rng, **kw)
    noise = []
    for _ in range(d["n_noise_dirs"]):
        q, *_ = place_source(dim, lis, rng, (0.5, 6.0), (-30.0, 60.0))
        noise.append(make_brir(dim, rt60, lis, q, hrtf, fs, rng, **kw)[0])
    meta = {"dim": dim.tolist(), "rt60": rt60, "az": az, "el": el, "dist": r}
    return full, early, np.stack(noise), meta


def make_bank(n_rooms: int, cfg_rooms: dict, hrtf: HrtfSet, fs: int, seed: int, workers: int = 1) -> dict:
    """``n_rooms`` rooms, each with one target BRIR (full+early) and
    ``n_noise_dirs`` noise BRIRs from random positions in the same room."""
    args = [(i, cfg_rooms, hrtf, fs, seed) for i in range(n_rooms)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as ex:
            rooms = list(ex.map(make_room, *zip(*args), chunksize=4))
    else:
        rooms = [make_room(*a) for a in args]
    full, early, noise, meta = zip(*rooms)
    return {"target_full": np.stack(full), "target_early": np.stack(early), "noise": np.stack(noise),
            "meta": list(meta), "hrtf": hrtf.name}
