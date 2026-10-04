"""Room simulation of the ReSpeaker array with a voiced talker, diffuse noise,
optional point noise (fan) and optional interfering talker.

Uses pyroomacoustics (image-source method) when available; falls back to a
free-field fractional-delay model otherwise (no reverberation -> distance
cues other than level are meaningless in the fallback, the experiments say so).
"""

from __future__ import annotations

import numpy as np

from ..analysis.geometry import SPEED_OF_SOUND, respeaker_v2_positions

try:
    import pyroomacoustics as pra

    HAS_PRA = True
except ImportError:  # pragma: no cover
    HAS_PRA = False

# Rough vowel formants (Hz) - enough spectral variety for MFCC/tilt features.
_VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]


def voiced_speech(
    duration_s: float,
    fs: int,
    f0: float = 140.0,
    seed: int = 0,
    syllable_hz: float = 4.0,
    pause_prob: float = 0.2,
) -> np.ndarray:
    """Speech-like signal: harmonic source with F0 contour, vowel formants,
    syllabic envelope, word pauses and a little aspiration noise. Unit RMS
    over the voiced part. Generalises home2 ``vad_benchmark.gen_voiced_speech``.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    drift = np.cumsum(rng.normal(0, 0.002, n))
    drift -= np.linspace(drift[0], drift[-1], n)
    f0_t = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.8 * t + rng.uniform(0, 6.28)) + drift)
    phase = 2 * np.pi * np.cumsum(f0_t) / fs

    syl_len = int(fs / syllable_hz)
    n_syl = int(np.ceil(n / syl_len))
    out = np.zeros(n)
    k_max = int((fs / 2 - 200) // (f0 * 1.1))
    for s in range(n_syl):
        a, b = s * syl_len, min(n, (s + 1) * syl_len)
        if rng.random() < pause_prob:
            continue
        formants = np.array(_VOWELS[rng.integers(len(_VOWELS))]) * rng.uniform(0.9, 1.1)
        seg = np.zeros(b - a)
        for k in range(1, k_max + 1):
            fk = k * f0
            amp = (1.0 / k) * sum(np.exp(-0.5 * ((fk - F) / (60 + 0.06 * F)) ** 2) for F in formants)
            seg += (amp + 0.02 / k) * np.sin(k * phase[a:b])
        env = np.sin(np.pi * np.linspace(0, 1, b - a)) ** 0.6
        out[a:b] = seg * env
    voiced = np.abs(out) > 0
    out += 0.02 * rng.standard_normal(n) * voiced
    rms = np.sqrt(np.mean(out[voiced] ** 2)) if voiced.any() else 1.0
    return out / rms


def pink_noise(n: int, rng) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    spec /= np.sqrt(np.maximum(f, 1))
    x = np.fft.irfft(spec, n)
    return x / x.std()


def diffuse_noise(mics_xy: np.ndarray, n: int, fs: int, rng, nfft: int = 512, c=SPEED_OF_SOUND):
    """Spherically diffuse pink noise with the exact sinc coherence (Habets-style
    Cholesky mixing in the STFT domain). Unit RMS per channel."""
    import scipy.signal

    M = mics_xy.shape[0]
    hop = nfft // 2
    n_frames = n // hop + 3
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    D = np.linalg.norm(mics_xy[:, None] - mics_xy[None], axis=-1)
    Z = (rng.standard_normal((M, len(freqs), n_frames)) + 1j * rng.standard_normal((M, len(freqs), n_frames)))
    Z /= np.sqrt(np.maximum(freqs, 50.0))[None, :, None]
    for fi, f in enumerate(freqs):
        G = np.sinc(2 * f * D / c) + 1e-6 * np.eye(M)
        L = np.linalg.cholesky(G)
        Z[:, fi] = L @ Z[:, fi]
    _, x = scipy.signal.istft(Z, fs=fs, nperseg=nfft, noverlap=nfft - hop)
    x = x[:, :n]
    return x / x.std(axis=1, keepdims=True)


def _mic_positions_3d(cfg):
    sim = cfg["sim"]
    xy = respeaker_v2_positions(cfg["audio"]["mic_radius"])
    c = np.array(sim["array_pos"], dtype=float)
    return np.vstack([xy.T + c[:2, None], np.full(xy.shape[0], c[2])]), xy


def source_position(cfg, az_deg: float, dist_m: float, height: float | None = None):
    sim = cfg["sim"]
    c = np.array(sim["array_pos"], dtype=float)
    h = sim["source_height"] if height is None else height
    pos = np.array([c[0] + dist_m * np.cos(np.deg2rad(az_deg)), c[1] + dist_m * np.sin(np.deg2rad(az_deg)), h])
    room = np.array(sim["room_dim"], dtype=float)
    if np.any(pos < 0.2) or np.any(pos > room - 0.2):
        raise ValueError(f"source at az={az_deg} d={dist_m} falls outside the room {room}")
    return pos


def simulate_scene(
    cfg: dict,
    sources: list[dict],
    rt60: float,
    noise_dbfs: float | None = -55.0,
    snr_db: float | None = None,
    fan: dict | None = None,
    seed: int = 0,
):
    """Simulate the 4 ReSpeaker raw channels.

    sources: [{"signal": np.ndarray (unit RMS), "az_deg", "dist_m", "level_dbfs_1m", "delay_s"}]
    Noise level: absolute ``noise_dbfs`` (keeps level/SNR distance cues physical)
    or relative ``snr_db`` to the first source's speech power at the mics.
    Returns dict(mix (4,T), images (n_src,4,T), noise (4,T), fs).
    """
    fs = int(cfg["audio"]["fs"])
    sim = cfg["sim"]
    rng = np.random.default_rng(seed)
    mic3d, mic_xy = _mic_positions_3d(cfg)
    n_len = max(int((s.get("delay_s", 0) + len(s["signal"]) / fs) * fs) for s in sources)
    n_len += int(0.3 * fs)

    all_src = list(sources) + ([fan] if fan else [])
    # pyroomacoustics propagates with 1/r (not 1/4 pi r): gain = level at 1 m
    gains = [10 ** (s.get("level_dbfs_1m", -30.0) / 20.0) for s in all_src]

    if HAS_PRA:
        room_dim = np.array(sim["room_dim"], dtype=float)
        e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_abs),
            max_order=int(min(max_order, sim["max_order"])), air_absorption=True,
        )
        for s, g in zip(all_src, gains):
            pos = source_position(cfg, s["az_deg"], s["dist_m"], s.get("height"))
            room.add_source(pos, signal=g * s["signal"], delay=s.get("delay_s", 0.0))
        room.add_microphone_array(mic3d)
        premix = room.simulate(return_premix=True)  # (n_src, M, T)
        premix = premix[:, :, :n_len]
        if premix.shape[-1] < n_len:
            premix = np.pad(premix, ((0, 0), (0, 0), (0, n_len - premix.shape[-1])))
    else:  # free-field fallback
        premix = np.zeros((len(all_src), mic3d.shape[1], n_len))
        for k, (s, g) in enumerate(zip(all_src, gains)):
            pos = source_position(cfg, s["az_deg"], s["dist_m"], s.get("height"))
            sig = np.zeros(n_len)
            d0 = int(s.get("delay_s", 0) * fs)
            sig[d0 : d0 + len(s["signal"])] = g * s["signal"][: n_len - d0]
            S = np.fft.rfft(sig)
            f = np.fft.rfftfreq(n_len, 1 / fs)
            for m in range(mic3d.shape[1]):
                r = np.linalg.norm(pos - mic3d[:, m])
                premix[k, m] = np.fft.irfft(S * np.exp(-2j * np.pi * f * r / SPEED_OF_SOUND), n_len) / r

    images = premix[: len(sources)]
    fan_img = premix[len(sources)] if fan else 0.0

    noise = diffuse_noise(mic_xy, n_len, fs, rng)
    noise += 10 ** (-10 / 20) * rng.standard_normal(noise.shape)  # self-noise of the mics
    noise /= noise.std()
    if fan:  # directional noise at the same power as the diffuse part
        noise = (noise + fan_img / (fan_img.std() + 1e-12)) / np.sqrt(2.0)
    if snr_db is not None:
        ref = images[0]
        active = np.abs(ref[0]) > 1e-3 * np.abs(ref[0]).max()
        p_s = np.mean(ref[:, active] ** 2)
        noise *= np.sqrt(p_s / 10 ** (snr_db / 10))
    else:
        noise *= 10 ** (noise_dbfs / 20.0)
    mix = images.sum(axis=0) + noise
    return {"mix": mix, "images": images, "noise": noise, "fs": fs}


def to_respeaker_6ch(mix4: np.ndarray) -> np.ndarray:
    """Pack 4 raw mics into the ReSpeaker 6-ch firmware layout:
    ch0 = processed (here: mic average), ch1..4 = raw mics, ch5 = playback ref (zeros)."""
    T = mix4.shape[1]
    return np.vstack([mix4.mean(axis=0, keepdims=True), mix4, np.zeros((1, T))])


def moving_speaker_demo(cfg: dict, seed: int = 0):
    """Talker walks around the robot: one utterance + pause per waypoint.
    Returns (6ch (6,T), fs, waypoints[list of dict(t0,t1,az_deg,dist_m)])."""
    d = cfg["radar_demo"]
    fs = int(cfg["audio"]["fs"])
    rng = np.random.default_rng(seed)
    chunks, waypoints, t = [], [], 0.0
    for i, (az, dist) in enumerate(d["waypoints"]):
        sig = voiced_speech(d["utterance_s"], fs, f0=d["f0_hz"], seed=seed + i)
        lead = int(d["pause_s"] * fs)
        sig = np.concatenate([np.zeros(lead), sig])
        out = simulate_scene(
            cfg,
            [{"signal": sig, "az_deg": az, "dist_m": dist, "level_dbfs_1m": cfg["sim"]["source_level_dbfs_1m"]}],
            rt60=d["rt60_s"], noise_dbfs=-140.0, seed=seed + i,
        )["images"][0]
        chunks.append(out)
        waypoints.append({"t0": t + d["pause_s"], "t1": t + out.shape[1] / fs, "az_deg": az, "dist_m": dist})
        t += out.shape[1] / fs
    mix = np.concatenate(chunks, axis=1)
    mic_xy = respeaker_v2_positions(cfg["audio"]["mic_radius"])
    noise = diffuse_noise(mic_xy, mix.shape[1], fs, rng)
    noise += 10 ** (-10 / 20) * rng.standard_normal(noise.shape)
    noise *= 10 ** (d["noise_dbfs"] / 20) / noise.std()
    return to_respeaker_6ch(mix + noise), fs, waypoints
