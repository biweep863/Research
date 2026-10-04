"""Streaming speaker localizer: VAD -> SRP-PHAT -> distance features -> tracker.

One code path for live audio, WAV replay and the simulation experiments.
Input is float audio in [-1, 1], shape (M, n) with the 4 raw ReSpeaker mics.

Per STFT frame (nfft/hop from config):
  1. VAD on the mic-average of the last ``vad.window`` samples (home2 logic).
  2. Silence frames (outside the hangover) update the noise model:
     noise floor RMS and noise cross-spectra Phi_n.
  3. Speech frames update a fast (DOA) and a slow (distance) cross-spectrum.
  4. Every ``doa_every`` frames: SRP-PHAT on (Phi_fast - Phi_n), bins weighted
     by speech presence 1 - tr(Phi_n)/tr(Phi_fast) -> azimuth,
     distance features on (Phi_slow, Phi_n) -> range, tracker update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal

from .distance import DistanceModel, extract_features
from .doa import harmonic_weights, peak_azimuth, snr_weights, srp_phat
from .geometry import SPEED_OF_SOUND, respeaker_v2_positions
from .stft import ema_alpha
from .tracker import SpeakerTracker, circular_mean_deg
from .vad import INT16_SCALE, adaptive_threshold, frame_vad, update_floor


@dataclass
class FrameEstimate:
    t: float
    is_speech: bool
    f0_hz: float
    rms: float
    threshold: float
    az_deg: float = float("nan")  # raw measurement, array frame
    doa_conf: float = 0.0
    range_m: float = float("nan")  # raw measurement
    features: dict | None = None
    track_az_deg: float = float("nan")
    track_range_m: float = float("nan")
    track_conf: float = 0.0
    track_active: bool = False


class _Ema:
    """Bias-corrected exponential average of cross-spectral matrices."""

    def __init__(self, shape, alpha):
        self.acc = np.zeros(shape, dtype=complex)
        self.w = 0.0
        self.alpha = alpha

    def update(self, value):
        self.acc = (1 - self.alpha) * self.acc + self.alpha * value
        self.w = (1 - self.alpha) * self.w + self.alpha

    def reset(self):
        self.acc[:] = 0
        self.w = 0.0

    @property
    def value(self):
        return self.acc / self.w if self.w > 0 else None


@dataclass
class _Totals:
    """Whole-clip sums (offline analysis / calibration)."""

    Phi_x: np.ndarray | None = None
    n_x: int = 0
    Phi_n: np.ndarray | None = None
    n_n: int = 0
    azs: list = field(default_factory=list)
    confs: list = field(default_factory=list)
    f0s: list = field(default_factory=list)


class Localizer:
    def __init__(self, cfg: dict, distance_model: DistanceModel | None = None):
        a, v, t = cfg["audio"], cfg["vad"], cfg["tracker"]
        self.fs = int(a["fs"])
        self.nfft, self.hop = int(a["nfft"]), int(a["hop"])
        self.c = float(a.get("c", SPEED_OF_SOUND))
        self.band = tuple(a["band_hz"])
        self.mics = respeaker_v2_positions(a["mic_radius"])
        self.M = self.mics.shape[0]
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / self.fs)
        self.grid = np.deg2rad(np.arange(0.0, 360.0, a["grid_step_deg"]))
        self.window = scipy.signal.get_window("hann", self.nfft)
        # scale so that mean_k |X_k|^2 equals the frame's mean power (dBFS features)
        self.window = self.window / np.sqrt(np.sum(self.window**2))
        self.cdr_method = a.get("cdr_method", "doa_independent")
        self.harmonic = bool(a.get("harmonic_weighting", False))
        self.noise_subtract = bool(a.get("srp_noise_subtract", True))
        self.snr_weighting = bool(a.get("srp_snr_weighting", True))
        self.doa_every = int(a.get("doa_every", 2))
        self.min_speech_frames = int(a.get("min_speech_frames", 4))

        self.vad_window = int(v["window"])
        self.abs_thr = float(v["energy_threshold_int16"]) / INT16_SCALE
        self.snr_gate_db = float(v["snr_gate_db"])
        self.corr_thr = float(v["correlation_threshold"])
        self.f0_range = tuple(v["f0_range_hz"])
        self.hangover = int(round(v["hangover_s"] * self.fs / self.hop))
        self.reset_frames = int(round(a.get("reset_after_silence_s", 0.6) * self.fs / self.hop))
        self.floor_up, self.floor_down = float(v["floor_up"]), float(v["floor_down"])

        shape = (len(self.freqs), self.M, self.M)
        self.fast = _Ema(shape, ema_alpha(a["doa_tau_s"], self.hop, self.fs))
        self.slow = _Ema(shape, ema_alpha(a["feat_tau_s"], self.hop, self.fs))
        self.noise = _Ema(shape, ema_alpha(a["noise_tau_s"], self.hop, self.fs))
        self.floor: float | None = None
        self.tracker = SpeakerTracker(**t)
        self.model = distance_model or DistanceModel()
        # room RT60 covariate for the distance model (config, or firmware estimate live)
        self.room_rt60 = cfg.get("distance", {}).get("room_rt60_s")
        self.totals = _Totals()

        self._buf = np.zeros((self.M, 0))
        self._consumed = 0  # absolute sample index of _buf[:, 0]
        self._next = 0  # absolute start sample of next frame
        self._frame = 0
        self._since_speech = 10**9
        self._speech_run = 0
        self.last_srp: np.ndarray | None = None
        self.last_f0 = 0.0
        self.last_features: dict | None = None

    # ------------------------------------------------------------------ io
    def process(self, block: np.ndarray) -> list[FrameEstimate]:
        block = np.asarray(block, dtype=np.float64)
        assert block.shape[0] == self.M, f"expected {self.M} channels"
        self._buf = np.concatenate([self._buf, block], axis=1)
        out = []
        end_abs = self._consumed + self._buf.shape[1]
        while self._next + self.nfft <= end_abs:
            out.append(self._process_frame(self._next - self._consumed))
            self._next += self.hop
        # the next frame needs [next, next+nfft) and its VAD window before that
        keep_abs = min(self._next, self._next + self.nfft - self.vad_window)
        keep_from = int(np.clip(keep_abs - self._consumed, 0, self._buf.shape[1]))
        self._buf = self._buf[:, keep_from:]
        self._consumed += keep_from
        return out

    # --------------------------------------------------------------- frame
    def _process_frame(self, p: int) -> FrameEstimate:
        t = (self._next + self.nfft) / self.fs
        frame = self._buf[:, p : p + self.nfft]
        vstart = max(0, p + self.nfft - self.vad_window)
        mono = self._buf[:, vstart : p + self.nfft].mean(axis=0)

        thr = adaptive_threshold(self.floor, self.abs_thr, self.snr_gate_db)
        vad = frame_vad(mono, self.fs, thr, self.corr_thr, self.f0_range)
        X = np.fft.rfft(frame * self.window[None, :], axis=-1)  # (M, F)
        xx = np.einsum("if,jf->fij", X, X.conj())

        est = FrameEstimate(t, vad.is_speech, vad.f0_hz, vad.rms, thr)
        if vad.is_speech:
            if self._since_speech > self.reset_frames:
                # new utterance after a long pause: the talker may have moved,
                # forget the old spatial statistics instead of blending them
                self.fast.reset()
                self.slow.reset()
            self._since_speech = 0
            self._speech_run += 1
            self.last_f0 = vad.f0_hz
            self.fast.update(xx)
            self.slow.update(xx)
            tot = self.totals
            tot.Phi_x = xx if tot.Phi_x is None else tot.Phi_x + xx
            tot.n_x += 1
            tot.f0s.append(vad.f0_hz)
        else:
            self._since_speech += 1
            self._speech_run = 0
            if self._since_speech > self.hangover:
                self.floor = update_floor(self.floor, vad.rms, self.floor_up, self.floor_down)
                self.noise.update(xx)
                tot = self.totals
                tot.Phi_n = xx if tot.Phi_n is None else tot.Phi_n + xx
                tot.n_n += 1

        if vad.is_speech and self._frame % self.doa_every == 0 and self.fast.w > 0:
            self._localize(est)
        else:
            st = self.tracker.predict(t)
            self._copy_track(est, st)
        self._frame += 1
        return est

    def _localize(self, est: FrameEstimate) -> None:
        Rx = self.fast.value
        Rn = self.noise.value
        P = self._srp(Rx, Rn, self.last_f0)
        az, conf = peak_azimuth(P, self.grid)
        self.last_srp = P
        est.az_deg, est.doa_conf = float(np.rad2deg(az)), conf
        self.totals.azs.append(est.az_deg)
        self.totals.confs.append(conf)

        range_m = None
        if self.slow.w > 0 and self._speech_run >= self.min_speech_frames:
            feats = extract_features(
                self.slow.value, Rn, self.freqs, self.mics, az, self.c, self.cdr_method
            )
            self._add_room(feats)
            range_m = self.model.predict(feats)
            est.features, est.range_m = feats, range_m
            self.last_features = feats
        st = self.tracker.update(est.t, est.az_deg, range_m, conf)
        self._copy_track(est, st)

    def _srp(self, Rx, Rn, f0):
        """SRP-PHAT with the two uses of the silence model: subtract the noise
        cross-spectra (removes directional noise such as a fan) and weight bins
        by speech presence. Optional F0-harmonic weighting from the VAD pitch."""
        R, w = Rx, np.ones_like(self.freqs)
        if self.noise_subtract and Rn is not None:
            R = Rx - Rn
        if self.snr_weighting and Rn is not None:
            w = w * snr_weights(Rx, Rn)
        if self.harmonic:
            w = w * harmonic_weights(self.freqs, f0)
        return srp_phat(R, self.freqs, self.mics, self.grid, self.band, self.c, w)

    def _add_room(self, feats: dict) -> None:
        if self.room_rt60 is not None:
            feats["rt60_s"] = float(self.room_rt60)

    @staticmethod
    def _copy_track(est: FrameEstimate, st) -> None:
        est.track_az_deg = st.az_deg if st.active else float("nan")
        est.track_range_m = st.range_m if st.active else float("nan")
        est.track_conf = st.confidence
        est.track_active = st.active

    # ------------------------------------------------------------ offline
    def clip_summary(self) -> dict:
        """Whole-clip DOA + distance features (static speaker assumed)."""
        tot = self.totals
        if tot.n_x == 0:
            return {"n_speech_frames": 0}
        Phi_x = tot.Phi_x / tot.n_x
        Phi_n = tot.Phi_n / tot.n_n if tot.n_n else None
        f0 = float(np.median(tot.f0s))
        P = self._srp(Phi_x, Phi_n, f0)
        az, conf = peak_azimuth(P, self.grid)
        feats = extract_features(Phi_x, Phi_n, self.freqs, self.mics, az, self.c, self.cdr_method)
        self._add_room(feats)
        return {
            "n_speech_frames": tot.n_x,
            "n_noise_frames": tot.n_n,
            "az_deg_pooled": float(np.rad2deg(az)),
            "conf_pooled": conf,
            "az_deg_framewise_mean": circular_mean_deg(tot.azs, tot.confs) if tot.azs else float("nan"),
            "f0_median_hz": f0,
            "features": feats,
            "range_m": self.model.predict(feats),
        }


def analyze(x: np.ndarray, cfg: dict, model: DistanceModel | None = None, block: int = 1024):
    """Run the localizer over a (M, T) clip in streaming blocks."""
    loc = Localizer(cfg, model)
    frames = []
    for s in range(0, x.shape[1], block):
        frames.extend(loc.process(x[:, s : s + block]))
    return loc, frames
