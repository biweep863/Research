"""Speaker distance from acoustic cues (TDOA cannot give range on a 6.4 cm array).

Features, all computed in the voice band from speech-frame cross-spectra
``Phi_x`` and silence-frame cross-spectra ``Phi_n`` (the noise / "sound of
silence" model):

- ``cdr_db``   Coherent-to-diffuse ratio. Direct sound is coherent across
               mics, reverberation is diffuse; the direct/reverberant ratio
               falls ~6 dB per doubling of distance beyond the critical
               distance. Schwarz & Kellermann (2015) estimators; the
               DOA-independent one is the default (see ``cdr_doa_independent``).
- ``level_db`` Speech power at the mics (dBFS) - inverse-square law, needs an
               assumed vocal effort and a fixed mic gain.
- ``snr_db``   Speech power over the silence noise floor.
- ``tilt_db``  High (2-3.8 kHz) over low (300-1000 Hz) speech energy; air
               absorption + reverberation colour the far field.

``DistanceModel`` maps features to log-distance with ridge regression and
falls back to an inverse-square law on ``level_db`` until calibrated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geometry import SPEED_OF_SOUND, far_field_delays, mic_pairs

FEATURES = ["cdr_db", "level_db", "snr_db", "tilt_db"]
_EPS = 1e-12


def diffuse_coherence(freqs: np.ndarray, d: float, c: float = SPEED_OF_SOUND):
    """Spherically isotropic diffuse-field coherence sinc(2 pi f d / c)."""
    return np.sinc(2.0 * freqs * d / c)


def _band_power(Phi: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    sel = (freqs >= lo) & (freqs <= hi)
    diag = np.real(np.einsum("fii->fi", Phi[sel]))
    return float(np.maximum(diag, 0).mean()) if sel.any() else 0.0


def cdr_doa_dependent(gx, gn, gs):
    """Schwarz & Kellermann (2015) DOA-dependent unbiased estimator."""
    return np.real((gn - gx) / (gx - gs + _EPS))


def cdr_doa_independent(gx, gn):
    """Schwarz & Kellermann (2015) DOA-independent estimator ("proposed 2").

    Preferred here: the talker's mouth is ~0.4 m above the array, so a
    horizontal-only steering vector mis-models the direct-path phase,
    especially at short range where the elevation is largest.
    """
    gn = np.real(gn)
    rx, ax2 = np.real(gx), np.abs(gx) ** 2
    disc = gn**2 * rx**2 - gn**2 * ax2 + gn**2 - 2 * gn * rx + ax2
    num = gn * rx - ax2 - np.sqrt(np.maximum(disc, 0.0))
    return num / np.minimum(ax2 - 1.0, -_EPS)


def cdr_db(
    Phi_s: np.ndarray,
    freqs: np.ndarray,
    mics: np.ndarray,
    az_rad: float | None = None,
    band=(400.0, 7000.0),
    c: float = SPEED_OF_SOUND,
    max_diffuse_coherence: float = 0.7,
) -> float:
    """Pair- and frequency-averaged CDR in dB from a noise-reduced Phi_s (F,M,M).

    ``az_rad=None`` uses the DOA-independent estimator. Per pair, only bins
    where the diffuse coherence is below ``max_diffuse_coherence`` are used:
    on a 4.5-6.4 cm pair the diffuse field is still nearly coherent below
    ~1.5 kHz, where every CDR estimator is ill-conditioned and biased high.
    Spatial aliasing does not affect CDR, so the band extends past 3.8 kHz.
    """
    tau = None if az_rad is None else far_field_delays(mics, np.array([az_rad]), c)[0]
    pair_db = []
    for i, j in mic_pairs(mics.shape[0]):
        d = float(np.linalg.norm(mics[i] - mics[j]))
        sel = (freqs >= band[0]) & (freqs <= band[1])
        sel &= np.abs(diffuse_coherence(freqs, d, c)) < max_diffuse_coherence
        if not sel.any():
            continue
        f = freqs[sel]
        pii = np.real(Phi_s[sel, i, i])
        pjj = np.real(Phi_s[sel, j, j])
        gx = Phi_s[sel, i, j] / np.sqrt(np.maximum(pii * pjj, _EPS))
        gn = diffuse_coherence(f, d, c)
        if tau is None:
            cdr = cdr_doa_independent(gx, gn)
        else:
            cdr = cdr_doa_dependent(gx, gn, np.exp(-2j * np.pi * f * (tau[i] - tau[j])))
        cdr = np.clip(cdr, 1e-3, 1e3)
        pair_db.append(10 * np.log10(np.median(cdr)))
    # median over pairs: one degenerate pair (e.g. a pair that is perfectly
    # symmetric w.r.t. source and room) must not dominate
    return float(np.median(pair_db)) if pair_db else float("nan")


def extract_features(
    Phi_x: np.ndarray,
    Phi_n: np.ndarray | None,
    freqs: np.ndarray,
    mics: np.ndarray,
    az_rad: float,
    c: float = SPEED_OF_SOUND,
    cdr_method: str = "doa_independent",
) -> dict[str, float]:
    if Phi_n is None:
        Phi_n = np.zeros_like(Phi_x)
    Phi_s = Phi_x - Phi_n
    p_x = _band_power(Phi_x, freqs, 200, 3800)
    p_s = max(_band_power(Phi_s, freqs, 200, 3800), _EPS)
    p_n = max(_band_power(Phi_n, freqs, 200, 3800), _EPS)
    p_hi = max(_band_power(Phi_s, freqs, 2000, 3800), _EPS)
    p_lo = max(_band_power(Phi_s, freqs, 300, 1000), _EPS)
    return {
        "cdr_db": cdr_db(Phi_s, freqs, mics, None if cdr_method == "doa_independent" else az_rad, c=c),
        "level_db": float(10 * np.log10(max(p_x, _EPS))),
        # noise-subtracted speech power: the level cue for the online relative range
        "speech_db": float(10 * np.log10(p_s)),
        "snr_db": float(10 * np.log10(p_s / p_n)),
        "tilt_db": float(10 * np.log10(p_hi / p_lo)),
    }


@dataclass
class DistanceModel:
    """log(r) = b + sum_k w_k * (x_k - mu_k) / sd_k, ridge-fitted.

    Untrained default: inverse-square on level_db with a reference level at
    1 m (``ref_level_db``) -> r = 10 ** ((ref - level) / 20).
    """

    features: list[str] = field(default_factory=lambda: list(FEATURES))
    mu: list[float] | None = None
    sd: list[float] | None = None
    w: list[float] | None = None
    b: float = 0.0
    ref_level_db: float = -30.0
    min_range: float = 0.3
    max_range: float = 5.0
    source: str = "inverse-square prior"

    @property
    def trained(self) -> bool:
        return self.w is not None

    def predict(self, feats: dict[str, float]) -> float:
        if self.trained:
            x = np.array([feats[k] for k in self.features])
            z = (x - np.array(self.mu)) / np.array(self.sd)
            log_r = self.b + float(np.dot(self.w, z))
            r = float(np.exp(log_r))
        else:
            r = 10 ** ((self.ref_level_db - feats["level_db"]) / 20.0)
        return float(np.clip(r, self.min_range, self.max_range))

    @classmethod
    def fit(
        cls,
        rows: list[dict[str, float]],
        distances: list[float],
        features: list[str] | None = None,
        ridge: float = 1.0,
        **kw,
    ) -> "DistanceModel":
        features = list(features or FEATURES)
        X = np.array([[r[k] for k in features] for r in rows], dtype=float)
        y = np.log(np.asarray(distances, dtype=float))
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        b = float(y.mean())
        A = Z.T @ Z + ridge * np.eye(len(features))
        w = np.linalg.solve(A, Z.T @ (y - b))
        return cls(features, mu.tolist(), sd.tolist(), w.tolist(), b, **kw)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "DistanceModel":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def load_or_default(cls, path: str | Path | None, **kw) -> "DistanceModel":
        if path and Path(path).exists():
            return cls.from_json(path)
        return cls(**kw)
