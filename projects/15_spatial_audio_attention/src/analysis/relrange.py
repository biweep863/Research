"""Self-adapting near / mid / far range from the microphone alone (no room calibration).

Absolute range from one array needs a reference (a quiet voice near sounds like
a loud voice far). Instead of a per-room model, the reference is learned online
from what the robot hears during the session:

- the *noise* reference is the silence model (Phi_n, re-learned whenever nobody
  talks), so the speech level is measured as noise-subtracted speech power in
  the voice band (``speech_db``), gated by its SNR;
- the *voice* reference is the history of past utterances: each finished
  utterance commits its median ``speech_db``; the current utterance is
  classified by how far it sits from the history's median, in units of the
  history's spread (floored at ``min_band_db``, a bit over half a doubling of
  distance under the inverse-square law, so a talker who never moves stays "mid");
- before enough history exists, a hardware prior (the level of normal speech
  at 1 m on the raw mics: a property of the ReSpeaker's fixed gain, not the
  room) stands in as pseudo-utterances that real ones wash out.
"""

from __future__ import annotations

from collections import deque

import numpy as np

BANDS = ("near", "mid", "far")


class RelativeRange:
    def __init__(
        self,
        prior_speech_db_1m: float | None = None,
        prior_weight: int = 2,
        history: int = 20,
        min_band_db: float = 3.0,
        spread_k: float = 1.0,
        hysteresis_db: float = 0.5,
        min_snr_db: float = 3.0,
        band_radius_m=(0.75, 1.75, 3.0),
    ):
        self.hist: deque = deque(maxlen=history)
        self.prior = [prior_speech_db_1m] * prior_weight if prior_speech_db_1m is not None else []
        self.min_band_db, self.spread_k = min_band_db, spread_k
        self.hyst, self.min_snr_db = hysteresis_db, min_snr_db
        self.radius = dict(zip(BANDS, band_radius_m))
        self.cur: list[float] = []  # speech_db samples of the ongoing utterance
        self.band: str | None = None

    @classmethod
    def from_config(cls, rcfg: dict | None) -> "RelativeRange":
        rcfg = dict(rcfg or {})
        rcfg.pop("enabled", None)
        rcfg.pop("cue", None)  # read by the localizer
        if "band_radius_m" in rcfg:
            rcfg["band_radius_m"] = tuple(rcfg["band_radius_m"])
        return cls(**rcfg)

    # ---------------------------------------------------------------- state
    def reference(self) -> tuple[float, float] | None:
        """(centre dB, half-width dB) of the 'mid' band, or None without any reference."""
        ref = list(self.hist) or self.prior
        if len(self.hist) < 3:
            ref = list(self.hist) + self.prior
        if not ref:
            return None
        c = float(np.median(ref))
        mad = 1.4826 * float(np.median(np.abs(np.asarray(ref) - c)))
        return c, max(self.min_band_db, self.spread_k * mad)

    def update(self, speech_db: float, snr_db: float) -> tuple[str | None, float]:
        """Feed one speech-frame measurement -> (band, display radius m)."""
        if not (np.isfinite(speech_db) and np.isfinite(snr_db)) or snr_db < self.min_snr_db:
            return self.band, self.radius.get(self.band, float("nan"))
        self.cur.append(speech_db)
        level = float(np.median(self.cur))
        ref = self.reference()
        if ref is None:  # first utterance, no prior: nothing to compare with
            self.band = "mid"
        else:
            c, w = ref
            h = self.hyst
            # leaving the current band costs an extra `h` dB (no flicker at the edges)
            near_thr = c + w + (h if self.band != "near" else -h)
            far_thr = c - w - (h if self.band != "far" else -h)
            self.band = "near" if level > near_thr else "far" if level < far_thr else "mid"
        return self.band, self.radius[self.band]

    def end_utterance(self) -> None:
        """Commit the finished utterance to the history (needs a few samples)."""
        if len(self.cur) >= 3:
            self.hist.append(float(np.median(self.cur)))
        self.cur = []

    def reset(self) -> None:
        self.hist.clear()
        self.cur = []
        self.band = None
