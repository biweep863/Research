"""Build any model of the sweep from a run spec + config.

A run spec is a dict ``{family, latency_ms, head, causal, lookahead}``. The
latency budget fixes the hop: ``latency = (2 + lookahead) * hop``.
"""

from __future__ import annotations

from .stft_net import StftEnhancer
from .tasnet import BinauralTasNet

FAMILIES = ("stft_asym", "stft_sym", "tasnet")


def hop_for(latency_ms: float, fs: int, lookahead: int = 0) -> int:
    total = round(latency_ms * fs / 1000)
    hop, rem = divmod(total, 2 + lookahead)
    if rem or hop < 1:
        raise ValueError(f"latency {latency_ms} ms is not (2+{lookahead}) whole hops at fs={fs}")
    return hop


def run_name(spec: dict) -> str:
    name = f"{spec['family']}_{spec['latency_ms']:g}ms_{spec.get('head', 'ear')}"
    if not spec.get("causal", True):
        name += "_nc"
    if spec.get("lookahead", 0):
        name += f"_la{spec['lookahead']}"
    if spec.get("steps"):
        name += f"_s{spec['steps']}"
    return name


def build_model(spec: dict, cfg: dict):
    fs = cfg["fs"]
    la = spec.get("lookahead", 0)
    hop = hop_for(spec["latency_ms"], fs, la)
    head = spec.get("head", "ear")
    causal = spec.get("causal", True)
    fam = spec["family"]
    if fam in ("stft_asym", "stft_sym"):
        m = cfg["model"]["stft"]
        n_fft = max(int(m["analysis_ms"] * fs / 1000), 2 * hop) if fam == "stft_asym" else 2 * hop
        n_fft = -(-n_fft // hop) * hop  # multiple of hop
        return StftEnhancer(
            n_fft, hop, hidden=m["hidden"], layers=m["layers"], df_order=m["df_order"],
            head=head, causal=causal, lookahead=la,
        )
    if fam == "tasnet":
        if la:
            raise ValueError("lookahead is only implemented for the STFT family")
        m = cfg["model"]["tasnet"]
        # >= 2 filters per frame sample so the +/- basis can pass the frame losslessly (doc/05)
        n_filters = max(m["n_filters"], 2 * 2 * hop)
        return BinauralTasNet(
            hop, n_filters=n_filters, bottleneck=m["bottleneck"], hidden=m["hidden"],
            kernel=m["kernel"], blocks=m["blocks"], repeats=m["repeats"], head=head, causal=causal,
        )
    raise ValueError(fam)
