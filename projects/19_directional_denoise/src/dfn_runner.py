"""DeepFilterNet3 batch runner. Runs in its own venv (.venv-dfn, Python 3.11):
DeepFilterLib 0.5.6 has no Windows wheel for Python 3.12 (doc/05).

    .venv-dfn/Scripts/python.exe src/dfn_runner.py jobs.json

jobs.json: {"attens": [null, 6, 12, ...], "modes": ["cont", "home2"],
            "jobs": [{"in": "a.wav", "out": "dir/a"}, ...]}
writes <out>__<mode>__att<A>.wav (16 kHz mono) and skips existing files.

Modes:
- ``cont``  one enhance() over the whole utterance (DF keeps its recurrent state).
- ``home2`` exactly home2 ``noise_cancellation.py``: 1024-sample chunks at 16 kHz,
  each peak-normalised, upsampled x3, enhanced *independently*, downsampled,
  rescaled by its peak.
``atten`` null = no limit. Note DF treats 0 as "no limit" too, so the
"no DFN" condition is simply the input file.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from df.enhance import enhance, init_df

CHUNK = 1024
UP = 3


def _enh(model, state, x16: np.ndarray, atten) -> np.ndarray:
    peak = float(np.max(np.abs(x16)))
    if peak == 0:
        return x16
    x48 = scipy.signal.resample_poly(x16 / peak, UP, 1).astype(np.float32)
    with torch.no_grad():
        y48 = enhance(model, state, torch.from_numpy(x48)[None], atten_lim_db=atten).numpy()[0]
    y16 = scipy.signal.resample_poly(y48, 1, UP).astype(np.float32)[: len(x16)]
    if np.isnan(y16).any():
        return x16
    return y16 * peak


def run(model, state, x: np.ndarray, atten, mode: str) -> np.ndarray:
    if mode == "cont":
        return _enh(model, state, x, atten)
    out = np.zeros_like(x)
    for i in range(0, len(x), CHUNK):
        c = x[i : i + CHUNK]
        out[i : i + len(c)] = _enh(model, state, c, atten)[: len(c)]
    return out


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    torch.set_num_threads(spec.get("threads", 8))
    model, state, _ = init_df(log_level="ERROR")
    t0, n = time.time(), 0
    for job in spec["jobs"]:
        x, fs = sf.read(job["in"], dtype="float32")
        assert fs == 16000, fs
        for mode in job.get("modes", spec["modes"]):
            for a in job.get("attens", spec["attens"]):
                out = Path(f"{job['out']}__{mode}__att{'inf' if a is None else int(a)}.wav")
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, run(model, state, x, a, mode), 16000, subtype="FLOAT")
                n += 1
    print(f"dfn_runner: {n} files in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
