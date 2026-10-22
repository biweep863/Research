"""Stage 2: each front end -> DeepFilterNet3 (atten_lim_db sweep, continuous and
home2 chunk mode) -> Whisper. Scored by WER against the read prompt, SI-SDR
against the clean reference mic and ESTOI against the close-talk Akane.

DFN runs in .venv-dfn (src/dfn_runner.py) in batches; its outputs are deleted
after scoring except for ``dfn.keep_audio_for`` (listening examples).
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np

from ..analysis import asr
from ..analysis.metrics import estoi, si_sdr, wer
from ..data.texts import TEXTS
from ..data.wavio import load_wav
from .common import ROOT, setup, work_dir, write


def _att(a):
    return "inf" if a is None else str(int(a))


def score(path, clean, akane, text, cfg, fs=16000) -> dict:
    y, _ = load_wav(path)
    y = y[0][: len(clean)]
    hyp = asr.transcribe(y, cfg["asr"]["model"], cfg["asr"]["language"])
    r = {"wer": wer(text, hyp), "hyp": hyp, "si_sdr": si_sdr(y, clean[: len(y)])}
    if akane is not None:
        r["estoi_akane"] = estoi(y, akane[: len(y)], fs)
    if cfg["dfn"].get("estoi_clean", True):
        r["estoi_clean"] = estoi(y, clean[: len(y)], fs)
    return r


def main():
    cfg, out = setup(__doc__)
    d = cfg["dfn"]
    wd = work_dir(cfg)
    sp = json.loads((out / "spatial.json").read_text())
    scenes = sp["scenes"]
    print(f"{len(scenes)} scenes x {len(d['methods'])} methods")

    # references: clean mic and close-talk mic, per utterance
    refs = {}
    rows = []
    for utt in sorted({s["utt"] for s in scenes}):
        clean = load_wav(wd / f"{utt}__clean.wav")[0][0]
        ak = wd / f"{utt}__akane.wav"
        akane = load_wav(ak)[0][0] if ak.exists() else None
        text = TEXTS[next(s["text_id"] for s in scenes if s["utt"] == utt)]
        refs[utt] = (clean, akane, text)
        for name, sig in (("clean", clean), ("akane", akane)):
            if sig is None:
                continue
            hyp = asr.transcribe(sig, cfg["asr"]["model"], cfg["asr"]["language"])
            rows.append({"utt": utt, "sid": f"{utt}__{name}", "method": name, "dfn": "off", "atten": None,
                         "wer": wer(text, hyp), "hyp": hyp})
    print("references:", {r["sid"]: round(r["wer"], 3) for r in rows})

    dfn_dir = wd / "dfn"
    batch = d.get("batch", 16)
    py = str((ROOT / d["python"]).resolve())
    runner = str(ROOT / "src" / "dfn_runner.py")

    def launch(b0):
        """Start DFN (CPU) for one batch; it runs while the previous batch is scored (GPU)."""
        jobs = []
        for s in scenes[b0 : b0 + batch]:
            for m in d["methods"]:
                inp = wd / f"{s['sid']}__{m}.wav"
                jobs.append({"in": str(inp), "out": str(dfn_dir / f"{s['sid']}__{m}"),
                             "modes": ["cont"], "attens": d["attens"]})
                if m in d["home2_methods"]:
                    jobs.append({"in": str(inp), "out": str(dfn_dir / f"{s['sid']}__{m}"),
                                 "modes": ["home2"], "attens": d["home2_attens"]})
        spec = wd / f"dfn_jobs_{b0}.json"
        spec.write_text(json.dumps({"jobs": jobs, "modes": ["cont"], "attens": d["attens"],
                                    "threads": d.get("threads", 8)}))
        return subprocess.Popen([py, runner, str(spec)], cwd=ROOT)

    done_sids = {r["sid"] for r in rows}
    if (out / "dfn_asr.json").exists() and d.get("resume", True):
        prev = json.loads((out / "dfn_asr.json").read_text())["rows"]
        rows = [r for r in prev if r["method"] not in ("clean", "akane")] + rows
        done_sids = {r["sid"] for r in prev}
        print(f"resuming: {len({r['sid'] for r in prev if r['method'] not in ('clean', 'akane')})} scenes already scored")
    starts = [b0 for b0 in range(0, len(scenes), batch)
              if any(s["sid"] not in done_sids for s in scenes[b0 : b0 + batch])]
    t0 = time.time()
    proc = launch(starts[0]) if starts else None
    for k, b0 in enumerate(starts):
        if proc.wait() != 0:
            raise SystemExit("dfn_runner failed")
        proc = launch(starts[k + 1]) if k + 1 < len(starts) else None
        for s in scenes[b0 : b0 + batch]:
            if s["sid"] in done_sids:
                continue
            clean, akane, text = refs[s["utt"]]
            for m in d["methods"]:
                conds = [("off", None, wd / f"{s['sid']}__{m}.wav")]
                conds += [("cont", a, dfn_dir / f"{s['sid']}__{m}__cont__att{_att(a)}.wav") for a in d["attens"]]
                if m in d["home2_methods"]:
                    conds += [("home2", a, dfn_dir / f"{s['sid']}__{m}__home2__att{_att(a)}.wav")
                              for a in d["home2_attens"]]
                for mode, a, p in conds:
                    r = score(p, clean, akane, text, cfg)
                    rows.append({"utt": s["utt"], "sid": s["sid"], "method": m, "dfn": mode,
                                 "atten": "inf" if (mode != "off" and a is None) else a, **r})
                    if mode != "off" and s["sid"] not in d["keep_audio_for"]:
                        p.unlink(missing_ok=True)
        done = min(b0 + batch, len(scenes))
        print(f"  {done}/{len(scenes)} scenes, {time.time() - t0:.0f} s")
        write(out / "dfn_asr.json", {"rows": rows, "config": {"dfn": d, "asr": cfg["asr"]}})


if __name__ == "__main__":
    main()
