"""Figures, LaTeX tables and macros for the format-tax project."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.metrics import holm, linfit, mcnemar_p, time_per_valid, wilcoxon_p  # noqa: E402
from src.data.renderers import READ_FORMATS  # noqa: E402
from utils.plotting import COL_W, TEXT_W, apply_style  # noqa: E402
from utils.stats import bootstrap_ci  # noqa: E402

# repo palette (project 19), fixed order; every series also has its own marker
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRAY, LIGHT = "#6b6b6b", "#a8a8a8"
MARK = ["o", "s", "^", "D", "v"]
FAMILY = {"txt": "plain", "md": "plain", "json": "markup", "html": "markup", "tex": "markup",
          "pdf_text": "extracted", "docx_text": "extracted", "pptx_text": "extracted", "xlsx_text": "extracted",
          "docx_xml": "raw XML", "pptx_xml": "raw XML", "xlsx_xml": "raw XML"}
FAM_COLOR = {"plain": GRAY, "markup": S[0], "extracted": S[2], "raw XML": S[1]}
LABEL = {"txt": "TXT", "md": "Markdown", "json": "JSON", "html": "HTML", "tex": "LaTeX",
         "pdf_text": "PDF (pypdf)", "docx_text": "DOCX (text)", "pptx_text": "PPTX (text)",
         "xlsx_text": "XLSX (cells)", "docx_xml": "DOCX (raw XML)", "pptx_xml": "PPTX (raw XML)",
         "xlsx_xml": "XLSX (raw XML)",
         "pdf_via_md": "PDF via MD", "pdf_via_html": "PDF via HTML", "pdf_via_tex": "PDF via LaTeX",
         "docx_code": "DOCX via python-docx", "docx_xml_w": "DOCX via raw XML", "pptx_code": "PPTX via python-pptx",
         "xlsx_code": "XLSX via openpyxl", "csv": "CSV"}
WRITE_ORDER = ["txt", "md", "json", "html", "tex", "pdf_via_md", "pdf_via_html", "pdf_via_tex",
               "docx_code", "docx_xml", "pptx_code", "xlsx_code", "csv"]
WRITE_LABEL = {**LABEL, "tex": "LaTeX (compiles)", "html": "HTML", "docx_xml": "DOCX via raw XML"}
DIGIT = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
         "7": "Seven", "8": "Eight", "9": "Nine"}


def _name(s: str) -> str:
    s = re.sub(r"(\d+)", lambda m: "".join(DIGIT[c] for c in m.group(1)), s)
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z]+", s) if w)


def _load(R: Path, pattern: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(R.glob(pattern))]


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def _log_ticks(axis, ticks):
    from matplotlib.ticker import NullFormatter
    axis.axes.set_xscale("log") if axis is axis.axes.xaxis else axis.axes.set_yscale("log")
    lo, hi = axis.get_view_interval()
    ticks = [t for t in ticks if lo <= t <= hi]
    axis.set_ticks(ticks, [f"{t:g}" for t in ticks])
    axis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------- Phase R
def fig_tokens(read, out, mac):
    df = pd.DataFrame(read["rows"])
    toks = read["tokenizers"]
    main = toks[0]
    for t in toks:
        base = df[df.format == "txt"].set_index("doc")[f"tok_{t}"]
        df[f"ratio_{t}"] = df[f"tok_{t}"] / df.doc.map(base)
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))
    y = np.arange(len(READ_FORMATS))[::-1]
    for k, t in enumerate(toks):
        med = [df[df.format == f][f"ratio_{t}"].median() for f in READ_FORMATS]
        ax.scatter(med, y + (k - (len(toks) - 1) / 2) * 0.16, marker=MARK[k], s=18,
                   color=S[k] if k else GRAY, label=t, zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y, [LABEL[f] for f in READ_FORMATS])
    _log_ticks(ax.xaxis, [1, 1.5, 2, 3, 5, 8])
    ax.axvline(1, color=LIGHT, lw=0.8, zorder=1)
    ax.set_xlabel("tokens relative to plain text (median over docs)")
    ax.legend(title="tokenizer", loc="upper right", frameon=False)
    fig.savefig(out / "tokens.pdf")
    plt.close(fig)

    for f in READ_FORMATS:
        r = df[df.format == f]
        mac[f"TokRatio{_name(f)}"] = f"{r[f'ratio_{main}'].median():.2f}"
        mac[f"Density{_name(f)}"] = f"{100 * r[f'density_{main}'].median():.0f}"
        mac[f"Loader{_name(f)}"] = f"{1000 * r.loader_s.median():.1f}"
        # markup cost = fixed + per-content-token: tokens_fmt = a + b * tokens_txt
        x = df[df.format == "txt"].set_index("doc")[f"tok_{main}"].reindex(r.doc).values
        if len(set(x)) > 2:
            a, b, r2 = linfit(x, r[f"tok_{main}"].values)
            mac[f"Fixed{_name(f)}"], mac[f"Slope{_name(f)}"] = f"{a:.0f}", f"{b:.2f}"
    spread = {t: (df[df.format == "pptx_xml"][f"ratio_{t}"].median()) for t in toks}
    mac["PptxXmlRatioMin"], mac["PptxXmlRatioMax"] = f"{min(spread.values()):.1f}", f"{max(spread.values()):.1f}"
    mac["TxtTokens"] = f"{df[df.format == 'txt'][f'tok_{main}'].median():.0f}"
    return df


def table_overhead(df, read, tables):
    main = read["tokenizers"][0]
    has_fit = df.doc.nunique() > 2 and df[df.format == "txt"][f"tok_{main}"].nunique() > 2
    lines = [r"\begin{tabular}{lrrr" + ("rr" if has_fit else "") + "}", r"\toprule",
             r"Format & tokens/TXT & density & loader (ms)" + (r" & fixed & per TXT tok" if has_fit else "")
             + r" \\", r"\midrule"]
    x_all = df[df.format == "txt"].set_index("doc")[f"tok_{main}"]
    for f in READ_FORMATS:
        r = df[df.format == f]
        row = [LABEL[f], f"{r[f'ratio_{main}'].median():.2f}", f"{100 * r[f'density_{main}'].median():.0f}\\%",
               f"{1000 * r.loader_s.median():.1f}" if r.loader_s.max() > 0 else "--"]
        if has_fit:
            a, b, _ = linfit(x_all.reindex(r.doc).values, r[f"tok_{main}"].values)
            row += [f"{a:.0f}", f"{b:.2f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "overhead.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- Phase U
def understand_frames(runs):
    rows, pre = [], []
    for u in runs:
        m = u["model"]["short"]
        rows += [{**r, "model": m} for r in u["rows"]]
        pre += [{**r, "model": m} for r in u["prefill"]]
    return pd.DataFrame(rows), pd.DataFrame(pre)


def fig_understand(df, pre, models, out, mac, head):
    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W * 0.42), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(READ_FORMATS))[::-1]
    for k, m in enumerate(models):
        for cond, filled in (("native", True), ("budget", False)):
            pts = []
            for f in READ_FORMATS:
                c = df[(df.model == m) & (df.format == f) & (df.cond == cond)].correct.astype(float).values
                pts.append(bootstrap_ci(c, n_boot=500)[0] * 100)
            col = S[k]
            a1.scatter(pts, y + (k - (len(models) - 1) / 2) * 0.2, marker=MARK[k], s=20, zorder=3,
                       facecolor=col if filled else "white", edgecolor=col, linewidth=1,
                       label=f"{m}, {'full document' if filled else 'TXT token budget'}")
    a1.set_yticks(y, [LABEL[f] for f in READ_FORMATS])
    a1.set_xlabel("QA accuracy (%)")
    a1.set_xlim(-3, 103)
    a1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, frameon=False, fontsize=6)
    a1.set_title("Understanding: same content, different format", loc="left")

    p = pre[pre.cond == "native"]
    for k, m in enumerate(models):
        q = p[p.model == m]
        for fam, col in FAM_COLOR.items():
            r = q[q.format.map(FAMILY) == fam]
            a2.scatter(r.prefix_tokens, r.prefill_s, s=12, marker=MARK[k], color=col, alpha=0.85,
                       edgecolor="white", linewidth=0.3, label=fam if k == 0 else None)
    _log_ticks(a2.xaxis, [250, 500, 1000, 2000, 4000, 8000, 16000])
    _log_ticks(a2.yaxis, [0.1, 0.3, 1, 3, 10, 30, 100, 300])
    a2.set_xlabel("prompt tokens (document + instruction)")
    a2.set_ylabel("prefill time (s)")
    a2.legend(title="format family", frameon=False, loc="upper left")
    a2.set_title("Reading time: linear, then a memory cliff", loc="left")
    fig.tight_layout()
    fig.savefig(out / "understand.pdf")
    plt.close(fig)

    h = df[df.model == head]
    acc = {(f, c): h[(h.format == f) & (h.cond == c)].correct.mean() * 100
           for f in READ_FORMATS for c in ("native", "budget")}
    for f in READ_FORMATS:
        mac[f"Acc{_name(f)}"] = f"{acc[(f, 'native')]:.0f}"
        mac[f"AccBudget{_name(f)}"] = f"{acc[(f, 'budget')]:.0f}"
    nat = [acc[(f, "native")] for f in READ_FORMATS]
    mac["AccSpread"] = f"{max(nat) - min(nat):.0f}"
    mac["AccXmlBudgetMean"] = f"{np.mean([acc[(f, 'budget')] for f in READ_FORMATS if f.endswith('_xml')]):.0f}"
    # paired McNemar vs Markdown (same doc, question, condition), Holm-adjusted
    ref = h[(h.format == "md") & (h.cond == "native")].set_index(["doc", "qid"]).correct
    ps = {}
    for f in READ_FORMATS:
        if f == "md":
            continue
        o = h[(h.format == f) & (h.cond == "native")].set_index(["doc", "qid"]).correct.reindex(ref.index)
        ps[f] = mcnemar_p(list(o.values), list(ref.values))
    adj = holm(ps)
    mac["NSigVsMd"] = str(sum(v < 0.05 for v in adj.values()))
    mac["MinPVsMd"] = f"{min(adj.values()):.2g}"
    # evidence-conditioned accuracy under the budget: does markup hurt beyond the lost content?
    b = h[(h.cond == "budget") & h.evidence_in_context]
    mac["AccBudgetEvidenceXml"] = f"{100 * b[b.format.str.endswith('_xml')].correct.mean():.0f}"
    mac["AccBudgetEvidencePlain"] = f"{100 * b[b.format.isin(['txt', 'md'])].correct.mean():.0f}"
    nq = h[(h.format == "txt") & (h.cond == "native")]
    mac["NQuestions"], mac["NQuestionsPerDoc"] = str(len(nq)), f"{len(nq) / nq.doc.nunique():.0f}"
    hp = p[p.model == head]
    lin, cliff = prefill_regimes(hp)
    a, slope, r2 = linfit(lin.prefix_tokens, 1000 * lin.prefill_s)
    mac["PrefillMsPerKtok"], mac["PrefillRsq"] = f"{1000 * slope:.0f}", f"{r2:.2f}"
    rate = hp.prefill_s / hp.prefix_tokens
    mac["CliffTokens"] = f"{cliff:,}" if cliff else "none"
    mac["CliffSlowdown"] = f"{rate.max() / rate[hp.format == 'txt'].median():.0f}"
    for m in models:
        c = prefill_regimes(p[p.model == m])[1]
        mac[f"CliffTokens{_name(m)}"] = f"{c:,}" if c else "none"
    ratio = hp[hp.format == "pptx_xml"].prefill_s.median() / hp[hp.format == "txt"].prefill_s.median()
    mac["PrefillXmlOverTxt"] = f"{ratio:.1f}"
    return adj


def prefill_regimes(hp, factor=3.0):
    """Split prefill points into the linear regime and the memory cliff.

    The per-token rate of the TXT prompts is the base; a prompt whose rate is
    more than `factor` times that has left the GPU's memory (on Windows the
    driver spills to system RAM instead of failing). Returns the linear points
    and the smallest prompt past the cliff (None if no point is past it).
    """
    rate = hp.prefill_s / hp.prefix_tokens
    base = rate[hp.format == "txt"].median()
    past = hp[rate > factor * base]
    return hp[rate <= factor * base], (int(past.prefix_tokens.min()) if len(past) else None)


def table_accuracy(df, models, tables):
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{_tex(m)}}}" for m in models)
    lines = [r"\begin{tabular}{l" + "rr" * len(models) + "}", r"\toprule", f"Format & {head} \\\\",
             " & ".join([""] + ["full", "budget"] * len(models)) + r" \\", r"\midrule"]
    for f in READ_FORMATS:
        cells = [LABEL[f]]
        for m in models:
            for c in ("native", "budget"):
                cells.append(f"{100 * df[(df.model == m) & (df.format == f) & (df.cond == c)].correct.mean():.0f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "accuracy.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- Phase W
def write_frame(runs):
    return pd.DataFrame([{**r, "model": w["model"]["short"]} for w in runs for r in w["rows"]])


def write_summary(w: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, d), g in w.groupby(["model", "deliverable"]):
        out.append({"model": m, "deliverable": d, "valid": 100 * g.valid.mean(),
                    "first_valid": 100 * g.first_try_valid.mean(), "out_tokens": g.out_tokens.median(),
                    "time_per_valid": time_per_valid(list(g.total_s), list(g.valid)),
                    "recall": 100 * g.fact_recall.mean(), "chrf": g.chrf.mean(), "n": len(g)})
    s = pd.DataFrame(out)
    base = s[s.deliverable == "md"].set_index("model").time_per_valid
    s["tax"] = s.time_per_valid / s.model.map(base)
    return s


def fig_write(s, models, out):
    apply_style()
    order = [d for d in WRITE_ORDER if d in set(s.deliverable)]
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, TEXT_W * 0.42), sharey=True)
    y = np.arange(len(order))[::-1]
    never_rows = set()
    for k, m in enumerate(models):
        q = s[s.model == m].set_index("deliverable").reindex(order)
        off = y + (k - (len(models) - 1) / 2) * 0.22
        kw = dict(marker=MARK[k], s=20, color=S[k], zorder=3, edgecolor="white", linewidth=0.4)
        axs[0].scatter(q.valid, off, label=m, **kw)
        axs[0].scatter(q.first_valid, off, marker="|", s=30, color=S[k], zorder=2)
        tax = q.tax.replace(np.inf, np.nan)
        axs[1].scatter(tax, off, **kw)
        # never valid: an open marker at the left edge, one gray label per row (added below)
        bad = ~np.isfinite(q.tax.values)
        axs[1].scatter(np.full(bad.sum(), 0.03), off[bad], transform=axs[1].get_yaxis_transform(),
                       marker=MARK[k], s=16, facecolor="none", edgecolor=S[k], linewidth=0.8, zorder=3)
        never_rows.update(y[bad])
        axs[2].scatter(q.recall, off, **kw)
    axs[0].set_yticks(y, [WRITE_LABEL[d] for d in order])
    axs[0].set_xlabel("valid files (%)\n| = first try, marker = with retries")
    axs[0].set_xlim(-5, 105)
    _log_ticks(axs[1].xaxis, [0.1, 0.3, 1, 3, 10, 30, 100])
    for yy in never_rows:
        axs[1].annotate("never valid", (0.07, yy), xycoords=axs[1].get_yaxis_transform(), fontsize=5.5,
                        color=GRAY, va="center")
    axs[1].axvline(1, color=LIGHT, lw=0.8)
    axs[1].set_xlabel("time per valid file / Markdown\n(format tax, log)")
    axs[2].set_xlabel("planted facts in the built file (%)")
    axs[2].set_xlim(-5, 105)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "write.pdf")
    plt.close(fig)


def table_write(s, models, tables):
    order = [d for d in WRITE_ORDER if d in set(s.deliverable)]
    head = " & ".join(rf"\multicolumn{{4}}{{c}}{{{_tex(m)}}}" for m in models)
    sub = " & ".join(["valid", "tokens", "tax", "facts"] * len(models))
    lines = [r"\begin{tabular}{l" + "rrrr" * len(models) + "}", r"\toprule", f"Deliverable & {head} \\\\",
             f" & {sub} \\\\", r"\midrule"]
    for d in order:
        cells = [WRITE_LABEL[d]]
        for m in models:
            r = s[(s.model == m) & (s.deliverable == d)]
            if r.empty:
                cells += ["--"] * 4
                continue
            r = r.iloc[0]
            cells += [f"{r.valid:.0f}", f"{r.out_tokens:.0f}",
                      f"{r.tax:.1f}" if np.isfinite(r.tax) else r"$\infty$", f"{r.recall:.0f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "write.tex").write_text("\n".join(lines) + "\n")


def write_macros(s, w, head, mac):
    h = s[s.model == head].set_index("deliverable")
    for d in h.index:
        n = _name(d)
        mac[f"Valid{n}"] = f"{h.loc[d, 'valid']:.0f}"
        mac[f"FirstValid{n}"] = f"{h.loc[d, 'first_valid']:.0f}"
        mac[f"Tax{n}"] = f"{h.loc[d, 'tax']:.1f}" if np.isfinite(h.loc[d, "tax"]) else r"$\infty$"
        mac[f"OutTok{n}"] = f"{h.loc[d, 'out_tokens']:.0f}"
        mac[f"Recall{n}"] = f"{h.loc[d, 'recall']:.0f}"
    mac["NWriteDocs"] = str(w[w.model == head].doc.nunique())
    # paired time test: docx via code vs via raw XML (same docs)
    hw = w[w.model == head]
    a = hw[hw.deliverable == "docx_code"].set_index("doc").total_s
    b = hw[hw.deliverable == "docx_xml"].set_index("doc").total_s.reindex(a.index)
    mac["PDocxCodeVsXml"] = f"{wilcoxon_p(a.values, b.values):.2g}"
    if "docx_xml" in h.index and "docx_code" in h.index:
        mac["DocxXmlOverCodeTokens"] = f"{h.loc['docx_xml', 'out_tokens'] / h.loc['docx_code', 'out_tokens']:.1f}"


def retry_macros(runs, mac):
    """Per model: how many failed attempts were retried, fixed, or failed again with the same error."""
    for w in runs:
        att = {(a["doc"], a["gen"], a["attempt"]): a for a in w["attempts"]}
        nxt = [(a, att[(a["doc"], a["gen"], a["attempt"] + 1)]) for a in att.values()
               if not a["valid"] and (a["doc"], a["gen"], a["attempt"] + 1) in att]
        m = _name(w["model"]["short"])
        mac[f"Retries{m}"] = str(len(nxt))
        mac[f"RetriesFixed{m}"] = str(sum(n["valid"] for _, n in nxt))
        mac[f"RetriesSameError{m}"] = str(sum(n["error"] == a["error"] for a, n in nxt))
        mac[f"NWriteDocs{m}"] = str(len({r["doc"] for r in w["rows"]}))


# ---------------------------------------------------------------- lifecycle (read x write)
def fig_lifecycle(read_df, und, s, head, read_tok, out, mac):
    """One point per file type: read cost vs write tax for the headline model."""
    pairs = {"Markdown": ("md", "md"), "TXT": ("txt", "txt"), "JSON": ("json", "json"), "HTML": ("html", "html"),
             "LaTeX/PDF": ("pdf_text", "pdf_via_tex"), "DOCX": ("docx_text", "docx_code"),
             "PPTX": ("pptx_text", "pptx_code"), "XLSX": ("xlsx_text", "xlsx_code")}
    h = s[s.model == head].set_index("deliverable")
    finite = h.tax[np.isfinite(h.tax)]
    ceil = finite.max() * 2 if len(finite) else 2      # where never-valid routes are drawn
    nudge = {"TXT": (4, -9), "Markdown": (4, 3), "JSON": (-4, -10), "HTML": (-4, 6)}
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.8))
    never_names, never_x = [], []
    for name, (rf, wd) in pairs.items():
        if wd not in h.index:
            continue
        rr = read_df[read_df.format == rf][f"ratio_{read_tok}"].median()
        acc = und[(und.model == head) & (und.format == rf) & (und.cond == "native")].correct.mean() * 100
        tax = h.loc[wd, "tax"]
        never = not np.isfinite(tax)
        ax.scatter(rr, ceil if never else tax, s=24 + acc / 3, marker="^" if never else "o",
                   color=FAM_COLOR[FAMILY[rf]], edgecolor="white", linewidth=0.5, zorder=3)
        if never:
            never_names.append(f"{name} ({acc:.0f}%)")
            never_x.append(rr)
        else:
            dx, dy = nudge.get(name, (4, 2))
            ax.annotate(f"{name} ({acc:.0f}%)", (rr, tax), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6, ha="right" if dx < 0 else "left")
    if never_names:
        ax.annotate("never valid: " + ", ".join(never_names), (max(never_x), ceil), xytext=(6, -2),
                    textcoords="offset points", fontsize=6, va="center")
    ax.margins(x=0.18, y=0.12)
    ax.set_xlabel("read cost: tokens / TXT")
    ax.set_ylabel("write cost: time per valid file / MD")
    _log_ticks(ax.yaxis, [0.25, 0.5, 1, 2, 4, 8, 16, 32])
    ax.axhline(1, color=LIGHT, lw=0.8)
    ax.axvline(1, color=LIGHT, lw=0.8)
    ax.set_title("Read vs write cost (label: QA accuracy)", loc="left")
    fig.savefig(out / "lifecycle.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    ap.add_argument("--headline", default=None, help="model short for the macros (default: last model)")
    a = ap.parse_args()
    R, out = HERE / a.results, HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    tables = out.parent / ("tables" + out.name.removeprefix("figures"))
    tables.mkdir(parents=True, exist_ok=True)
    mac: dict[str, str] = {}

    read = json.loads((R / "read.json").read_text(encoding="utf-8"))
    read_df = fig_tokens(read, out, mac)
    table_overhead(read_df, read, tables)
    mac["NDocs"] = str(read_df.doc.nunique())

    und_runs, wr_runs = _load(R, "understand_*.json"), _load(R, "write_*.json")
    models = [u["model"]["short"] for u in und_runs]
    head = a.headline or (models[-1] if models else None)
    mac["HeadModel"] = _tex(head or "")
    if und_runs:
        und, pre = understand_frames(und_runs)
        fig_understand(und, pre, models, out, mac, head)
        table_accuracy(und, models, tables)
    if wr_runs:
        w = write_frame(wr_runs)
        wmodels = [x["model"]["short"] for x in wr_runs]
        s = write_summary(w)
        s.to_csv(R / "write_summary.csv", index=False)
        fig_write(s, wmodels, out)
        table_write(s, wmodels, tables)
        write_macros(s, w, head, mac)
        retry_macros(wr_runs, mac)
        mac["MaxAttempts"] = str(max(x.get("max_attempts", int(w.attempts.max())) for x in wr_runs))
        if und_runs:
            fig_lifecycle(read_df, und, s, head, read["tokenizers"][0], out, mac)

    lines = ["% auto-generated by src/figures/plot_results.py"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    (HERE / a.macros).write_text("\n".join(lines) + "\n")
    print(f"{len(mac)} macros -> {a.macros}; figures -> {a.out}; tables -> {tables}")


if __name__ == "__main__":
    main()
