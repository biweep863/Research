"""Phase U: does the format change what the model understands?

For every model x doc x read format x condition, the document is prefilled
once (timed: this is the read latency) and every question is answered from a
copy of that KV cache with greedy decoding.

Conditions:
  native  - the whole document in that format;
  budget  - the format truncated to the token count of the plain-text version
            (same context budget as .txt: markup-heavy formats lose content).
"""
from __future__ import annotations

import numpy as np

from ..analysis.metrics import fact_recall, score_answer
from ..data.renderers import READ_FORMATS
from .common import LM, common_prefix, corpus, rendered, setup, write

PROMPT = ("Read the document below and answer the question. Reply with only the answer, no explanation.\n\n"
          "<document>\n{doc}\n</document>\n\nQuestion: {q}")


def main():
    cfg, out, models = setup("Phase U: QA accuracy and read latency per format")
    docs = corpus(cfg)
    rend = rendered(cfg, docs)
    U = cfg["understand"]
    for spec in models:
        lm = LM(spec, cfg.get("device", "auto"))
        # warm-up so the first timed prefill is not a CUDA init
        lm.prefill(lm.chat_ids([{"role": "user", "content": "hello " * 200}]))
        rows, prefill = [], []
        for d in docs:
            r = rend[d.doc_id]
            n_txt = lm.n_tokens(r["txt"])
            for fmt in READ_FORMATS:
                for cond in U["conditions"]:
                    text = r[fmt]
                    if cond == "budget":
                        ids = lm.tok(text, add_special_tokens=False)["input_ids"]
                        if len(ids) > n_txt:
                            text = lm.tok.decode(ids[:n_txt])
                    prompts = [lm.chat_ids([{"role": "user", "content": PROMPT.format(doc=text, q=q.question)}])
                               for q in d.qa]
                    P = common_prefix(prompts)
                    reps = U["prefill_repeats"] if cond == "native" else 1
                    ts = []
                    for _ in range(reps):
                        cache, t = lm.prefill(prompts[0][:P])
                        ts.append(t)
                    prefill.append({"doc": d.doc_id, "format": fmt, "cond": cond, "prefix_tokens": P,
                                    "doc_tokens": lm.n_tokens(text), "prefill_s": float(np.median(ts))})
                    for q, ids in zip(d.qa, prompts):
                        ans, n_new, dt, _ = lm.generate(ids, U["max_new_tokens"], prefix_cache=cache)
                        rows.append({"doc": d.doc_id, "format": fmt, "cond": cond, "qid": q.qid, "kind": q.kind,
                                     "gold": q.answer, "pred": ans.strip()[:200],
                                     "correct": score_answer(ans, q.answer, q.distractor),
                                     "evidence_in_context": fact_recall(text, [q.answer]) == 1.0,
                                     "answer_s": dt, "answer_tokens": n_new})
                    del cache
                acc = np.mean([x["correct"] for x in rows if x["doc"] == d.doc_id and x["cond"] == "native"])
            print(spec["short"], d.doc_id, f"native acc so far {acc:.2f}", flush=True)
        write(out / f"understand_{spec['short']}.json",
              {"model": spec, "rows": rows, "prefill": prefill})
        lm.free()


if __name__ == "__main__":
    main()
