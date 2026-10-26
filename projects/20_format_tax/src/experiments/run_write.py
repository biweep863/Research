"""Phase W: what it costs to get a valid file of each type out of the model.

The model gets the report as plain text and one instruction per generation
route. Each reply is built into its deliverable(s) and validated. A failed
build is fed back once per retry (what an agent loop does). Time to a valid
file = generation + build time over all attempts.
"""
from __future__ import annotations

from ..analysis import builders as B
from ..analysis.metrics import chrf, fact_recall
from .common import LM, ROOT, corpus, rendered, setup, write


def doc_facts(d) -> list[str]:
    f = [str(x.value) for x in d.facts] + [x.lead for x in d.facts] + [x.entity for x in d.facts]
    return f + d.list_items + [r[0] for r in d.table_rows] + [d.equation_text.split("+")[-1].strip()]


def table_facts(d) -> list[str]:
    return [c for r in d.table_rows for c in r]


def main():
    cfg, out, models = setup("Phase W: generation cost, validity and fidelity per output format")
    docs = corpus(cfg)
    rend = rendered(cfg, docs)
    W = cfg["write"]
    docs = docs[:W.get("n_docs", len(docs))]   # writing is the slow phase; sizes cycle, so a prefix keeps the mix
    for spec in models:
        lm = LM(spec, cfg.get("device", "auto"))
        lm.prefill(lm.chat_ids([{"role": "user", "content": "hello " * 200}]))
        rows, attempts_log = [], []
        for d in docs:
            source = rend[d.doc_id]["txt"]
            ref_table = "\n".join(",".join(r) for r in [d.table_header] + d.table_rows)
            for gen in W["generations"]:
                work = B.scratch(ROOT / cfg["work_dir"], "write", spec["short"], d.doc_id, gen)
                msgs = [{"role": "user", "content": B.prompt(gen, source)}]
                gen_s = 0.0
                out_tokens = 0
                for k in range(W["max_attempts"]):
                    ids = lm.chat_ids(msgs)
                    reply, n_new, dt, hit = lm.generate(ids, W["max_new_tokens"])
                    gen_s += dt
                    out_tokens += n_new
                    payload = B.extract_payload(reply)
                    primary = B.build(B.RETRY_ON[gen], payload, work / f"try{k}")
                    attempts_log.append({"doc": d.doc_id, "gen": gen, "attempt": k, "in_tokens": len(ids),
                                         "out_tokens": n_new, "gen_s": dt, "hit_limit": hit,
                                         "valid": primary.valid, "build_s": primary.build_s,
                                         "error": primary.error[:300]})
                    if primary.valid:
                        break
                    msgs += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"That did not work: {primary.error[:500] or 'invalid'}. "
                              "Fix it and reply with the complete corrected result in one code block."}]
                (work / "final_payload.txt").write_text(payload, encoding="utf-8")
                build_total = sum(a["build_s"] for a in attempts_log if a["doc"] == d.doc_id and a["gen"] == gen)
                for deliv, g in B.DELIVERABLES.items():
                    if g != gen:
                        continue
                    b = primary if deliv == B.RETRY_ON[gen] else B.build(deliv, payload, work / deliv)
                    extra = 0.0 if deliv == B.RETRY_ON[gen] else b.build_s
                    facts = table_facts(d) if gen in B.TABLE_GENERATIONS else doc_facts(d)
                    ref = ref_table if gen in B.TABLE_GENERATIONS else source
                    rows.append({"doc": d.doc_id, "deliverable": deliv, "gen": gen, "attempts": k + 1,
                                 "valid": b.valid, "first_try_valid": attempts_log[-(k + 1)]["valid"],
                                 "out_tokens": out_tokens, "gen_s": gen_s, "build_s": build_total + extra,
                                 "total_s": gen_s + build_total + extra,
                                 "fact_recall": fact_recall(b.text, facts) if b.valid else 0.0,
                                 "chrf": chrf(b.text, ref) if b.valid else 0.0, "wrong_path": b.wrong_path,
                                 "error": b.error[:300]})
                print(spec["short"], d.doc_id, gen, f"attempts={k + 1} valid={primary.valid} "
                      f"tok={out_tokens} t={gen_s:.1f}s", flush=True)
            write(out / f"write_{spec['short']}.json",
                  {"model": spec, "max_attempts": W["max_attempts"], "rows": rows, "attempts": attempts_log})
        lm.free()


if __name__ == "__main__":
    main()
