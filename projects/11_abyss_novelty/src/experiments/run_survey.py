"""Experiment 5 -- how many species remain, and where to spend sensor hours.
Species accumulation, Chao1 / Good-Turing vs effort, and greedy allocation
under a satellite habitat prior of varying quality.  Writes results/survey.json."""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from src.survey.allocate import greedy_allocate, realised_species
from src.survey.community import logseries_abundances, sample
from src.survey.estimators import chao1, good_turing_discovery


def main():
    cfg, out_dir = load_cfg()
    sv = cfg["survey"]
    seed = cfg.get("seed", 0)
    rng = np.random.default_rng(seed + 7)
    S_true = sv["n_species"]
    abund = logseries_abundances(S_true, sv["logseries_alpha"], rng)

    acc = []
    for h in sv["effort_hours"]:
        obs, chao, se, gt = [], [], [], []
        for _ in range(sv["n_rep"]):
            c = sample(abund, h, sv["detect_rate_per_hour"], 1.0, rng)
            obs.append(int((c > 0).sum())); s_hat, s_se = chao1(c); chao.append(s_hat); se.append(s_se)
            gt.append(good_turing_discovery(c))
        acc.append({"hours": h, "s_obs": float(np.mean(obs)), "chao1": float(np.mean(chao)), "chao1_se": float(np.mean(se)),
                    "discovery_prob": float(np.mean(gt)), "frac_seen": float(np.mean(obs) / S_true)})
        print(f"[survey] {h:5d} h: seen {np.mean(obs):6.1f}/{S_true}  Chao1 {np.mean(chao):6.1f} +- {np.mean(se):.1f}  "
              f"P(next is new) {np.mean(gt):.3f}")

    # Allocation with a satellite habitat prior.
    hab_cfg = sv["habitats"]
    comm = {h: logseries_abundances(v["richness"], sv["logseries_alpha"], rng) for h, v in hab_cfg.items()}
    names = list(hab_cfg)
    fracs = np.array([hab_cfg[h]["area_frac"] for h in names])
    alloc = []
    for q in sv["prior_quality"]:
        gains_g, gains_u = [], []
        for rep in range(sv.get("n_rep_alloc", sv["n_rep"] // 3)):
            true_hab = list(rng.choice(names, sv["n_sites"], p=fracs / fracs.sum()))
            priors = []
            for th in true_hab:
                guess = th if rng.random() < q else rng.choice([n for n in names if n != th])
                priors.append({n: (q if n == guess else (1 - q) / (len(names) - 1)) for n in names})
            hours = greedy_allocate(priors, comm, sv["budget_hours"], sv["detect_rate_per_hour"], step_h=sv["budget_hours"] / 30)
            gains_g.append(realised_species(hours, true_hab, comm, sv["detect_rate_per_hour"], rng))
            gains_u.append(realised_species(np.full(sv["n_sites"], sv["budget_hours"] / sv["n_sites"]), true_hab, comm,
                                            sv["detect_rate_per_hour"], rng))
        alloc.append({"prior_quality": q, "species_greedy": float(np.mean(gains_g)), "species_uniform": float(np.mean(gains_u)),
                      "gain_pct": float(100 * (np.mean(gains_g) / np.mean(gains_u) - 1))})
        print(f"[survey] prior q={q}: greedy {np.mean(gains_g):.1f} vs uniform {np.mean(gains_u):.1f} species (+{alloc[-1]['gain_pct']:.0f}%)")
    dump(out_dir, "survey.json", {"n_species": S_true, "alpha": sv["logseries_alpha"], "accumulation": acc,
                                  "allocation": alloc, "habitats": hab_cfg, "budget_hours": sv["budget_hours"]})


if __name__ == "__main__":
    main()
