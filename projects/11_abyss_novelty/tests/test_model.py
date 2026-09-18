"""Unit tests: physics closed forms, mel features, synthetic species, novelty
scores, clustering, richness estimators and allocation.  Seconds, no downloads."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

from src.physics import acoustics, edna, light, radio                      # noqa: E402
from src.novelty.features import log_mel, mel_filterbank, to_uint8, from_uint8  # noqa: E402
from src.sim.species import FAMILIES, make_species, synth_call             # noqa: E402
from src.sim.dataset import make_clips, make_species_set                   # noqa: E402
from src.novelty.models import train_classifier, logits_and_embed, fit_pca_logistic  # noqa: E402
from src.novelty import scores as S                                        # noqa: E402
from src.novelty.cluster import cluster_novel, ari                          # noqa: E402
from src.survey.community import logseries_abundances, sample              # noqa: E402
from src.survey.estimators import chao1, good_turing_discovery             # noqa: E402
from src.survey.allocate import expected_species, greedy_allocate, realised_species  # noqa: E402

CFG = yaml.safe_load(open(HERE / "configs" / "local_small.yaml"))
AUDIO = CFG["audio"]


def test_light_closed_forms():
    assert abs(light.euphotic_depth(0.046) - np.log(100) / 0.046) < 1e-9
    z1 = light.reach_depth(100, 1e-9, 0.03, two_way=False)
    assert abs(light.reach_depth(100, 1e-9, 0.03, two_way=True) - z1 / 2) < 1e-9
    assert light.reach_depth(100, 1e-9, 0.03, two_way=True) < 1000     # nothing optical reaches 1 km
    assert 5 < light.bioluminescence_range(1e10, 100, 1e-3, 0.05) < 200


def test_radio_skin_depth():
    d = radio.skin_depth([1e9, 1e6, 100.0])
    assert abs(d[0] - 7.96e-3) < 1e-4 and abs(d[1] - 0.2516) < 1e-3 and abs(d[2] - 25.16) < 0.1
    assert radio.reach_depth(1e9, 60.0) < 0.1


def test_acoustics_monotone_and_sonar_equation():
    a = acoustics.thorp_alpha_db_km([10, 100, 1000, 10000, 100000])
    assert np.all(np.diff(a) > 0)
    r = acoustics.detection_range_m(180, 50, spreading="hybrid")
    excess = 180 - (acoustics.wenz_noise_db(50) + 10 * np.log10(0.23 * 50)) - 10
    assert abs(acoustics.transmission_loss_db(r, 50, "hybrid") - excess) < 0.2
    assert acoustics.detection_range_m(100, 1000) < 100      # quiet soft body: tens of metres at most
    assert acoustics.detection_range_m(180, 50, spreading="hybrid") > acoustics.detection_range_m(180, 50)


def test_edna_budget():
    assert edna.memory_h(0.05) == 20.0
    f1 = edna.footprint_m(1e8, 0.02, 0.05, 1.0, 0.01)
    f2 = edna.footprint_m(1e8, 0.10, 0.05, 1.0, 0.01)
    assert f1 > f2 > 0


def test_mel_features():
    fb = mel_filterbank(16000, 512, 64, 50, 7500)
    assert fb.shape == (64, 257) and np.allclose(fb.sum(axis=1), 1.0)
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 1000 * np.arange(32000) / 16000) + 0.01 * rng.normal(size=32000)
    M = log_mel(x, 16000, 64, 96, 512, 50, 7500)
    assert M.shape == (64, 96) and abs(M.mean()) < 1e-3
    peak_bin = int(np.argmax(M.mean(axis=1)))
    assert 15 < peak_bin < 45                                   # 1 kHz sits mid-band
    U = to_uint8(M); inside = np.abs(M) < 4
    assert np.abs(from_uint8(U) - M)[inside].max() < 0.05


def test_species_families_differ():
    rng = np.random.default_rng(1)
    calls = {}
    for fam in FAMILIES:
        sp = make_species(0, fam, rng)
        c = synth_call(sp, 16000, 2.0, rng)
        assert np.isfinite(c).all() and np.abs(c).max() > 0
        calls[fam] = log_mel(c + 1e-3 * rng.normal(size=len(c)), 16000, 64, 96, 512, 50, 7500)
    mats = np.stack([calls[f].ravel() for f in FAMILIES])
    C = np.corrcoef(mats)
    assert C[np.triu_indices(4, 1)].max() < 0.9


def _small():
    rng = np.random.default_rng(2)
    sp = make_species_set(6, 3, rng)
    cs = make_clips(sp, 6, 16, AUDIO, CFG["sim"], rng)
    tr = (np.arange(len(cs.y)) % 4 != 0) & cs.known
    return cs, tr, ~tr


def test_novelty_scores_beat_chance():
    cs, tr, te = _small()
    m = train_classifier(cs.X[tr], cs.y[tr], 6, epochs=15, batch=16, lr=3e-3)
    lg, z = logits_and_embed(m, cs.X[te]); _, ztr = logits_and_embed(m, cs.X[tr])
    unk = ~cs.known[te]
    assert np.mean(lg.argmax(1)[~unk] == cs.y[te][~unk]) > 0.5
    assert S.metrics(S.knn(ztr, z), unk)["auroc"] > 0.7
    assert S.metrics(S.mahalanobis(ztr, cs.y[tr], z), unk)["auroc"] > 0.7
    pipe = fit_pca_logistic(cs.X[tr], cs.y[tr], dim=20)
    p = pipe.predict_proba(cs.X[te].reshape(len(te.nonzero()[0]), -1))
    assert 0 <= S.metrics(1 - p.max(1), unk)["auroc"] <= 1
    m_ = S.metrics(np.r_[np.ones(unk.sum()), np.zeros((~unk).sum())], np.r_[np.ones(unk.sum(), bool), np.zeros((~unk).sum(), bool)])
    assert m_["auroc"] == 1.0 and m_["fpr95"] == 0.0


def test_clustering_recovers_separated_groups():
    rng = np.random.default_rng(3)
    centres = rng.normal(size=(4, 16)) * 5
    z = np.concatenate([c + 0.3 * rng.normal(size=(20, 16)) for c in centres])
    truth = np.repeat(np.arange(4), 20)
    lab, k, sil = cluster_novel(z, (2, 8))
    assert k == 4 and ari(truth, lab) > 0.95


def test_chao1_and_good_turing():
    rng = np.random.default_rng(4)
    ab = logseries_abundances(200, 40.0, rng)
    assert abs(ab.sum() - 1) < 1e-9 and (ab > 0).all()
    c = sample(ab, 3000, 0.02, 1.0, rng)
    s_hat, se = chao1(c)
    assert (c > 0).sum() <= s_hat <= 400 and se >= 0
    assert 0 <= good_turing_discovery(c) <= 1
    assert chao1(np.array([5, 3, 1, 1, 2]))[0] == 5 + 4 / 2
    assert good_turing_discovery(np.array([5, 3, 1, 1, 2])) == 2 / 12


def test_allocation_beats_uniform_with_good_prior():
    rng = np.random.default_rng(5)
    comm = {"rich": logseries_abundances(300, 40.0, rng), "poor": logseries_abundances(80, 40.0, rng)}
    hab = ["rich"] * 4 + ["poor"] * 12
    priors = [{h: 1.0} for h in hab]                       # perfect prior
    hours = greedy_allocate(priors, comm, 160.0, 0.02, step_h=10.0)
    assert abs(hours.sum() - 160.0) < 1e-9 and hours[:4].mean() > hours[4:].mean()
    e_g = np.mean([realised_species(hours, hab, comm, 0.02, rng) for _ in range(20)])
    e_u = np.mean([realised_species(np.full(16, 10.0), hab, comm, 0.02, rng) for _ in range(20)])
    assert e_g > e_u
    assert expected_species(comm["rich"], 1000, 0.02) > expected_species(comm["rich"], 10, 0.02)
