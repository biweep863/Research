# 01 · Dataset

| Source | What | How |
|---|---|---|
| Gaia DR3 `gaiadr3.gaia_source` | positions, parallaxes, proper motions, RVs, G mag | ADQL over TAP sync, CSV, split into equal-volume distance shells; total checked against `COUNT(*)` |
| GCNS `external.gaiaedr3_gcns_main_1` (Smart et al. 2021) | membership filter + literature RVs (`adoptedrv`) | inner join on `source_id` (EDR3 = DR3 ids) |
| Literature supplement | α Cen A/B, Sirius A, Procyon A (saturated in Gaia); Luhman 16 AB, WISE 0855−0714 (too faint/red) | hard-coded in `src/data/gaia.py` with references |
| NASA Exoplanet Archive `pscomppars` | confirmed host Gaia DR3 ids | TAP sync CSV |

Cuts: `parallax > 1000/R`, `parallax_over_error >= 10`, `pmra IS NOT NULL`,
GCNS member.

Local (R = 50 pc): **40,229 objects**, 73% with RV, 602 exoplanet hosts. Star
density is flat at 0.077 pc⁻³ across 10-pc shells.

## Pitfalls found (and fixed)
- **Silent truncation.** `astroquery` async jobs returned 11–13k of 48.8k rows
  without any error. The fix is plain TAP sync with CSV, chunked, plus a count
  check that raises.
- **Spurious parallaxes.** 8,586 raw-cut rows (18%) are not GCNS members. Their
  median tangential speed is 2.6 km/s (GCNS no-RV stars: 30.6 km/s), so they
  are distant background sources. They made density rise with distance and,
  once given RVs, fell toward the Sun.
- **Missing objects matter.** Without Luhman 16 and WISE 0855 the Sun sits in a
  4-object pocket (Sun, Proxima, Barnard's, Ross 154) that opens only at 2.27 pc.

## License
Gaia data: ESA/Gaia/DPAC, CC BY-SA 3.0 IGO. NASA Exoplanet Archive: public
domain. Cached files live in `data/cache/` and are not committed.
