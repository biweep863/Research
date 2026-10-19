"""Gaia DR3 nearby-star catalogue: download once, cache, load as arrays.

Query: every gaiadr3.gaia_source row with parallax > 1000/R (inside R pc) and
parallax_over_error >= `poe_min`, with full astrometry, radial velocity where
Gaia measured one, and G magnitude. A handful of very bright nearby stars that
saturate Gaia (alpha Cen A/B, Sirius A, Procyon A) are added from Hipparcos-era
astrometry so the immediate solar neighbourhood is not artificially empty.

Contamination. A plain parallax cut admits background sources with spurious
parallaxes (crowded fields, close pairs). Inside 50 pc ~18% of the raw cut is
not in the Gaia Catalogue of Nearby Stars (GCNS; Gaia Collaboration, Smart et
al. 2021), and those rows have a median tangential speed of 2.6 km/s -- the
signature of distant, not nearby, stars. We keep only GCNS members, and use the
GCNS literature RV ("adoptedrv") where DR3 has none.

Exoplanet hosts (NASA Exoplanet Archive, pscomppars) are matched by Gaia DR3
source_id and flagged; they serve as "known-resource" refuelling nodes.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "source_id", "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmra_error", "pmdec", "pmdec_error",
    "radial_velocity", "radial_velocity_error", "phot_g_mean_mag",
]

# Objects missing from Gaia DR3 astrometry. Bright stars saturate Gaia:
# astrometry from van Leeuwen 2007 (Hipparcos) / Pourbaix & Boffin 2016 for
# alpha Cen, RVs from SIMBAD. The two nearest brown dwarfs are too faint/red:
# Luhman 16 AB (Garcia et al. 2017; RV Faherty et al. 2014) and WISE 0855-0714
# (Luhman & Esplin 2016; no RV -> filled like any other missing RV).
# source_id < 0 marks a supplement row. G ~ V for the stars; brown dwarfs get
# G = 25 (effectively dark: they never act as sail-braking stars).
SUPPLEMENT = [
    # name,          id, ra,        dec,       plx,     pmra,     pmdec,    rv,     G
    ("alpha Cen A",  -1, 219.9021, -60.8340, 747.1,   -3679.25,  473.67, -21.4,  -0.01),
    ("alpha Cen B",  -2, 219.8961, -60.8375, 747.1,   -3614.39,  802.98, -18.6,   1.33),
    ("Sirius A",     -3, 101.2872, -16.7161, 379.21,   -546.01, -1223.07, -5.5,  -1.46),
    ("Procyon A",    -4, 114.8255,   5.2250, 284.56,   -714.59, -1036.80, -3.2,   0.37),
    ("Luhman 16 AB", -5, 162.3292, -53.3194, 501.557, -2762.16,  354.0,   20.1,  25.0),
    ("WISE 0855-07", -6, 133.7861,  -7.2443, 439.0,   -8123.7,   673.3,  np.nan, 25.0),
]


def _cache_path(cache_dir: Path, radius_pc: float, poe_min: float) -> Path:
    return cache_dir / f"gaia_dr3_r{radius_pc:g}pc_poe{poe_min:g}.parquet"


GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def _tap_csv(query: str) -> pd.DataFrame:
    import requests
    r = requests.post(GAIA_TAP, data={"REQUEST": "doQuery", "LANG": "ADQL",
                                      "FORMAT": "csv", "QUERY": query}, timeout=900)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def fetch_gaia(radius_pc: float, poe_min: float, cache_dir: Path,
               rows_per_chunk: int = 60_000) -> pd.DataFrame:
    """Download (or load cached) Gaia DR3 stars within `radius_pc`.

    Plain TAP sync requests returning CSV, split into equal-volume distance
    shells so each request stays small; the total is checked against a
    COUNT(*) because the archive has been seen to truncate large results
    silently (astroquery's async path lost ~75% of rows in testing)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, radius_pc, poe_min)
    if path.exists():
        return pd.read_parquet(path)
    base = f"parallax_over_error >= {poe_min} AND pmra IS NOT NULL"
    n_expected = int(_tap_csv(
        f"SELECT COUNT(*) AS n FROM gaiadr3.gaia_source "
        f"WHERE parallax > {1000.0 / radius_pc} AND {base}")["n"][0])
    n_chunks = max(1, int(np.ceil(n_expected / rows_per_chunk)))
    edges = radius_pc * (np.arange(n_chunks + 1) / n_chunks) ** (1 / 3)   # equal volume
    print(f"[gaia] querying Gaia archive (R={radius_pc} pc, poe>={poe_min}, "
          f"{n_expected} rows, {n_chunks} chunks) ...")
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        plx_cut = f"parallax > {1000.0 / hi}" + (f" AND parallax <= {1000.0 / lo}" if lo > 0 else "")
        parts.append(_tap_csv(f"SELECT {', '.join(COLUMNS)} FROM gaiadr3.gaia_source "
                              f"WHERE {plx_cut} AND {base}"))
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[COLUMNS]
    if len(df) != n_expected:
        raise RuntimeError(f"Gaia download incomplete: {len(df)} of {n_expected} rows")
    df.to_parquet(path)
    print(f"[gaia] cached {len(df)} stars -> {path}")
    return df


def fetch_gcns(radius_pc: float, cache_dir: Path) -> pd.DataFrame:
    """GCNS members (EDR3 source_id == DR3 source_id) inside radius_pc."""
    path = cache_dir / f"gcns_r{radius_pc:g}pc.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = _tap_csv("SELECT source_id, gcns_prob, adoptedrv, adoptedrv_error "
                  f"FROM external.gaiaedr3_gcns_main_1 WHERE parallax > {1000.0 / radius_pc}")
    df.to_parquet(path)
    print(f"[gaia] cached {len(df)} GCNS members -> {path}")
    return df


def fetch_exoplanet_hosts(cache_dir: Path) -> np.ndarray:
    """Gaia DR3 source_ids of confirmed exoplanet hosts (cached CSV)."""
    path = cache_dir / "exoplanet_hosts.csv"
    if not path.exists():
        import requests
        q = ("select distinct gaia_dr3_id from pscomppars "
             "where gaia_dr3_id is not null")
        r = requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                         params={"query": q, "format": "csv"}, timeout=120)
        r.raise_for_status()
        path.write_text(r.text)
    df = pd.read_csv(io.StringIO(path.read_text()))
    ids = df["gaia_dr3_id"].astype(str).str.extract(r"(\d+)$")[0].dropna()
    return ids.astype(np.int64).to_numpy()


def supplement_frame() -> pd.DataFrame:
    rows = []
    for name, sid, ra, dec, plx, pmra, pmdec, rv, g in SUPPLEMENT:
        rows.append(dict(source_id=sid, ra=ra, dec=dec, parallax=plx,
                         parallax_error=0.5, pmra=pmra, pmra_error=1.0,
                         pmdec=pmdec, pmdec_error=1.0, radial_velocity=rv,
                         radial_velocity_error=0.5 if np.isfinite(rv) else np.nan,
                         phot_g_mean_mag=g))
    return pd.DataFrame(rows)


def load_catalog(radius_pc: float, poe_min: float, cache_dir: str | Path,
                 with_supplement: bool = True) -> dict:
    """Return the catalogue as a dict of numpy arrays (Sun NOT included)."""
    cache_dir = Path(cache_dir)
    df = fetch_gaia(radius_pc, poe_min, cache_dir)
    gc = fetch_gcns(radius_pc, cache_dir)
    df = df.merge(gc, on="source_id", how="inner")
    lit = df["radial_velocity"].isna() & df["adoptedrv"].notna()
    df.loc[lit, "radial_velocity"] = df.loc[lit, "adoptedrv"]
    df.loc[lit, "radial_velocity_error"] = df.loc[lit, "adoptedrv_error"]
    df = df[COLUMNS]
    if with_supplement:
        sup = supplement_frame()
        sup = sup[sup["parallax"] > 1000.0 / radius_pc]
        df = pd.concat([df, sup], ignore_index=True)
    hosts = set(fetch_exoplanet_hosts(cache_dir).tolist())
    cat = {c: df[c].to_numpy(dtype=np.float64) for c in COLUMNS if c != "source_id"}
    cat["source_id"] = df["source_id"].to_numpy(dtype=np.int64)
    cat["has_rv"] = np.isfinite(cat["radial_velocity"])
    cat["is_host"] = np.array([s in hosts for s in cat["source_id"]])
    return cat
