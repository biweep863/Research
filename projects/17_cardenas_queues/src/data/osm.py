"""Download the street network around Av. Lázaro Cárdenas from OpenStreetMap.

Two sources, same output format ({"nodes": {id: [lat, lon, tags]},
"ways": {id: [node_ids, tags]}, "source": ..., "fetched": ...}):

1. Overpass API (one query; often busy), tried first with a short timeout.
2. The main OSM API /api/0.6/map, tiled into 0.01-degree boxes along the
   corridor spine (each box is far below the API's area and node limits).

The result is cached in data/osm_raw.json; delete it to refetch.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

UA = "lc-queues-research/0.1 (academic project; OSM data under ODbL)"


def _get(url: str, timeout: float, data: bytes | None = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _overpass(cfg: dict) -> dict:
    s, w, n, e = cfg["bbox"]
    name = cfg["name_regex"]
    q = (f'[out:json][timeout:90];way["highway"]["name"~"{name}"]({s},{w},{n},{e})->.lc;'
         f'node(w.lc)->.n;way(bn.n)["highway"]->.x;(.lc;.x;)->.all;.all out body;>;out body;')
    last = None
    for url in cfg["overpass_urls"]:
        try:
            raw = _get(url, cfg["overpass_timeout_s"], urllib.parse.urlencode({"data": q}).encode())
            d = json.loads(raw)
            nodes = {str(el["id"]): [el["lat"], el["lon"], el.get("tags", {})]
                     for el in d["elements"] if el["type"] == "node"}
            ways = {str(el["id"]): [[str(x) for x in el["nodes"]], el.get("tags", {})]
                    for el in d["elements"] if el["type"] == "way"}
            if ways:
                return {"nodes": nodes, "ways": ways, "source": url}
        except Exception as exc:  # noqa: BLE001 - any failure falls through to the next source
            last = exc
    raise RuntimeError(f"overpass failed: {last}")


def spine_tiles(spine, step: float = 0.01, buffer: float = 0.006):
    """0.01-degree tiles whose centre lies within `buffer` degrees of the spine."""
    spine = np.asarray(spine, float)            # (k, 2) lon, lat
    lo = spine.min(0) - buffer
    hi = spine.max(0) + buffer
    tiles = []
    for x0 in np.arange(np.floor(lo[0] / step) * step, hi[0], step):
        for y0 in np.arange(np.floor(lo[1] / step) * step, hi[1], step):
            c = np.array([x0 + step / 2, y0 + step / 2])
            if _dist_to_polyline(c, spine) <= buffer + step / 2:
                tiles.append((round(x0, 4), round(y0, 4), round(x0 + step, 4), round(y0 + step, 4)))
    return tiles


def _dist_to_polyline(p, line) -> float:
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-18), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.sqrt(((proj - p) ** 2).sum(1)).min())


def _osm_api(cfg: dict, log=print) -> dict:
    nodes, ways = {}, {}
    tiles = spine_tiles(cfg["spine_lonlat"])
    for i, (w, s, e, n) in enumerate(tiles):
        url = f"{cfg['api_url']}?bbox={w},{s},{e},{n}"
        root = ET.fromstring(_get(url, 60))
        for el in root.iter("node"):
            tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
            nodes[el.get("id")] = [float(el.get("lat")), float(el.get("lon")), tags]
        for el in root.iter("way"):
            tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
            if "highway" in tags:
                ways[el.get("id")] = [[nd.get("ref") for nd in el.iter("nd")], tags]
        log(f"[osm] tile {i + 1}/{len(tiles)}: {len(ways)} highway ways so far")
        time.sleep(cfg.get("api_sleep_s", 1.0))
    used = {nid for nds, _ in ways.values() for nid in nds}
    return {"nodes": {k: v for k, v in nodes.items() if k in used}, "ways": ways,
            "source": cfg["api_url"], "tiles": len(tiles)}


def load_osm(cfg: dict, cache: Path, refresh: bool = False, log=print) -> dict | None:
    """Cached OSM extract, or None if every source fails (caller falls back to synthetic)."""
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    out = None
    for fetch in (_overpass, _osm_api):
        try:
            out = fetch(cfg) if fetch is _overpass else fetch(cfg, log)
            break
        except Exception as exc:  # noqa: BLE001
            log(f"[osm] {fetch.__name__} failed: {exc}")
    if out is None:
        return None
    out["fetched"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    log(f"[osm] {len(out['ways'])} ways, {len(out['nodes'])} nodes from {out['source']} -> {cache}")
    return out
