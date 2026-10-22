"""Turn an OSM extract into a corridor inventory.

Output (both directions, chainage in metres from the Garza Sada end):
  * cross-section: express and lateral lanes per 100 m bin and direction
  * approaches: every (way, node) where a street can *deliver* vehicles onto a
    corridor carriageway (oneway respected; departures kept separately)
  * accesses: approaches of the same street clustered within `access_cluster_m`
  * junctions: clusters of major approaches; they split the corridor into segments
  * signals: traffic_signals nodes on the corridor ways

Direction "WB" travels with increasing chainage (towards San Pedro), "EB"
against it (towards Garza Sada / Monterrey).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

BIN_M = 100.0


class Projector:
    """Local equirectangular projection (metres) and chainage along a spine."""

    def __init__(self, spine_lonlat):
        sp = np.asarray(spine_lonlat, float)
        self.lon0, self.lat0 = sp[:, 0].mean(), sp[:, 1].mean()
        self.kx = 111320.0 * math.cos(math.radians(self.lat0))
        self.ky = 110540.0
        # Chainage 0 at the Garza Sada (east) end: spine is listed east -> west.
        self.spine = self.xy(sp[:, 0], sp[:, 1])
        seg = np.diff(self.spine, axis=0)
        self.seg_len = np.sqrt((seg ** 2).sum(1))
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])

    def xy(self, lon, lat):
        return np.stack([(np.asarray(lon) - self.lon0) * self.kx,
                         (np.asarray(lat) - self.lat0) * self.ky], axis=-1)

    def chain(self, p):
        """(chainage, offset) of points p (n, 2) onto the spine polyline."""
        p = np.atleast_2d(p)
        a = self.spine[:-1][None]
        ab = (self.spine[1:] - self.spine[:-1])[None]
        t = np.clip(((p[:, None] - a) * ab).sum(-1) / (ab ** 2).sum(-1), 0, 1)
        proj = a + t[..., None] * ab
        d = np.sqrt(((proj - p[:, None]) ** 2).sum(-1))
        k = d.argmin(1)
        idx = np.arange(len(p))
        return self.cum[k] + t[idx, k] * self.seg_len[k], d[idx, k]


def _lanes(tags: dict, cls_default: int, direction: str | None = None) -> tuple[int, bool]:
    """Lanes in one direction of travel; (value, came_from_tag)."""
    oneway = tags.get("oneway") in ("yes", "-1", "true", "1")
    if direction and f"lanes:{direction}" in tags:
        try:
            return max(1, int(float(tags[f"lanes:{direction}"]))), True
        except ValueError:
            pass
    if "lanes" in tags:
        try:
            v = int(float(tags["lanes"].split(";")[0]))
            return (max(1, v) if oneway else max(1, v // 2)), True
        except ValueError:
            pass
    return cls_default, False


def _cls(hw: str) -> str:
    return "link" if hw.endswith("_link") else hw


def demand_class(hw: str) -> str:
    if hw.startswith(("motorway", "trunk", "primary", "secondary")):
        return "major"
    if hw.startswith("tertiary"):
        return "tertiary"
    if hw == "service":
        return "service"
    return "minor"


def build_corridor(osm: dict, ccfg: dict, spine_lonlat) -> dict:
    nodes, ways = osm["nodes"], osm["ways"]
    pj = Projector(spine_lonlat)
    nid = [k for k in nodes]
    pos = {k: i for i, k in enumerate(nid)}
    lonlat = np.array([[nodes[k][1], nodes[k][0]] for k in nid])
    ch, off = pj.chain(pj.xy(lonlat[:, 0], lonlat[:, 1]))

    name_re = re.compile("Lázaro Cárdenas")
    excl = re.compile(ccfg["exclude_name_regex"])
    corr_hw = set(ccfg["corridor_highways"])

    # ---- corridor ways -------------------------------------------------------
    corridor = {}
    for wid, (nds, tags) in ways.items():
        nm = tags.get("name", "")
        if not (name_re.search(nm) and not excl.search(nm) and tags.get("highway") in corr_hw):
            continue
        idx = [pos[n] for n in nds if n in pos]
        if len(idx) < 2 or np.median(off[idx]) > ccfg["max_offset_m"]:
            continue
        c = ch[idx]
        ow = tags.get("oneway")
        if ow == "-1":
            c = c[::-1]
        if ow in ("yes", "-1", "true", "1"):
            dirs = ["WB"] if c[-1] > c[0] else ["EB"]
        else:
            dirs = ["WB", "EB"]
        hw = tags["highway"]
        kind = "link" if hw.endswith("_link") else ("express" if hw in ("trunk", "primary") else "lateral")
        corridor[wid] = {"dirs": dirs, "kind": kind, "c0": float(c.min()), "c1": float(c.max()),
                         "lanes": _lanes(tags, ccfg["default_lanes"].get(_cls(hw), 2))[0],
                         "lanes_tagged": "lanes" in tags, "highway": hw, "name": tags.get("name", "")}

    express = [w for w in corridor.values() if w["kind"] == "express"]
    c_start = min(w["c0"] for w in express)
    c_end = max(w["c1"] for w in express)
    bins = np.arange(c_start, c_end, BIN_M)
    lanes = {d: {"express": np.zeros(len(bins)), "lateral": np.zeros(len(bins))} for d in ("WB", "EB")}
    for w in corridor.values():
        if w["kind"] == "link":
            continue
        cover = (bins + BIN_M / 2 >= w["c0"]) & (bins + BIN_M / 2 <= w["c1"])
        for d in w["dirs"]:
            lanes[d][w["kind"]][cover] += w["lanes"]

    # Corridor node -> set of (direction, kind)
    cnode = defaultdict(set)
    for wid, w in corridor.items():
        for n in ways[wid][0]:
            for d in w["dirs"]:
                cnode[n].add((d, w["kind"]))

    # Signals on corridor nodes
    signals = []
    for n, dk in cnode.items():
        if n in nodes and nodes[n][2].get("highway") == "traffic_signals":
            for d in {x[0] for x in dk}:
                signals.append({"node": n, "dir": d, "chainage": float(ch[pos[n]]),
                                "on": sorted({x[1] for x in dk if x[0] == d})})

    # ---- approaches and departures ------------------------------------------
    feeder_hw = set(ccfg["feeder_highways"])
    approaches, departures, internal = [], [], 0
    node_ways = defaultdict(list)
    for wid, (nds, tags) in ways.items():
        if wid in corridor or tags.get("highway") not in feeder_hw:
            continue
        for j, n in enumerate(nds):
            if n in cnode:
                node_ways[n].append((wid, j))

    internal_ways, link_feeds = _internal_ways(ways, corridor, cnode, feeder_hw)
    for wid, (nds, tags) in ways.items():
        if wid in corridor or tags.get("highway") not in feeder_hw:
            continue
        touch = [j for j, n in enumerate(nds) if n in cnode]
        if not touch:
            continue
        hw = tags["highway"]
        if tags.get("area") == "yes" or tags.get("access") in ("no",):
            continue
        if wid in internal_ways:
            internal += 1
            continue
        ow = tags.get("oneway")
        fwd_ok = ow != "-1"
        bwd_ok = ow not in ("yes", "true", "1")
        if ow == "-1":
            fwd_ok, bwd_ok = False, True
        default = ccfg["default_lanes"].get(_cls(hw), 1)
        for j in touch:
            n = nds[j]
            dks = cnode[n]
            dirs = sorted({d for d, _ in dks})
            side = "express" if any(k == "express" for _, k in dks) else "lateral"
            others = [w for w, _ in node_ways[n] if w != wid]
            crossing = (0 < j < len(nds) - 1) or bool(others)
            for nb, inbound_ok, outbound_ok, ldir in (
                    (j - 1, fwd_ok, bwd_ok, "forward"), (j + 1, bwd_ok, fwd_ok, "backward")):
                if nb < 0 or nb >= len(nds) or nds[nb] in cnode:
                    continue
                ln, tagged = _lanes(tags, default, ldir if not ow else None)
                ln_out = _lanes(tags, default, ("backward" if ldir == "forward" else "forward") if not ow else None)[0]
                rec = {"way": wid, "node": n, "name": tags.get("name", ""), "highway": hw,
                       "dclass": link_feeds.get(wid, demand_class(hw)), "lanes": ln, "lanes_tagged": tagged,
                       "chainage": float(ch[pos[n]]), "dirs": dirs, "side": side,
                       "kind": "crossing" if crossing else "terminal",
                       "service": tags.get("service", "")}
                if inbound_ok:
                    approaches.append(rec)
                if outbound_ok:
                    departures.append(dict(rec, lanes=ln_out))

    # Retornos: internal chains that take vehicles from one carriageway to the
    # other.  They are an approach of the receiving direction and an exit of
    # the giving one.
    retornos = _retornos(ways, internal_ways, cnode, ch, pos, ccfg["default_lanes"].get("link", 1))
    for r in retornos:
        base = {"way": r["way"], "name": "", "highway": "retorno", "dclass": "retorno", "lanes": r["lanes"],
                "lanes_tagged": r["lanes_tagged"], "dirs": [r["to"]], "side": r["side_to"],
                "kind": "terminal", "service": "", "uturn": True}
        approaches.append(dict(base, node=r["node_to"], chainage=r["ch_to"]))
        departures.append(dict(base, node=r["node_from"], chainage=r["ch_from"], dirs=[r["from"]],
                               side=r["side_from"]))

    sig_ch = np.array([s["chainage"] for s in signals]) if signals else np.zeros(0)
    for a in approaches + departures:
        a["signal"] = bool(sig_ch.size and np.abs(sig_ch - a["chainage"]).min() < 40)
        # One carriageway per approach: if the node is shared, split by side.
        a["dir"] = a["dirs"][0] if len(a["dirs"]) == 1 else "both"

    inside = lambda a: c_start - 50 <= a["chainage"] <= c_end + 50  # noqa: E731
    approaches = [a for a in approaches if inside(a)]
    departures = [a for a in departures if inside(a)]

    accesses = _cluster_accesses(approaches, ccfg["access_cluster_m"])
    junctions = _junctions(approaches, set(ccfg["major_classes"]), ccfg["junction_cluster_m"],
                           ccfg.get("junction_labels", []))
    return {
        "chainage_start_m": float(c_start), "chainage_end_m": float(c_end),
        "length_km": float((c_end - c_start) / 1000), "bins_m": bins,
        "lanes": lanes, "signals": signals, "approaches": approaches,
        "departures": departures, "accesses": accesses, "junctions": junctions,
        "internal_links": internal,
        "retornos": len(retornos),
        "corridor_ways": len(corridor),
        "corridor_lanes_tag_coverage": float(np.mean([w["lanes_tagged"] for w in corridor.values()])),
    }


def _retornos(ways, internal, cnode, ch, pos, default_lanes) -> list[dict]:
    """U-turns / median crossings among the internal chains: an origin node on
    one direction's carriageway and a destination node on the other's."""
    by_node = defaultdict(list)
    for w in internal:
        for n in ways[w][0]:
            if n not in cnode:
                by_node[n].append(w)
    seen, out = set(), []
    for w0 in internal:
        if w0 in seen:
            continue
        comp, stack = {w0}, [w0]
        while stack:
            w = stack.pop()
            for n in ways[w][0]:
                for o in by_node.get(n, []):
                    if o not in comp:
                        comp.add(o)
                        stack.append(o)
        seen |= comp
        origins, dests = [], []
        for w in comp:
            nds, t = ways[w]
            ow = t.get("oneway")
            seq = nds[::-1] if ow == "-1" else nds
            two_way = ow not in ("yes", "true", "1", "-1")
            for end, lst in ((seq[0], origins), (seq[-1], dests)):
                if end in cnode:
                    lst.append((end, w))
                    if two_way:
                        (dests if lst is origins else origins).append((end, w))
        for nd, wd in dests:
            for no, wo in origins:
                d_to = {d for d, _ in cnode[nd]}
                d_from = {d for d, _ in cnode[no]}
                if len(d_to) == 1 and len(d_from) == 1 and d_to != d_from:
                    t = ways[wd][1]
                    try:
                        ln, tagged = max(1, int(float(t["lanes"]))), True
                    except (KeyError, ValueError):
                        ln, tagged = default_lanes, False
                    side = lambda n: "express" if any(k == "express" for _, k in cnode[n]) else "lateral"  # noqa: E731
                    out.append({"way": wd, "node_to": nd, "node_from": no, "to": d_to.pop(),
                                "from": d_from.pop(), "ch_to": float(ch[pos[nd]]), "ch_from": float(ch[pos[no]]),
                                "lanes": ln, "lanes_tagged": tagged, "side_to": side(nd), "side_from": side(no)})
                    break
    # One record per (destination node, direction).
    uniq = {}
    for r in out:
        uniq.setdefault((r["node_to"], r["to"]), r)
    return list(uniq.values())


def _internal_ways(ways, corridor, cnode, feeder_hw) -> tuple[set, dict]:
    """Links / unnamed non-service ways whose connected chain only joins the
    corridor to itself (express <-> lateral transfers, U-turns, median
    crossings).  They move vehicles within the corridor, they do not feed it.

    For the remaining chains, the demand class is that of the most important
    road the chain reaches (a ramp that only reaches a parking aisle is a
    service access, not a major feeder)."""
    cand = {w for w, (nds, t) in ways.items()
            if w not in corridor and t.get("highway") in feeder_hw and t.get("highway") != "service"
            and (t["highway"].endswith("_link") or not t.get("name"))}
    by_node = defaultdict(list)
    for w, (nds, _) in ways.items():
        if w in corridor or ways[w][1].get("highway") not in feeder_hw:
            continue
        for n in nds:
            by_node[n].append(w)
    seen, internal, feeds = set(), set(), {}
    order = ["major", "tertiary", "minor", "service"]
    for w0 in cand:
        if w0 in seen:
            continue
        comp, stack = {w0}, [w0]
        while stack:
            w = stack.pop()
            for n in ways[w][0]:
                for o in by_node[n]:
                    if o in cand and o not in comp:
                        comp.add(o)
                        stack.append(o)
        seen |= comp
        nodes_c = {n for w in comp for n in ways[w][0]}
        touches = {n for n in nodes_c if n in cnode}
        outside = {o for n in nodes_c if n not in cnode for o in by_node[n] if o not in comp}
        if len(touches) >= 2 and not outside:
            internal |= comp
        elif outside:
            best = min((demand_class(ways[o][1]["highway"]) for o in outside), key=order.index)
            for w in comp:
                feeds[w] = best
    return internal, feeds


def _cluster_accesses(approaches, radius):
    groups = []
    for a in sorted(approaches, key=lambda a: a["chainage"]):
        key = (a["dir"], a["name"] or a["way"])
        for g in groups:
            if g["key"] == key and abs(g["chainage"] - a["chainage"]) <= radius:
                g["members"].append(a)
                g["lanes"] += a["lanes"]
                break
        else:
            groups.append({"key": key, "chainage": a["chainage"], "members": [a], "lanes": a["lanes"]})
    return [{"dir": g["key"][0], "name": g["members"][0]["name"], "chainage": g["chainage"],
             "lanes": g["lanes"], "n_approaches": len(g["members"]),
             "dclass": g["members"][0]["dclass"]} for g in groups]


RANK = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4,
        "unclassified": 5, "residential": 6}


def _is_major(a, major) -> bool:
    return a["highway"] in major or (a["highway"] == "tertiary" and a["lanes"] >= 2)


def _junctions(approaches, major, radius, labels=()):
    pts = sorted([a for a in approaches if _is_major(a, major)], key=lambda a: a["chainage"])
    js = []
    for a in pts:
        if js and a["chainage"] - js[-1]["c1"] <= radius:
            js[-1]["c1"] = a["chainage"]
            js[-1]["n_major"] += 1
        else:
            js.append({"c0": a["chainage"], "c1": a["chainage"], "n_major": 1})
    named = [a for a in approaches if a["name"] and a["highway"] in RANK]
    for j in js:
        j["chainage"] = (j["c0"] + j["c1"]) / 2
        near = [a for a in named if j["c0"] - radius <= a["chainage"] <= j["c1"] + radius]
        near.sort(key=lambda a: (RANK[a["highway"]], -a["lanes"], abs(a["chainage"] - j["chainage"])))
        j["name"] = near[0]["name"] if near else "(unnamed)"
        for lab in labels:
            if lab["c0"] <= j["chainage"] <= lab["c1"]:
                j["name"] = lab["name"]
    return js
