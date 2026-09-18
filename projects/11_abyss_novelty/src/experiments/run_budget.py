"""Experiment 1 -- the physical sensing budget: what reaches the abyss and back.
Closed forms per modality vs depth / frequency.  Writes results/budget.json."""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from src.physics import acoustics, edna, light, radio


def main():
    cfg, out_dir = load_cfg()
    b = cfg["budget"]
    depths = np.array(b["depths_m"], dtype=float)

    # Light
    L = b["light"]
    light_rows = []
    for name, kd in L["jerlov"].items():
        light_rows.append({"type": name, "kd": kd,
                           "euphotic_m": light.euphotic_depth(kd),
                           "reach_one_way_m": light.reach_depth(L["surface_irradiance_w_m2"], L["detector_floor_w_m2"], kd, False),
                           "reach_two_way_m": light.reach_depth(L["surface_irradiance_w_m2"], L["detector_floor_w_m2"], kd, True),
                           "transmission": (light.irradiance(depths, 1.0, kd) ** (2 if L["two_way"] else 1)).tolist()})
    biolum = light.bioluminescence_range(L["biolum_photons"], L["camera_threshold_photons"], L["camera_aperture_m2"], L["biolum_kd"])

    # Radio
    R = b["radio"]
    radio_rows = [{"f_hz": f, "skin_depth_m": float(radio.skin_depth(f, R["sigma_s_m"])),
                   "reach_m": float(radio.reach_depth(f, R["detect_db"], R["sigma_s_m"]))} for f in R["freqs_hz"]]

    # Acoustics
    A = b["acoustics"]
    ac_rows = []
    for name, src in A["sources"].items():
        for ss in A["sea_states"]:
            for spreading in ("spherical", "hybrid"):
                r = acoustics.detection_range_m(src["sl"], src["f_hz"], ss, A["shipping"], A["directivity_db"],
                                                A["detection_threshold_db"], None, A["max_range_km"] * 1000,
                                                spreading, A["transition_m"])
                ac_rows.append({"source": name, "sl_db": src["sl"], "f_hz": src["f_hz"], "sea_state": ss,
                                "spreading": spreading, "range_m": r})
    alpha = {str(f): float(acoustics.thorp_alpha_db_km(f)) for f in A["freqs_hz"]}
    noise = {str(ss): [float(acoustics.wenz_noise_db(f, ss, A["shipping"])) for f in A["freqs_hz"]] for ss in A["sea_states"]}
    # Range vs frequency for a fixed 180 dB source (hybrid, sea state 3)
    range_vs_f = [acoustics.detection_range_m(180.0, f, 3, A["shipping"], 0.0, A["detection_threshold_db"], None,
                                              A["max_range_km"] * 1000, "hybrid", A["transition_m"]) for f in A["freqs_hz"]]

    # eDNA
    E = b["edna"]
    edna_rows = [{"k_per_h": k, "memory_h": edna.memory_h(k),
                  "footprint_m": edna.footprint_m(E["shed_copies_per_h"], k, E["current_m_s"], E["diffusivity_m2_s"], E["detect_copies_per_l"])}
                 for k in E["decay_per_h"]]

    out = {"depths_m": depths.tolist(), "light": light_rows, "biolum_camera_range_m": biolum,
           "radio": radio_rows, "acoustics": ac_rows, "thorp_alpha_db_km": alpha, "wenz_db": noise,
           "freqs_hz": A["freqs_hz"], "range_vs_f_180db_m": range_vs_f, "edna": edna_rows}
    for r in light_rows:
        print(f"[budget] light {r['type']:3s} kd={r['kd']:.3f}  euphotic {r['euphotic_m']:.0f} m  satellite/lidar reach {r['reach_two_way_m']:.0f} m")
    print(f"[budget] bioluminescence camera range {biolum:.0f} m")
    for r in radio_rows:
        print(f"[budget] radio {r['f_hz']:>12.0f} Hz  skin {r['skin_depth_m']:.3g} m  reach {r['reach_m']:.3g} m")
    for r in ac_rows:
        if r["sea_state"] == 3:
            print(f"[budget] sound {r['source']:18s} {r['spreading']:9s} range {r['range_m']/1000:.1f} km")
    for r in edna_rows:
        print(f"[budget] eDNA k={r['k_per_h']}  memory {r['memory_h']:.0f} h  footprint {r['footprint_m']:.0f} m")
    dump(out_dir, "budget.json", out)


if __name__ == "__main__":
    main()
