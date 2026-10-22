# 05 · Decisions

1. **OSM over the LLM's numbers.** The motivating figures had no source. Rejected:
   using them as inputs. OSM is open, versioned and checkable; its totals turned out
   to agree, which is itself a result.
2. **Tiled OSM API fallback.** Overpass was saturated. Rejected: hard-failing or
   relying on a single mirror. 25 small bbox requests with 1 s spacing respect the
   API usage policy.
3. **Internal links by connected components.** A pairwise "both ends on corridor"
   rule missed multi-way ramp chains (17–23k veh/h phantom demand). Rejected:
   dropping all unnamed links (would delete real ramps from Garza Sada).
4. **Links inherit the class of the best road they reach.** A ramp that leads to
   a parking aisle is a service access. Rejected: rating every `trunk_link` as major.
5. **Per-lane rates by class, not counts.** No public counts exist. Rejected: a
   single "AADT" scaling (hides the class structure that is the point of the study).
   Ranges are wide and swept; field counts can override (`counts.py`).
6. **Exits balanced to side inflow.** Without exits flow only accumulates and the
   last cell is always the bottleneck. Rejected: OD matrices (no data to fit them).
7. **Inferred signals.** OSM has 4. Rejected: trusting OSM (no signals at Diego
   Rivera etc. is not credible) and signalising every access (over-penalises).
8. **Aggregated express + frontage cross-section.** Rejected: two parallel CTM links
   with transfer ramps — more parameters (transfer splits) with no data to set them.
   Cost: the merge at 4.4 km is a lane drop rather than a weave.
9. **Partial FIFO diverge (φ) as a swept parameter.** Strict FIFO made delay 5× the
   single-bottleneck fluid queue; non-FIFO is optimistic for exits served from the
   frontage road. Rejected: picking one; φ is reported as the dominant uncertainty.
10. **Capacity drop added (0–15 %).** Without it ALINEA cannot help by construction.
    Rejected: leaving it out (would bias against metering).
11. **Consolidation via collectors.** v1 dumped closed-access demand onto the next
    access with unchanged capacity (+213 % delay, a straw man). v2 gives each
    segment one 1600 veh/h merge lane.
12. **Hit rate as primary score.** Active-bottleneck shares are mostly zero, so
    Spearman is dominated by ties. Spearman is still reported.
13. **Decomposition predictors.** The cumulative lane ratio scored 70 % — added
    feeder-only (0 %) and capacity-only (100 %) predictors to show the success
    came from the denominator (geometry).
14. **Gap-filled robustness run.** The headline depends on one mapped merge; the
    paper reports what happens if it does not exist.
15. **Topic "Decision".** Closest existing badge (project 12 is also an
    operations-research/queueing study); avoids touching `web/`.
16. **Retornos as accesses.** U-turns and median crossings move vehicles between
    directions, so they feed one and drain the other. Rejected: keeping them
    internal (v1), which hid the cross-direction flow.
17. **Counts per direction fed, not per street.** A cross street can feed both
    carriageways, so each CSV row names the direction it feeds. At Alfonso Reyes
    the observer confirmed both lanes go towards San Pedro (WB).
18. **Tidal working day as a separate experiment.** The predictor test keeps a
    generic 2-h peak so it is not tied to one assumed daily shape; the day profile
    lives in config and is easy to change.
