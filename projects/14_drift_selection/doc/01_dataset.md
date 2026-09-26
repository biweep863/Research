# 01 · Data — NK fitness landscapes (synthetic, no external dataset)

There is no natural dataset for "a genetic algorithm's search dynamics on
landscapes of controlled ruggedness" — the landscape itself is the
generator, following Kauffman & Levin (1987): a bitstring genome of length
$L$; each locus $i$ contributes a fitness value looked up from an
independent random table indexed by the locus's own bit and the next $K$
loci (circular adjacency); total fitness is the mean contribution over all
$L$ loci. $K=0$ is purely additive; increasing $K$ multiplies the number of
local optima.

## Local vs GPU configuration

| | `local_small.yaml` | `gpu_full.yaml` |
|---|---|---|
| Genome length $L$ | 16 (exact optimum: $2^{16}$ enumeration) | 20 (exact optimum: $2^{20}$ enumeration) |
| Epistasis $K$ | 0, 2, 4, 6 | 0–9 (every integer) |
| Population grid $N$ | 4–256 (7 points) | 4–4096 (11 points) |
| Independent landscapes per $K$ | 5 | 30 |
| Replicates per (N, K, landscape) | 100 | 1000 |
| Generations | 150 | 400 |
| Device | CPU | CUDA |

$L$ is kept small enough in *both* configs that the true global optimum is
always known exactly by brute-force enumeration (`NKLandscape.brute_force_optimum`),
rather than approximated by a large random-restart search — this avoids
any ambiguity between "the GA found a very good local optimum" and "the GA
found the actual global optimum," which is the success criterion the whole
threshold-fitting analysis depends on.

Each of the two experiments (`run_threshold_sweep.py`, `run_effective_size.py`)
draws its own landscapes per $K$ (`landscapes_per_k` independent random
tables, seeded by $K$ and a landscape index), fresh from `NKLandscape`.
Nothing is cached to `data/` — landscape generation and brute-force
enumeration are cheap enough (milliseconds to seconds) to redo on every run,
so `data/` is empty by design (unlike the LLM-scale projects in this repo,
which cache downloaded corpora).
