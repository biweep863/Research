# 01 · "Dataset" — the procedural scene

There is no external dataset. The scene is generated deterministically from a
seed, so every number reproduces on a clean machine.

## The scene (`src/render/scene.py`)
- **Grass field.** A jittered grid of blades over a rectangle in front of the
  camera. Each blade is the *same* canonical shape in local space, placed by an
  affine transform `A x + b` (yaw rotation, small tilt, height scale, ground
  position). Blade colour is a green with per-blade variation and a brighter tip.
- **Canonical blade LODs** (`blade_local`): `full` = 4 triangles (two-segment
  tapered blade), `simplified` = 1 triangle, `billboard` = 2 triangles (flat
  card). This is the cost ladder for the LOD experiment.
- **Spheres.** A few coloured spheres resting on the ground plane.
- **Ground.** A horizontal plane at `y = 0`, matte brown.
- **Sun + sky.** One directional light; a top-to-horizon gradient sky used for
  ray misses and the indirect bounce.
- **Camera.** A pinhole looking across the field, slightly above and pitched
  down.

## Two equivalent representations
The same field is produced two ways that must agree when lossless:
- `bake_field(scene, modes)` → a flat world-space triangle soup (for brute force
  / BVH), with each blade contributing its assigned-LOD triangles.
- `scene.instances` → affine placements of one canonical blade (for the
  two-level / instanced BVH).

## Config knobs (`configs/local_small.yaml`)
- `render`: image size (64×48), reference spp (24), blades (200), spheres (3).
- `accel.blade_counts`: scene sizes for the cost-scaling sweep.
- `lod`: distance thresholds, far-modes, render spp.
- `sampling`: spp grid and denoiser sigmas.

`configs/gpu_full.yaml` scales all of these up (bigger images, more samples,
larger fields, finer sweeps) with no code changes. Everything is CPU/numpy;
"gpu" only names the convention shared with the other projects.
