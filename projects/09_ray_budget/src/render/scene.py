"""The canonical scene: a procedural grass field, spheres, ground, and a sun.

Everything is deterministic given a seed. The grass is defined once as a
canonical blade in local space and placed by affine transforms, so the same
field can be rendered three ways that must agree bit-for-bit when lossless:

  * ``bake_field`` -> a flat triangle soup (for BruteForce / BVH),
  * ``instances``  -> affine placements of one blade (for TwoLevelBVH).

Level-of-detail is expressed by giving distant blades a cheaper ``mode``
(fewer triangles), which only affects the *baked* geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bvh import Instance
from .geometry import Triangles, make_triangles

# Blade base colours (a mild green spread) and a brighter tip.
_TIP_GAIN = np.array([1.25, 1.30, 1.10])


def blade_local(mode: str = "full") -> tuple[np.ndarray, np.ndarray]:
    """Canonical blade in local space (base at origin, growing in +y).

    Returns ``(verts, albedo)`` with ``verts`` shape ``(T, 3, 3)``. Triangle
    counts: full=4, simplified=1, billboard=2 -- the LOD cost ladder.
    """
    w = 0.06          # half-width at the base
    h = 1.0           # height (scaled per instance)
    bend = 0.18       # forward lean of the tip in +z
    base = np.array([1.0, 1.0, 1.0])
    tip = _TIP_GAIN

    if mode == "simplified":
        v = np.array([[[-w, 0, 0], [w, 0, 0], [0.0, h, bend]]])
        alb = tip[None, :]
        return v, alb

    if mode == "billboard":
        # A flat card: two triangles, one quad, cheap and planar.
        v = np.array([
            [[-w, 0, 0], [w, 0, 0], [w, h, bend]],
            [[-w, 0, 0], [w, h, bend], [-w, h, bend]],
        ])
        alb = np.array([base, tip])
        return v, alb

    # "full": a two-segment tapered blade = 4 triangles.
    mh = 0.5 * h
    mw = 0.5 * w
    mb = 0.5 * bend
    v = np.array([
        [[-w, 0, 0], [w, 0, 0], [mw, mh, mb]],
        [[-w, 0, 0], [mw, mh, mb], [-mw, mh, mb]],
        [[-mw, mh, mb], [mw, mh, mb], [0.02, h, bend]],
        [[-mw, mh, mb], [0.02, h, bend], [-0.02, h, bend]],
    ])
    alb = np.array([base, base, tip, tip])
    return v, alb


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float
    width: int
    height: int

    def __post_init__(self):
        f = self.target - self.eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, self.up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        self._f, self._r, self._u = f, r, u
        self._scale = np.tan(np.radians(self.fov_deg) * 0.5)
        self._aspect = self.width / self.height

    def ray(self, px: int, py: int):
        # Pixel centre in [-1, 1], y up.
        ndc_x = (2 * (px + 0.5) / self.width - 1) * self._aspect * self._scale
        ndc_y = (1 - 2 * (py + 0.5) / self.height) * self._scale
        d = self._f + ndc_x * self._r + ndc_y * self._u
        d = d / np.linalg.norm(d)
        return self.eye.copy(), d


@dataclass
class Scene:
    camera: Camera
    instances: list          # list[Instance] (one canonical blade)
    blade: Triangles         # canonical full blade (local space)
    sphere_c: np.ndarray     # (S, 3)
    sphere_r: np.ndarray     # (S,)
    sphere_alb: np.ndarray   # (S, 3)
    sun_dir: np.ndarray      # unit, points *toward* the sun
    sun_color: np.ndarray
    sky_top: np.ndarray
    sky_bottom: np.ndarray
    ground_alb: np.ndarray
    ground_h: float = 0.0
    modes: list = None       # per-instance LOD mode, for baking

    def sky(self, direction: np.ndarray) -> np.ndarray:
        t = 0.5 * (direction[1] + 1.0)
        return (1 - t) * self.sky_bottom + t * self.sky_top


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def build_scene(n_blades: int, n_spheres: int, width: int, height: int,
                rng: np.random.Generator) -> Scene:
    """Assemble the grass field, spheres, sun, sky, and camera."""
    field = 6.0        # half-extent of the grass patch in x and z
    # Jittered grid of blade positions.
    side = int(np.ceil(np.sqrt(n_blades)))
    xs = np.linspace(-field, field, side)
    zs = np.linspace(-field, field, side)
    gx, gz = np.meshgrid(xs, zs)
    pts = np.stack([gx.ravel(), gz.ravel()], axis=1)
    pts = pts[:n_blades]
    jitter = (rng.random((len(pts), 2)) - 0.5) * (2 * field / side)
    pts = pts + jitter

    blade_v, blade_alb = blade_local("full")
    blade = make_triangles(blade_v, blade_alb)
    blade_lo = blade.v0.min(axis=0)
    blade_hi = (blade.v0 + np.maximum(blade.e1, blade.e2)).max(axis=0)
    lo3 = np.minimum.reduce([blade.v0, blade.v0 + blade.e1, blade.v0 + blade.e2]).min(axis=0)
    hi3 = np.maximum.reduce([blade.v0, blade.v0 + blade.e1, blade.v0 + blade.e2]).max(axis=0)

    instances = []
    for (x, z) in pts:
        yaw = rng.uniform(0, 2 * np.pi)
        tilt = rng.uniform(-0.12, 0.12)
        scale = rng.uniform(0.6, 1.15)
        A = _rot_y(yaw) @ _rot_x(tilt) @ np.diag([1.0, scale, 1.0])
        b = np.array([x, 0.0, z])
        green = np.array([0.18, 0.55, 0.16]) * rng.uniform(0.7, 1.15)
        instances.append(Instance(A, b, lo3, hi3, green))

    # A few spheres resting on the ground.
    sc, sr, sa = [], [], []
    palette = np.array([[0.85, 0.35, 0.30], [0.30, 0.45, 0.85],
                        [0.90, 0.80, 0.30], [0.60, 0.30, 0.75], [0.30, 0.75, 0.65]])
    for i in range(n_spheres):
        r = rng.uniform(0.5, 0.9)
        x = rng.uniform(-field * 0.6, field * 0.6)
        z = rng.uniform(-field * 0.4, field * 0.5)
        sc.append([x, r, z])
        sr.append(r)
        sa.append(palette[i % len(palette)])

    sun = np.array([0.4, 0.85, 0.35])
    sun = sun / np.linalg.norm(sun)

    cam = Camera(
        eye=np.array([0.0, 1.6, 9.0]),
        target=np.array([0.0, 0.7, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=45.0, width=width, height=height,
    )

    return Scene(
        camera=cam,
        instances=instances,
        blade=blade,
        sphere_c=np.array(sc).reshape(-1, 3),
        sphere_r=np.array(sr).reshape(-1),
        sphere_alb=np.array(sa).reshape(-1, 3),
        sun_dir=sun,
        sun_color=np.array([1.15, 1.08, 0.95]),
        sky_top=np.array([0.35, 0.55, 0.85]),
        sky_bottom=np.array([0.75, 0.82, 0.90]),
        ground_alb=np.array([0.32, 0.26, 0.19]),
        ground_h=0.0,
        modes=["full"] * len(instances),
    )


def assign_lod(scene: Scene, threshold: float, far_mode: str) -> list[str]:
    """Assign each blade 'full' if near the camera, ``far_mode`` if far.

    ``threshold`` in [0, 1] is the fraction of the camera-distance range within
    which blades keep full detail. ``threshold=1`` keeps everything full;
    ``threshold=0`` makes every blade the cheap ``far_mode``.
    """
    eye = scene.camera.eye
    d = np.array([np.linalg.norm(ins.b - eye) for ins in scene.instances])
    dmin, dmax = d.min(), d.max()
    span = max(dmax - dmin, 1e-9)
    frac = (d - dmin) / span
    return ["full" if f <= threshold else far_mode for f in frac]


def bake_field(scene: Scene, modes: list[str]) -> Triangles:
    """Flatten instances into a world-space triangle soup using per-blade LOD."""
    cache = {m: blade_local(m) for m in set(modes)}
    all_v = []
    all_alb = []
    for ins, mode in zip(scene.instances, modes):
        bv, ba = cache[mode]
        world = bv @ ins.A.T + ins.b        # (T,3,3)
        all_v.append(world)
        all_alb.append(ba * ins.albedo)
    verts = np.concatenate(all_v, axis=0)
    alb = np.concatenate(all_alb, axis=0)
    return make_triangles(verts, alb)
