"""Ray/primitive intersection in closed form, with an explicit test counter.

Every routine that performs a ray/primitive test increments a shared
:class:`Counter`. The counter is the machine-independent cost currency used
throughout the project: brute force pays ``rays x primitives`` triangle tests,
a BVH pays far fewer, and the numbers are reproducible on any machine.

Conventions: rays are ``origin + t * direction`` with ``t > EPS`` for a valid
hit; directions need not be unit length for the algebra but we keep them unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-6


@dataclass
class Counter:
    """Counts ray/primitive tests, split by primitive kind."""
    tri: int = 0
    box: int = 0
    sphere: int = 0
    plane: int = 0

    def total(self) -> int:
        return self.tri + self.box + self.sphere + self.plane

    def snapshot(self) -> dict:
        return {"tri": self.tri, "box": self.box, "sphere": self.sphere,
                "plane": self.plane, "total": self.total()}


@dataclass
class Triangles:
    """A soup of triangles as flat arrays (vertex 0 + two edges + albedo)."""
    v0: np.ndarray            # (P, 3)
    e1: np.ndarray            # (P, 3)  = v1 - v0
    e2: np.ndarray            # (P, 3)  = v2 - v0
    albedo: np.ndarray        # (P, 3)
    normal: np.ndarray = field(default=None)  # (P, 3) unit face normals

    def __post_init__(self):
        if self.normal is None:
            n = np.cross(self.e1, self.e2)
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            ln[ln < EPS] = 1.0
            self.normal = n / ln

    def __len__(self) -> int:
        return int(self.v0.shape[0])

    def centroids(self) -> np.ndarray:
        return self.v0 + (self.e1 + self.e2) / 3.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        v1 = self.v0 + self.e1
        v2 = self.v0 + self.e2
        lo = np.minimum(np.minimum(self.v0, v1), v2)
        hi = np.maximum(np.maximum(self.v0, v1), v2)
        return lo, hi

    def subset(self, idx: np.ndarray) -> "Triangles":
        return Triangles(self.v0[idx], self.e1[idx], self.e2[idx],
                         self.albedo[idx], self.normal[idx])


def make_triangles(tris: np.ndarray, albedo: np.ndarray) -> Triangles:
    """Build a :class:`Triangles` from an ``(P, 3, 3)`` vertex array."""
    v0 = tris[:, 0, :]
    e1 = tris[:, 1, :] - v0
    e2 = tris[:, 2, :] - v0
    return Triangles(v0.copy(), e1.copy(), e2.copy(), albedo.copy())


def ray_triangles(orig: np.ndarray, direction: np.ndarray, tris: Triangles,
                  counter: Counter | None = None):
    """Möller-Trumbore of one ray against a whole triangle set (vectorised).

    Returns ``(t_hit, index)`` for the nearest hit with ``t > EPS`` or
    ``(inf, -1)`` if the ray misses every triangle. Counts ``len(tris)`` tests.
    """
    p = len(tris)
    if counter is not None:
        counter.tri += p
    if p == 0:
        return np.inf, -1
    pvec = np.cross(direction, tris.e2)                 # (P,3)
    det = np.einsum("ij,ij->i", tris.e1, pvec)          # (P,)
    valid = np.abs(det) > EPS
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = orig - tris.v0                                # (P,3)
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, tris.e1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    t = np.einsum("ij,ij->i", tris.e2, qvec) * inv_det
    hit = valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > EPS)
    if not np.any(hit):
        return np.inf, -1
    t_masked = np.where(hit, t, np.inf)
    j = int(np.argmin(t_masked))
    return float(t_masked[j]), j


def ray_sphere(orig: np.ndarray, direction: np.ndarray, center: np.ndarray,
               radius: float, counter: Counter | None = None) -> float:
    """Nearest ``t > EPS`` where the ray meets the sphere, else ``inf``."""
    if counter is not None:
        counter.sphere += 1
    oc = orig - center
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - 4.0 * c            # a = 1 for unit direction
    if disc < 0:
        return np.inf
    s = np.sqrt(disc)
    t0 = (-b - s) * 0.5
    t1 = (-b + s) * 0.5
    if t0 > EPS:
        return float(t0)
    if t1 > EPS:
        return float(t1)
    return np.inf


def ray_plane(orig: np.ndarray, direction: np.ndarray, height: float,
              counter: Counter | None = None) -> float:
    """Intersection with the horizontal plane ``y = height``."""
    if counter is not None:
        counter.plane += 1
    dy = direction[1]
    if abs(dy) < EPS:
        return np.inf
    t = (height - orig[1]) / dy
    return float(t) if t > EPS else np.inf


def ray_aabb(orig: np.ndarray, inv_dir: np.ndarray, lo: np.ndarray,
             hi: np.ndarray, t_max: float, counter: Counter | None = None) -> bool:
    """Slab test: does the ray reach the box before ``t_max``?"""
    if counter is not None:
        counter.box += 1
    t0 = (lo - orig) * inv_dir
    t1 = (hi - orig) * inv_dir
    tmin = np.maximum(np.minimum(t0, t1), 0.0).max()
    tmax = np.maximum(t0, t1).min()
    return tmax >= tmin and tmin < t_max and tmax > EPS
