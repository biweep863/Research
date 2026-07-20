"""Acceleration structures: a median-split BVH and a two-level (instanced) BVH.

Both expose the same ``closest(orig, direction, counter)`` interface used by the
tracer, returning ``(t, info)`` where ``info`` is ``(normal, albedo)`` in world
space or ``None`` on a miss. This lets the renderer swap brute force, a BVH, or
instancing without changing a line of shading code.

The two-level BVH is the "Zelda grass trick": one canonical blade is stored
once (the bottom-level structure, BLAS) and referenced by many instances, each a
cheap affine transform in a top-level structure (TLAS). Rays are transformed
into each instance's local space, so the blade geometry is never duplicated.
"""
from __future__ import annotations

import numpy as np

from .geometry import EPS, Counter, Triangles, ray_aabb, ray_triangles


def _safe_inv(direction: np.ndarray) -> np.ndarray:
    d = direction.copy()
    d[np.abs(d) < EPS] = EPS
    return 1.0 / d


class _Node:
    __slots__ = ("lo", "hi", "left", "right", "prims")

    def __init__(self, lo, hi, left=None, right=None, prims=None):
        self.lo = lo
        self.hi = hi
        self.left = left
        self.right = right
        self.prims = prims       # np.ndarray of triangle indices, or None (internal)


class BruteForce:
    """Baseline: test every triangle for every ray. O(rays x primitives)."""

    def __init__(self, tris: Triangles):
        self.tris = tris

    @property
    def n_boxes(self) -> int:
        return 0

    def closest(self, orig, direction, counter: Counter):
        t, j = ray_triangles(orig, direction, self.tris, counter)
        if j < 0:
            return np.inf, None
        n = self.tris.normal[j]
        if np.dot(n, direction) > 0:
            n = -n
        return t, (n, self.tris.albedo[j])


class BVH:
    """Median-split bounding-volume hierarchy over a triangle soup (lossless)."""

    def __init__(self, tris: Triangles, leaf_size: int = 4):
        self.tris = tris
        self.leaf_size = max(1, int(leaf_size))
        self.n_nodes = 0
        lo, hi = tris.bounds()
        cent = tris.centroids()
        self.root = self._build(np.arange(len(tris)), lo, hi, cent)

    def _build(self, idx, lo, hi, cent) -> _Node:
        self.n_nodes += 1
        blo = lo[idx].min(axis=0)
        bhi = hi[idx].max(axis=0)
        if len(idx) <= self.leaf_size:
            return _Node(blo, bhi, prims=idx)
        extent = cent[idx].max(axis=0) - cent[idx].min(axis=0)
        axis = int(np.argmax(extent))
        order = idx[np.argsort(cent[idx, axis])]
        mid = len(order) // 2
        left_idx, right_idx = order[:mid], order[mid:]
        if len(left_idx) == 0 or len(right_idx) == 0:
            return _Node(blo, bhi, prims=idx)
        left = self._build(left_idx, lo, hi, cent)
        right = self._build(right_idx, lo, hi, cent)
        return _Node(blo, bhi, left, right)

    @property
    def n_boxes(self) -> int:
        return self.n_nodes

    def closest(self, orig, direction, counter: Counter):
        inv = _safe_inv(direction)
        best_t = np.inf
        best_j = -1
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not ray_aabb(orig, inv, node.lo, node.hi, best_t, counter):
                continue
            if node.prims is not None:
                t, j = ray_triangles(orig, direction, self.tris.subset(node.prims),
                                     counter)
                if t < best_t:
                    best_t = t
                    best_j = int(node.prims[j])
            else:
                stack.append(node.left)
                stack.append(node.right)
        if best_j < 0:
            return np.inf, None
        n = self.tris.normal[best_j]
        if np.dot(n, direction) > 0:
            n = -n
        return best_t, (n, self.tris.albedo[best_j])


class Instance:
    """An affine placement ``x -> A x + b`` of the canonical blade."""

    __slots__ = ("A", "b", "Ainv", "N", "lo", "hi", "albedo")

    def __init__(self, A: np.ndarray, b: np.ndarray, blade_lo, blade_hi,
                 albedo: np.ndarray):
        self.A = A
        self.b = b
        self.Ainv = np.linalg.inv(A)
        self.N = np.linalg.inv(A).T       # normal transform
        self.albedo = albedo
        # World-space AABB of the transformed blade box (8 corners).
        corners = np.array([[x, y, z]
                            for x in (blade_lo[0], blade_hi[0])
                            for y in (blade_lo[1], blade_hi[1])
                            for z in (blade_lo[2], blade_hi[2])])
        world = corners @ A.T + b
        self.lo = world.min(axis=0)
        self.hi = world.max(axis=0)


class TwoLevelBVH:
    """TLAS over instances of a single BLAS blade -- the grass instancing trick.

    Geometry is stored once (``blade``); each instance is an affine transform.
    Memory is ``blade + K transforms`` instead of ``K x blade`` triangles.
    """

    def __init__(self, blade: Triangles, instances: list[Instance],
                 leaf_size: int = 4):
        self.blade = blade
        self.blade_bvh = BVH(blade, leaf_size)
        self.instances = instances
        self.leaf_size = max(1, int(leaf_size))
        self.n_nodes = 0
        lo = np.array([ins.lo for ins in instances])
        hi = np.array([ins.hi for ins in instances])
        cent = 0.5 * (lo + hi)
        self.root = self._build(np.arange(len(instances)), lo, hi, cent)

    def _build(self, idx, lo, hi, cent) -> _Node:
        self.n_nodes += 1
        blo = lo[idx].min(axis=0)
        bhi = hi[idx].max(axis=0)
        if len(idx) <= self.leaf_size:
            return _Node(blo, bhi, prims=idx)
        extent = cent[idx].max(axis=0) - cent[idx].min(axis=0)
        axis = int(np.argmax(extent))
        order = idx[np.argsort(cent[idx, axis])]
        mid = len(order) // 2
        left_idx, right_idx = order[:mid], order[mid:]
        if len(left_idx) == 0 or len(right_idx) == 0:
            return _Node(blo, bhi, prims=idx)
        return _Node(blo, bhi, self._build(left_idx, lo, hi, cent),
                     self._build(right_idx, lo, hi, cent))

    @property
    def n_boxes(self) -> int:
        return self.n_nodes + self.blade_bvh.n_nodes

    def _blade_hit(self, orig_l, dir_l, counter):
        """Nearest triangle of the canonical blade in local space."""
        inv = _safe_inv(dir_l)
        best_t, best_j = np.inf, -1
        stack = [self.blade_bvh.root]
        while stack:
            node = stack.pop()
            if not ray_aabb(orig_l, inv, node.lo, node.hi, best_t, counter):
                continue
            if node.prims is not None:
                t, j = ray_triangles(orig_l, dir_l,
                                     self.blade.subset(node.prims), counter)
                if t < best_t:
                    best_t, best_j = t, int(node.prims[j])
            else:
                stack.append(node.left)
                stack.append(node.right)
        return best_t, best_j

    def closest(self, orig, direction, counter: Counter):
        inv = _safe_inv(direction)
        best_t = np.inf
        best_info = None
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not ray_aabb(orig, inv, node.lo, node.hi, best_t, counter):
                continue
            if node.prims is not None:
                for ii in node.prims:
                    ins = self.instances[int(ii)]
                    orig_l = ins.Ainv @ (orig - ins.b)
                    dir_l = ins.Ainv @ direction        # not renormalised: keeps t
                    t, j = self._blade_hit(orig_l, dir_l, counter)
                    if t < best_t:
                        best_t = t
                        n = ins.N @ self.blade.normal[j]
                        ln = np.linalg.norm(n)
                        if ln > EPS:
                            n = n / ln
                        if np.dot(n, direction) > 0:
                            n = -n
                        best_info = (n, self.blade.albedo[j] * ins.albedo)
            else:
                stack.append(node.left)
                stack.append(node.right)
        if best_info is None:
            return np.inf, None
        return best_t, best_info
