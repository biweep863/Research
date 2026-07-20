"""Unit tests: intersections vs closed forms, BVH == brute force, instancing ==
duplicated geometry, Monte-Carlo convergence, and metric/denoiser properties.
`make test` must pass before running experiments. Fast (tiny scenes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

from src.render.geometry import (  # noqa: E402
    Counter, make_triangles, ray_aabb, ray_plane, ray_sphere, ray_triangles,
)
from src.render.bvh import BVH, BruteForce, Instance, TwoLevelBVH  # noqa: E402
from src.render.scene import bake_field, blade_local, build_scene  # noqa: E402
from src.render.tracer import count_primary, intersect, render  # noqa: E402
from src.quality.metrics import bilateral_denoise, psnr, ssim  # noqa: E402


# --------------------------------------------------------------------------- #
# Closed-form intersections
# --------------------------------------------------------------------------- #
def test_ray_sphere_closed_form():
    c = Counter()
    # Unit sphere at (0,0,5), ray down +z from origin: first hit at t=4.
    t = ray_sphere(np.array([0.0, 0, 0]), np.array([0.0, 0, 1]),
                   np.array([0.0, 0, 5]), 1.0, c)
    assert abs(t - 4.0) < 1e-9
    # Ray pointing away misses.
    miss = ray_sphere(np.array([0.0, 0, 0]), np.array([0.0, 0, -1]),
                      np.array([0.0, 0, 5]), 1.0, c)
    assert not np.isfinite(miss)
    assert c.sphere == 2


def test_ray_plane_closed_form():
    c = Counter()
    t = ray_plane(np.array([0.0, 3, 0]), np.array([0.0, -1, 0]), 0.0, c)
    assert abs(t - 3.0) < 1e-9
    # Parallel ray misses.
    assert not np.isfinite(ray_plane(np.array([0.0, 3, 0]),
                                     np.array([1.0, 0, 0]), 0.0, c))


def test_ray_triangle_closed_form():
    c = Counter()
    tri = make_triangles(
        np.array([[[0.0, 0, 5], [1, 0, 5], [0, 1, 5]]]),
        np.array([[0.5, 0.5, 0.5]]))
    t, j = ray_triangles(np.array([0.2, 0.2, 0.0]), np.array([0.0, 0, 1]), tri, c)
    assert j == 0 and abs(t - 5.0) < 1e-9
    # Ray outside the triangle in-plane region misses.
    t2, j2 = ray_triangles(np.array([2.0, 2.0, 0.0]), np.array([0.0, 0, 1]), tri, c)
    assert j2 == -1
    assert c.tri == 2   # one test per triangle per call


def test_ray_aabb_slab():
    c = Counter()
    lo, hi = np.array([-1.0, -1, -1]), np.array([1.0, 1, 1])
    inv = 1.0 / np.array([0.0 + 1e-9, 0.0 + 1e-9, 1.0])
    assert ray_aabb(np.array([0.0, 0, -5]), inv, lo, hi, np.inf, c)
    # Box behind a nearer t_max is rejected.
    assert not ray_aabb(np.array([0.0, 0, -5]), inv, lo, hi, 1.0, c)


# --------------------------------------------------------------------------- #
# BVH is lossless: same nearest hit as brute force
# --------------------------------------------------------------------------- #
def test_bvh_matches_brute_force():
    rng = np.random.default_rng(1)
    verts = rng.normal(size=(200, 3, 3)) * 2.0
    verts[:, :, 2] += 10.0     # push in front of the camera
    alb = rng.random((200, 3))
    tris = make_triangles(verts, alb)
    brute = BruteForce(tris)
    bvh = BVH(tris, leaf_size=4)
    for _ in range(50):
        o = rng.normal(size=3) * 0.5
        d = np.array([rng.normal() * 0.2, rng.normal() * 0.2, 1.0])
        d = d / np.linalg.norm(d)
        cb, cv = Counter(), Counter()
        tb, ib = brute.closest(o, d, cb)
        tv, iv = bvh.closest(o, d, cv)
        assert abs(tb - tv) < 1e-9 or (not np.isfinite(tb) and not np.isfinite(tv))
        # BVH never does more triangle tests than brute force.
        assert cv.tri <= cb.tri


def test_bvh_fewer_tests_than_brute():
    rng = np.random.default_rng(2)
    scene = build_scene(n_blades=300, n_spheres=2, width=32, height=24, rng=rng)
    tris = bake_field(scene, scene.modes)
    brute = count_primary_tris(scene, BruteForce(tris))
    accel = count_primary_tris(scene, BVH(tris, leaf_size=4))
    assert accel < brute / 5   # at least 5x fewer triangle tests


def count_primary_tris(scene, accel):
    return count_primary(scene, accel, shadow=False).tri


# --------------------------------------------------------------------------- #
# Instancing (two-level BVH) == duplicated geometry, bit for bit
# --------------------------------------------------------------------------- #
def test_instancing_matches_duplicated_geometry():
    rng = np.random.default_rng(3)
    scene = build_scene(n_blades=120, n_spheres=2, width=48, height=36, rng=rng)
    baked = bake_field(scene, scene.modes)          # duplicated geometry
    bvh = BVH(baked, leaf_size=4)
    two = TwoLevelBVH(scene.blade, scene.instances, leaf_size=4)
    img_dup, _ = render(scene_with_accel(scene), bvh, spp=1, seed=7)
    img_ins, _ = render(scene_with_accel(scene), two, spp=1, seed=7)
    assert np.max(np.abs(img_dup - img_ins)) < 1e-9


def scene_with_accel(scene):
    return scene  # accel is passed separately to render


# --------------------------------------------------------------------------- #
# Monte-Carlo convergence: error ~ 1/sqrt(N)
# --------------------------------------------------------------------------- #
def test_monte_carlo_slope():
    rng = np.random.default_rng(4)
    scene = build_scene(n_blades=80, n_spheres=2, width=24, height=18, rng=rng)
    two = TwoLevelBVH(scene.blade, scene.instances, leaf_size=4)
    ref, _ = render(scene, two, spp=64, seed=1)
    spps = [1, 2, 4, 8]
    errs = []
    for s in spps:
        img, _ = render(scene, two, spp=s, seed=2)
        errs.append(np.sqrt(np.mean((img - ref) ** 2)))
    slope = np.polyfit(np.log(spps), np.log(errs), 1)[0]
    assert -0.75 < slope < -0.25   # theory: -0.5


# --------------------------------------------------------------------------- #
# Metrics and denoiser
# --------------------------------------------------------------------------- #
def test_metric_properties():
    rng = np.random.default_rng(5)
    img = rng.random((24, 24, 3))
    assert psnr(img, img) >= 99.9
    assert abs(ssim(img, img) - 1.0) < 1e-6
    noisy = np.clip(img + rng.normal(scale=0.1, size=img.shape), 0, 1)
    assert ssim(noisy, img) < 1.0
    assert psnr(noisy, img) < psnr(img, img)


def test_denoiser_does_not_degrade():
    rng = np.random.default_rng(6)
    scene = build_scene(n_blades=60, n_spheres=2, width=32, height=24, rng=rng)
    two = TwoLevelBVH(scene.blade, scene.instances, leaf_size=4)
    ref, _ = render(scene, two, spp=48, seed=1)
    noisy, _, gbuf = render(scene, two, spp=2, seed=2, gbuffer=True)
    den = bilateral_denoise(noisy, gbuf, sigma_spatial=1.5, sigma_color=0.15)
    assert ssim(den, ref) >= ssim(noisy, ref) - 1e-6
