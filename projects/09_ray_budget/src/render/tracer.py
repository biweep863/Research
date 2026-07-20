"""A minimal numpy path tracer: primary ray + sun shadow + one diffuse bounce.

Cost is counted, not just timed: every render returns the :class:`Counter` of
ray/primitive tests it performed, so a BVH and brute force can be compared in a
machine-independent currency. Quality comes from Monte-Carlo sampling — more
samples per pixel means less noise — which is exactly the knob the sampling and
denoising experiments turn.
"""
from __future__ import annotations

import numpy as np

from .geometry import EPS, Counter, ray_plane, ray_sphere
from .scene import Scene


def _onb(n: np.ndarray):
    """Orthonormal basis with ``n`` as the up axis."""
    if abs(n[0]) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    else:
        a = np.array([1.0, 0.0, 0.0])
    t = np.cross(a, n)
    t = t / np.linalg.norm(t)
    b = np.cross(n, t)
    return t, b


def _cosine_dir(n: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u1, u2 = rng.random(), rng.random()
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    x, y = r * np.cos(phi), r * np.sin(phi)
    z = np.sqrt(max(0.0, 1.0 - u1))
    t, b = _onb(n)
    return x * t + y * b + z * n


def intersect(scene: Scene, accel, orig, direction, counter: Counter):
    """Nearest hit among ground plane, spheres, and the triangle accelerator.

    Returns ``(t, point, normal, albedo, is_surface)``. ``is_surface`` is False
    for a sky miss.
    """
    best_t = np.inf
    normal = None
    albedo = None

    # Ground plane.
    tg = ray_plane(orig, direction, scene.ground_h, counter)
    if tg < best_t:
        best_t = tg
        normal = np.array([0.0, 1.0, 0.0])
        albedo = scene.ground_alb

    # Spheres.
    for c, r, a in zip(scene.sphere_c, scene.sphere_r, scene.sphere_alb):
        ts = ray_sphere(orig, direction, c, r, counter)
        if ts < best_t:
            best_t = ts
            p = orig + ts * direction
            n = (p - c) / r
            normal = n
            albedo = a

    # Grass (BVH / brute / instanced).
    tt, info = accel.closest(orig, direction, counter)
    if tt < best_t:
        best_t = tt
        normal, albedo = info

    if not np.isfinite(best_t):
        return np.inf, None, None, None, False
    point = orig + best_t * direction
    return best_t, point, normal, albedo, True


def _shadowed(scene: Scene, accel, point, counter: Counter) -> bool:
    o = point + scene.sun_dir * 1e-3
    t, _, _, _, hit = intersect(scene, accel, o, scene.sun_dir, counter)
    return hit and t < np.inf


def _direct(scene: Scene, accel, point, normal, albedo, counter: Counter):
    ndl = np.dot(normal, scene.sun_dir)
    if ndl <= 0:
        return np.zeros(3)
    if _shadowed(scene, accel, point, counter):
        return np.zeros(3)
    return albedo * scene.sun_color * ndl


def radiance(scene: Scene, accel, orig, direction, rng, counter: Counter):
    """One-bounce path-traced radiance for a single ray."""
    t, p, n, alb, hit = intersect(scene, accel, orig, direction, counter)
    if not hit:
        return scene.sky(direction)
    out = _direct(scene, accel, p, n, alb, counter)
    # One cosine-weighted indirect bounce (BRDF/pdf cancel to albedo).
    wi = _cosine_dir(n, rng)
    o2 = p + n * 1e-3
    t2, p2, n2, alb2, hit2 = intersect(scene, accel, o2, wi, counter)
    if not hit2:
        li = scene.sky(wi)
    else:
        li = _direct(scene, accel, p2, n2, alb2, counter)
    return out + alb * li


def render(scene: Scene, accel, spp: int, seed: int = 0, gbuffer: bool = False):
    """Render the scene at ``spp`` samples/pixel.

    Returns ``(image, counter)`` or ``(image, counter, gbuf)`` when
    ``gbuffer=True``. ``gbuf`` holds ``albedo``, ``normal``, ``depth`` from the
    deterministic centre ray (used to guide the denoiser).
    """
    cam = scene.camera
    h, w = cam.height, cam.width
    img = np.zeros((h, w, 3))
    counter = Counter()
    rng = np.random.default_rng(seed)

    gb_alb = np.zeros((h, w, 3)) if gbuffer else None
    gb_nrm = np.zeros((h, w, 3)) if gbuffer else None
    gb_dep = np.zeros((h, w)) if gbuffer else None

    for py in range(h):
        for px in range(w):
            acc = np.zeros(3)
            for _ in range(spp):
                # Jitter the primary ray inside the pixel for anti-aliasing.
                jx = px + (rng.random() - 0.5)
                jy = py + (rng.random() - 0.5)
                o, d = cam.ray(jx, jy)
                acc += radiance(scene, accel, o, d, rng, counter)
            img[py, px] = acc / spp
            if gbuffer:
                o, d = cam.ray(px, py)
                t, p, n, alb, hit = intersect(scene, accel, o, d, counter)
                if hit:
                    gb_alb[py, px] = alb
                    gb_nrm[py, px] = n
                    gb_dep[py, px] = t
                else:
                    gb_alb[py, px] = scene.sky(d)
                    gb_nrm[py, px] = np.array([0.0, 0.0, 1.0])
                    gb_dep[py, px] = 1e3

    img = np.clip(img, 0.0, 1.0)
    if gbuffer:
        return img, counter, {"albedo": gb_alb, "normal": gb_nrm, "depth": gb_dep}
    return img, counter


def cost_map(scene: Scene, accel, shadow: bool = True) -> np.ndarray:
    """Per-pixel ray/primitive test count for a primary (+shadow) ray.

    Used to visualise *where* a frame spends its intersection budget -- the
    grass, overwhelmingly.
    """
    cam = scene.camera
    out = np.zeros((cam.height, cam.width))
    for py in range(cam.height):
        for px in range(cam.width):
            o, d = cam.ray(px, py)
            c = Counter()
            t, p, n, alb, hit = intersect(scene, accel, o, d, c)
            if shadow and hit:
                _shadowed(scene, accel, p, c)
            out[py, px] = c.total()
    return out


def count_primary(scene: Scene, accel, shadow: bool = True) -> Counter:
    """Cast one primary ray per pixel (no Monte-Carlo), counting tests only.

    This isolates the acceleration structure's cost from shading noise and is
    the currency for the accel-scaling experiment.
    """
    cam = scene.camera
    counter = Counter()
    for py in range(cam.height):
        for px in range(cam.width):
            o, d = cam.ray(px, py)
            t, p, n, alb, hit = intersect(scene, accel, o, d, counter)
            if shadow and hit:
                _shadowed(scene, accel, p, counter)
    return counter
