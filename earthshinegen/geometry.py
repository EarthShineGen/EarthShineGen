"""Decay point sampling and the detector cylinders.

Ported from EarthShine's detector_simulation_tools.py, with the decay depth
sampling extended to draw from the dark photon's exponential decay profile
instead of filling the generation volume uniformly.

Coordinates are the detector frame: +y up, +z along the axis of the detector
cylinders, which are centred on the origin.  Their radii and half lengths come
from the parameter card, so nothing here assumes a particular detector.
Lengths are metres.
"""

import numpy as np

from . import constants as k

CM_PER_M = 100.0


def sample_depths_uniform(n, depth_min, depth_max, rng):
    """Depths uniform in the slab [depth_min, depth_max] (both negative, metres)."""
    return depth_min + (depth_max - depth_min) * rng.random(n)


def sample_depths_exponential(n, depth_min, depth_max, decay_length_cm, rng):
    """Depths drawn from the dark photon decay profile inside the slab.

    A dark photon leaving the core survives to radius r with probability
    exp(-r/L), so the density of decay points at radius r is proportional to
    exp(-r/L).  Writing d = -depth for the depth below the detector, r = R - d
    (R the Earth's radius) and the density in d is proportional to exp(+d/L):
    the deep end of the slab is favoured, because a dark photon that decays
    shallow had to survive further.

    Inverting the CDF over d in [d_lo, d_hi],

        d = L * log( exp(d_lo/L) + u * (exp(d_hi/L) - exp(d_lo/L)) ),

    evaluated through logaddexp so that d_hi/L of a few hundred does not
    overflow.  When the slab is much shorter than L this reduces to the uniform
    case to machine precision, which is the regime at small epsilon.
    """
    d_lo = min(abs(depth_min), abs(depth_max)) * CM_PER_M     # shallow edge, cm
    d_hi = max(abs(depth_min), abs(depth_max)) * CM_PER_M     # deep edge, cm
    L = float(decay_length_cm)

    if not np.isfinite(L) or L <= 0:
        raise ValueError("decay length must be positive and finite, got %r" % L)

    span = (d_hi - d_lo) / L
    if span < 1e-8:
        # Degenerate slab, or L astronomically larger than it: uniform.
        return -(d_lo + (d_hi - d_lo) * rng.random(n)) / CM_PER_M

    u = rng.random(n)
    # log( e^{a} + u (e^{b} - e^{a}) ) with a = d_lo/L, b = d_hi/L, done
    # relative to b to keep the exponentials bounded:
    #   = b + log( u + (1-u) e^{a-b} )
    a_minus_b = -span
    d = L * (d_hi / L + np.log(u + (1.0 - u) * np.exp(a_minus_b)))
    return -d / CM_PER_M


def sample_origins(n, disk_radius, depths, rng):
    """Decay points uniformly over a disk of `disk_radius` at the given depths.

    Returns an (n, 3) array of (x, y, z) in metres; y is the depth (negative,
    below the detector) and the disk lies in the horizontal (x, z) plane.
    """
    phi = 2 * np.pi * rng.random(n)
    r = disk_radius * np.sqrt(rng.random(n))
    x = r * np.cos(phi)
    z = r * np.sin(phi)
    return np.column_stack([x, depths, z])


def auto_disk_radius(depth):
    """EarthShine's angle-based disk radius when none is given.

    The radius that a track at 91 degrees from the vertical would need in order
    to still reach the detector from that depth, capped at 6 km.
    """
    angle = np.deg2rad(91)
    radius = int(np.ceil(abs(abs(depth) * np.tan(angle))))
    return min(radius, 6000)


def ray_cylinder_intersection(origins, directions, radius, half_length):
    """Entry and exit points of rays through a z-aligned cylinder at the origin.

    Ported verbatim in behaviour from EarthShine's ray_cylinder_intersection.
    Rays that miss get NaN in both outputs.  `directions` need not be
    normalised; the returned points are absolute positions in the same units as
    `origins`.
    """
    origins = np.atleast_2d(np.asarray(origins, dtype=float))
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    n = origins.shape[0]

    entry = np.full((n, 3), np.nan)
    exit_ = np.full((n, 3), np.nan)

    ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

    # Curved surface: (ox + t dx)^2 + (oy + t dy)^2 = R^2
    a = dx ** 2 + dy ** 2
    b = 2 * (ox * dx + oy * dy)
    c = ox ** 2 + oy ** 2 - radius ** 2
    disc = b ** 2 - 4 * a * c

    t_candidates = np.full((n, 4), np.inf)

    valid_curved = (disc >= 0) & (np.abs(a) > 1e-12)
    sqrt_disc = np.sqrt(np.maximum(disc, 0))

    # Rays that miss carry infinities through the arithmetic below (inf * 0 for
    # a horizontal ray, for instance).  Those entries are masked out again by
    # the `valid_curved` and cap tests, so the warnings are suppressed rather
    # than worked around.
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(valid_curved, (-b - sqrt_disc) / (2 * a), np.inf)
        t2 = np.where(valid_curved, (-b + sqrt_disc) / (2 * a), np.inf)

        z1 = oz + t1 * dz
        z2 = oz + t2 * dz
        t_candidates[:, 0] = np.where(valid_curved & (t1 >= 0)
                                      & (np.abs(z1) <= half_length), t1, np.inf)
        t_candidates[:, 1] = np.where(valid_curved & (t2 >= 0)
                                      & (np.abs(z2) <= half_length), t2, np.inf)

        # End caps at z = +/- half_length
        dz_safe = np.where(np.abs(dz) > 1e-12, dz, np.inf)
        t_top = (half_length - oz) / dz_safe
        t_bot = (-half_length - oz) / dz_safe

        r2_top = (ox + t_top * dx) ** 2 + (oy + t_top * dy) ** 2
        r2_bot = (ox + t_bot * dx) ** 2 + (oy + t_bot * dy) ** 2

    cap_ok = np.abs(dz) > 1e-12
    t_candidates[:, 2] = np.where((t_top >= 0) & (r2_top <= radius ** 2) & cap_ok,
                                  t_top, np.inf)
    t_candidates[:, 3] = np.where((t_bot >= 0) & (r2_bot <= radius ** 2) & cap_ok,
                                  t_bot, np.inf)

    t_sorted = np.sort(t_candidates, axis=1)
    t_entry, t_exit = t_sorted[:, 0], t_sorted[:, 1]
    hit = np.isfinite(t_entry) & np.isfinite(t_exit)

    entry[hit] = origins[hit] + t_entry[hit, None] * directions[hit]
    exit_[hit] = origins[hit] + t_exit[hit, None] * directions[hit]
    return entry, exit_


def path_length(origins, points):
    """Distance from each origin to each point; NaN where the point is NaN."""
    d = points - origins
    return np.sqrt(np.sum(d * d, axis=1))


def unit(vectors):
    """Normalise rows to unit length."""
    norm = np.linalg.norm(vectors, axis=1)
    return vectors / norm[:, None]


def volume_m3(depth_min, depth_max, disk_radius):
    """Volume of the generation cylinder [m^3]."""
    return abs(depth_max - depth_min) * np.pi * disk_radius ** 2


def cross_section_area_cm2(disk_radius_m):
    """Horizontal cross-sectional area of the generation cylinder [cm^2].

    This is the area that intercepts the dark photon flux from the core, and it
    replaces IceCube's effective area in the rate calculation.
    """
    return np.pi * (disk_radius_m * CM_PER_M) ** 2
