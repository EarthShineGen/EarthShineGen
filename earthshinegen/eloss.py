"""Muon energy loss in the overburden.

Ported from EarthShine's eloss_average.py.  Two average-loss models are
available and both solve  -dE/dx = a + b E  analytically:

  'running'   standard-rock coefficients from PDG 2004 p.230, with a and b
              running with log10(E).  This is the default, and it is the
              parameterisation the standard cosmic muon generators use, so
              samples compare like for like against existing single-muon
              productions.
  'constant'  fixed standard-rock coefficients a = 2.0 MeV/(g/cm^2) and
              b = 4.0e-6 (g/cm^2)^-1.
  'none'      no energy loss; useful for checking the geometry in isolation.

EarthShine can also sample the loss from GEANT4-derived splines, which needs a
several-hundred-megabyte pickle.  That is deliberately not carried here: a
gridpack has to be self-contained and small, and the average loss is what the
published acceptance numbers use.

The second half of the file is multiple Coulomb scattering, which EarthShine
does not simulate at all.  It matters here: over the ~700 m of rock a typical
accepted muon crosses, the RMS deflection is several times the opening angle of
the dark photon decay, so it is the scattering rather than the decay that sets
the pair's apparent invariant mass at the detector.
"""

import numpy as np

from . import constants as k

# Highland/Lynch-Dahl scale, PDG "Passage of particles through matter".
HIGHLAND_ES = 0.0136                       # GeV
# Fraction of the path at which the whole deflection is applied.  A single
# deflection at fraction f leaves a lateral offset (1-f) L theta at the end of
# the path, and the correct RMS offset for scattering spread along the path is
# L theta / sqrt(3), so f = 1 - 1/sqrt(3).  This reproduces both the arrival
# angle and the arrival offset with one kink instead of a stepped transport,
# which would cost the batch vectorisation.  It does not reproduce the
# angle-offset correlation (a single kink gives rho = 1, the true value is
# sqrt(3)/2); at these path lengths that is far below the other approximations.
PIVOT_FRACTION = 1.0 - 1.0 / np.sqrt(3.0)


def momentum_to_energy(p, mass=k.MUON_MASS):
    return np.sqrt(np.asarray(p) ** 2 + mass ** 2)


def energy_to_momentum(E, mass=k.MUON_MASS):
    return np.sqrt(np.maximum(np.asarray(E) ** 2 - mass ** 2, 0.0))


def water_equivalent(distance_m, density=k.ROCK_DENSITY):
    """Column density along the path [g/cm^2]."""
    return np.asarray(distance_m) * 100.0 * density


def propagate_constant(E0, distance_m, density=k.ROCK_DENSITY,
                       a=k.ROCK_A_IONIZATION, b=k.ROCK_B_RADIATIVE):
    """Final energy after `distance_m` of rock, fixed a and b [GeV]."""
    x = water_equivalent(distance_m, density)
    E = (np.asarray(E0) + a / b) * np.exp(-b * x) - a / b
    return np.maximum(E, 0.0)


def propagate_running(E0, distance_m, density=k.ROCK_DENSITY):
    """Final energy after `distance_m` of rock, a and b running with E [GeV].

    Standard rock coefficients from PDG 2004 p.230:

        a = (1.91514 + 0.254957 log10 E) * 1e-3      GeV cm^2/g
        b = (0.379763 + 1.69516 log10 E
                      - 0.175026 log10^2 E) * 1e-6   cm^2/g
        E'= (E + a/b) exp(-b X) - a/b

    a and b are frozen at their starting-energy values over the whole path,
    which is what the reference implementations do.  Muons whose energy would
    go negative have stopped; they are returned as zero rather than as a
    negative energy, which is the one place this differs from EarthShine's
    eloss_average.propagate_muon_CMSSW.
    """
    E0 = np.asarray(E0, dtype=float)
    x = water_equivalent(distance_m, density)

    with np.errstate(divide='ignore', invalid='ignore'):
        l10E = np.log10(np.maximum(E0, 1e-6))
    a = (1.91514 + 0.254957 * l10E) / 1000.0
    b = (0.379763 + 1.69516 * l10E - 0.175026 * l10E ** 2) / 1.0e6
    eps = a / b

    E = (E0 + eps) * np.exp(-b * x) - eps
    return np.maximum(E, 0.0)


def propagate(E0, distance_m, model='running', density=k.ROCK_DENSITY):
    """Dispatch to the requested energy-loss model."""
    if model == 'running':
        return propagate_running(E0, distance_m, density)
    if model == 'constant':
        return propagate_constant(E0, distance_m, density)
    if model == 'none':
        return np.array(E0, dtype=float, copy=True)
    raise ValueError("unknown eloss_model %r; expected 'running', 'constant' "
                     "or 'none'" % model)


def muon_range(p_initial, density=k.ROCK_DENSITY,
               a=k.ROCK_A_IONIZATION, b=k.ROCK_B_RADIATIVE):
    """Distance at which a muon of momentum `p_initial` stops [m]."""
    E0 = momentum_to_energy(p_initial)
    ratio = a / b
    x = (1.0 / b) * np.log((E0 + ratio) / (k.MUON_MASS + ratio))
    return x / (100.0 * density)


def radiation_length_m(density=k.ROCK_DENSITY, x0_g_per_cm2=k.ROCK_X0):
    """Radiation length of the overburden expressed as a distance [m]."""
    return x0_g_per_cm2 / density / 100.0


def scattering_sigma(p_initial, distance_m, model='running',
                     density=k.ROCK_DENSITY, x0_g_per_cm2=k.ROCK_X0,
                     mass=k.MUON_MASS, n_steps=64):
    """RMS projected multiple-scattering angle over a path of rock [rad].

    Highland/Lynch-Dahl, but integrated along the path rather than evaluated at
    a single momentum, because the muon softens as it goes and the scattering
    piles up at the far end:

        sigma^2 = (1 + 0.038 ln(L/X0))^2  int_0^L (Es / (beta p(x)))^2 dx / X0

    with Es = 13.6 MeV.  p(x) comes from the same energy-loss model the
    propagation uses, so the two cannot drift apart.  The logarithmic term is
    evaluated once at the total thickness, which is how PDG writes it.

    This is the Gaussian core only.  At the thousands of radiation lengths
    involved the single-hard-scatter tail is not a small correction to it, so
    treat the result as a floor on the deflection rather than a full
    description.  Muons that stop inside the path get sigma from the part of it
    they survived; they are cut on momentum anyway.
    """
    p_initial = np.atleast_1d(np.asarray(p_initial, dtype=float))
    distance_m = np.atleast_1d(np.asarray(distance_m, dtype=float))
    x0_m = radiation_length_m(density, x0_g_per_cm2)

    # Midpoint rule along each path; every muon gets its own step size.
    frac = (np.arange(n_steps) + 0.5) / n_steps
    x = distance_m[:, None] * frac
    E0 = momentum_to_energy(p_initial, mass)[:, None]
    E = propagate(np.broadcast_to(E0, x.shape), x, model=model, density=density)

    p_x = energy_to_momentum(E, mass)
    # beta * p = p^2 / E, and it is zero exactly where the muon has stopped.
    with np.errstate(divide='ignore', invalid='ignore'):
        beta_p = np.where(E > 0.0, p_x ** 2 / np.maximum(E, 1e-30), 0.0)
        integrand = np.where(beta_p > 0.0,
                             (HIGHLAND_ES / np.maximum(beta_p, 1e-30)) ** 2,
                             0.0)

    var = np.sum(integrand * (distance_m[:, None] / n_steps), axis=1) / x0_m

    # Highland's log term is only meaningful above ~1e-3 radiation lengths, and
    # it must not be allowed to go negative on a very thin path.
    n_rad = np.maximum(distance_m / x0_m, 1e-3)
    log_term = np.maximum(1.0 + 0.038 * np.log(n_rad), 0.0)

    sigma = np.sqrt(np.maximum(var, 0.0)) * log_term
    # Beyond a radian the small-angle picture is meaningless; such a muon is
    # not going anywhere near the detector in any case.
    return np.minimum(np.where(distance_m > 0.0, sigma, 0.0), 0.5 * np.pi)


def transverse_basis(u):
    """Two orthonormal vectors perpendicular to each row of `u`."""
    # Cross with whichever axis u leans on least, so the cross product is never
    # degenerate.
    helper = np.zeros_like(u)
    helper[np.arange(u.shape[0]), np.argmin(np.abs(u), axis=1)] = 1.0
    a = np.cross(u, helper)
    a /= np.linalg.norm(a, axis=1)[:, None]
    return a, np.cross(u, a)


def scatter_directions(directions, sigma, rng):
    """Deflect each direction by an independent Gaussian in two normal planes.

    `sigma` is the RMS projected angle, so the space angle is sigma * sqrt(2).
    Returns unit vectors.
    """
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    u = directions / np.linalg.norm(directions, axis=1)[:, None]
    a, b = transverse_basis(u)

    sigma = np.asarray(sigma, dtype=float)
    t_a = rng.normal(0.0, 1.0, u.shape[0]) * sigma
    t_b = rng.normal(0.0, 1.0, u.shape[0]) * sigma

    v = u + t_a[:, None] * a + t_b[:, None] * b
    return v / np.linalg.norm(v, axis=1)[:, None]


def degrade(p4, distance_m, model='running', density=k.ROCK_DENSITY,
            mass=k.MUON_MASS):
    """Apply the energy loss to a set of four-vectors, keeping the direction.

    The average-loss treatment has no angular component -- EarthShine does not
    simulate multiple scattering either -- so the direction is preserved and
    only the magnitude is rescaled.  Returns the degraded four-vectors and a
    boolean mask of the muons that survived.
    """
    E_final = propagate(p4[:, 3], distance_m, model=model, density=density)
    survived = E_final > mass

    p_final = np.sqrt(np.maximum(E_final ** 2 - mass ** 2, 0.0))
    p_initial = np.sqrt(np.sum(p4[:, :3] ** 2, axis=1))
    scale = np.where(p_initial > 0, p_final / np.where(p_initial > 0, p_initial, 1.0), 0.0)

    out = np.column_stack([p4[:, :3] * scale[:, None], E_final])
    return out, survived
