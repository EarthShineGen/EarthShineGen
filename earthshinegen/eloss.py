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
"""

import numpy as np

from . import constants as k


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
