"""Dark photon production and its two-body decay to a muon pair.

EarthShine draws the decay with `phasespace`, which pulls in TensorFlow.  For a
two-body decay of a spin-averaged resonance that is a very expensive way to
sample an isotropic direction, and TensorFlow is not something a gridpack
should have to carry, so the decay is done here in closed form.

Coordinates are the detector frame throughout: +y is up, +z is along the axis
of the detector cylinder, +x completes the right-handed set.  The dark photons
travel upward.
"""

import numpy as np

from . import constants as k


def dark_photon_momenta(m_X, m_A, n_events, dm_model, rng):
    """Four-momenta of the dark photons entering the generation volume.

    chi chi-bar -> A' A' with the dark matter effectively at rest (the AN's
    Maxwell-Boltzmann core is at 6000 K, giving speeds of order 10^-5 c), so
    each dark photon carries E = m_X and |p| = sqrt(m_X^2 - m_A^2).

    dm_model selects the angular distribution:

      'core'          all dark photons travel straight up, the Feng et al.
                      picture where the dark matter has thermalised into the
                      centre of the Earth and only the radial direction points
                      at the detector.
      'floating'      isotropic in the upward hemisphere, the Leane et al.
                      picture where the dark matter is spread through the crust.
      'monoenergetic' same directions as 'floating'; the difference appears
                      after the decay, where the muon momenta are forced to
                      exactly m_X/2 to match what a fixed-momentum cosmic muon
                      generator produces.  Used for validation against existing
                      single-muon samples.

    Returns an (n_events, 4) array of (px, py, pz, E) in GeV.
    """
    p_mag = np.sqrt(max(m_X ** 2 - m_A ** 2, 0.0))
    E = m_X * np.ones(n_events)

    if dm_model == 'core':
        px = np.zeros(n_events)
        py = p_mag * np.ones(n_events)
        pz = np.zeros(n_events)
    elif dm_model in ('floating', 'monoenergetic'):
        # cos(alpha) uniform on [0, 1] gives an isotropic upward hemisphere,
        # with alpha measured from the +y (vertical) axis.
        cos_alpha = rng.random(n_events)
        sin_alpha = np.sqrt(1.0 - cos_alpha ** 2)
        beta = 2 * np.pi * rng.random(n_events)
        px = p_mag * sin_alpha * np.cos(beta)
        py = p_mag * cos_alpha
        pz = p_mag * sin_alpha * np.sin(beta)
    else:
        raise ValueError("unknown dm_model %r; expected 'core', 'floating' or "
                         "'monoenergetic'" % dm_model)

    return np.column_stack([px, py, pz, E])


def two_body_decay(parent, m_A, m_daughter, rng):
    """Decay each parent four-vector isotropically into two equal-mass daughters.

    In the parent rest frame the daughters are back to back with
    E* = m_A/2 and |p*| = sqrt(m_A^2/4 - m^2); the direction is uniform on the
    sphere.  The result is then boosted with the parent's own velocity.

    Returns two (n, 4) arrays of (px, py, pz, E).
    """
    n = parent.shape[0]
    p_star2 = 0.25 * m_A ** 2 - m_daughter ** 2
    if p_star2 <= 0:
        raise ValueError("m_A = %g GeV is below the %g GeV pair threshold"
                         % (m_A, 2 * m_daughter))
    p_star = np.sqrt(p_star2)
    e_star = 0.5 * m_A

    cos_theta = 2.0 * rng.random(n) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = 2 * np.pi * rng.random(n)

    d1 = np.column_stack([p_star * sin_theta * np.cos(phi),
                          p_star * sin_theta * np.sin(phi),
                          p_star * cos_theta,
                          e_star * np.ones(n)])
    d2 = np.column_stack([-d1[:, 0], -d1[:, 1], -d1[:, 2], e_star * np.ones(n)])

    return boost(d1, parent, m_A), boost(d2, parent, m_A)


def boost(p_rest, parent, m_parent):
    """Boost `p_rest`, given in the parent rest frame, into the lab frame.

    The standard general Lorentz boost with beta = p_parent / E_parent:

        E   = gamma (E* + beta . p*)
        p   = p* + [ (gamma-1) (beta . p*) / |beta|^2 + gamma E* ] beta
    """
    E_par = parent[:, 3]
    beta = parent[:, :3] / E_par[:, None]
    beta2 = np.sum(beta * beta, axis=1)
    gamma = E_par / m_parent

    bdotp = np.sum(beta * p_rest[:, :3], axis=1)

    # beta2 -> 0 for a parent at rest; the bracket then reduces to gamma E*
    safe_beta2 = np.where(beta2 > 0, beta2, 1.0)
    coeff = np.where(beta2 > 0, (gamma - 1.0) * bdotp / safe_beta2, 0.0)
    coeff = coeff + gamma * p_rest[:, 3]

    p_lab = p_rest[:, :3] + coeff[:, None] * beta
    E_lab = gamma * (p_rest[:, 3] + bdotp)
    return np.column_stack([p_lab, E_lab])


def force_monoenergetic(p4, momentum, m_daughter):
    """Rescale each three-momentum to `momentum`, keeping the direction.

    This is the 'monoenergetic' model: the standard cosmic muon generators throw
    muons at one fixed momentum, so to compare against those samples the
    boost-induced momentum spread has to be removed.  EarthShine's dm_generation_tools scales
    the whole four-vector, which puts the muon off its mass shell; the energy is
    recomputed here instead, because an LHE record with an inconsistent mass is
    rejected downstream.  The two differ by O(m_mu^2/p^2), i.e. parts in 10^8 at
    these momenta.
    """
    p3 = p4[:, :3]
    mag = np.sqrt(np.sum(p3 * p3, axis=1))
    scaled = p3 * (momentum / mag)[:, None]
    energy = np.sqrt(np.sum(scaled * scaled, axis=1) + m_daughter ** 2)
    return np.column_stack([scaled, energy])


def opening_angle(p1, p2):
    """Angle between two three-momenta [rad]."""
    a, b = p1[:, :3], p2[:, :3]
    cos = (np.sum(a * b, axis=1)
           / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def invariant_mass(p1, p2):
    """Invariant mass of a pair of four-vectors [GeV]."""
    tot = p1 + p2
    m2 = tot[:, 3] ** 2 - np.sum(tot[:, :3] ** 2, axis=1)
    return np.sqrt(np.maximum(m2, 0.0))


def momentum_magnitude(p4):
    """|p| of each four-vector."""
    return np.sqrt(np.sum(p4[:, :3] ** 2, axis=1))


def muon_mass():
    return k.MUON_MASS
