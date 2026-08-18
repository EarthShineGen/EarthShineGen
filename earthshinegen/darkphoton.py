"""Dark photon capture, annihilation and decay.

A numpy port of DarkCapPy/DarkPhoton.py (Green, Rentala et al., implementing
Feng, Smolinsky and Tanedo, arXiv:1509.07525).  Everything here is the "cross
section" half of EarthShineGen: it turns the dark sector parameters
(m_X, m_A, epsilon, alpha_X) into an absolute rate of muon pairs arriving in
the generation volume, which the kinematics half then distributes over events.

Two things differ from the original, both deliberate:

  * kappa_0 is evaluated with the recoil-energy integral done in closed form,
    which turns DarkCapPy's ~5000 two-dimensional quadratures per mass point
    (minutes) into one vectorised one-dimensional quadrature (milliseconds).
    Set method='dblquad' to fall back on the original for validation; the two
    agree to better than a part in 10^3 (see test/validate_against_darkcappy.py).

  * decay_shell_fraction() takes the shell boundaries explicitly, so the
    generation volume below the detector can be used instead of DarkCapPy's
    hard-wired 1 km slab sitting just *above* the Earth's surface.
"""

import numpy as np
from scipy import integrate

from . import constants as k
from .planet import earth


################################################################
# Nuclear form factor and kinematics
################################################################

def form_factor_energy(element):
    """The form-factor scale E_N = 0.114 / A^(5/3) [GeV]."""
    return 0.114 / (k.ATOMIC_NUMBERS[element] ** (5.0 / 3.0))


def form_factor2(element, E_R):
    """|F_N(E_R)|^2 = exp(-E_R / E_N), with E_R the nuclear recoil energy [GeV]."""
    return np.exp(-np.asarray(E_R) / form_factor_energy(element))


def e_min(u, m_X):
    """Minimum recoil energy for gravitational capture [GeV]."""
    return 0.5 * m_X * np.asarray(u) ** 2


def e_max(element, m_X, esc_vel2, u):
    """Maximum recoil energy allowed by the kinematics [GeV].

    `esc_vel2` is v_esc^2/c^2 at the shell in question; it broadcasts against u.
    """
    m_N = k.amu2GeV(k.ATOMIC_NUMBERS[element])
    mu = m_N * m_X / (m_N + m_X)
    return 2 * mu ** 2 * (np.asarray(u) ** 2 + np.asarray(esc_vel2)) / m_N


def u_intersection(element, m_X, esc_vel2):
    """The speed at which e_min and e_max cross; capture is impossible above it."""
    m_N = k.amu2GeV(k.ATOMIC_NUMBERS[element])
    mu = m_N * m_X / (m_N + m_X)
    A = m_X / 2.0
    B = 2.0 * mu ** 2 / m_N
    return np.sqrt(B / (A - B)) * np.sqrt(np.asarray(esc_vel2))


################################################################
# kappa_0: the m_A- and epsilon-independent part of the capture rate
################################################################

def _kappa0_shell_integrals_fast(element, m_X, planet, n_u=4000):
    """The (u, E_R) integral of the kappa_0 integrand, for every shell at once.

    The integrand is

        sigma_0(E_R) * u * f(u)   with   sigma_0(E_R) = exp(-E_R / E_N),

    so the recoil-energy integral between e_min(u) and e_max(u, r) is available
    in closed form,

        E_N * [ exp(-e_min/E_N) - exp(-e_max/E_N) ],

    leaving a one-dimensional integral over the incident speed u.  That one is
    done on a fixed logarithmic-free grid with the trapezoid rule; f(u) is
    piecewise linear on its own 492-point table, so the grid below is refined
    enough that the discretisation error is far under the uncertainty on the
    tabulated distribution itself.

    Returns an array with one entry per shell.
    """
    E_N = form_factor_energy(element)
    esc2 = planet.esc_vel2                                   # (n_shell,)

    u_int = u_intersection(element, m_X, esc2)               # (n_shell,)
    u_high = np.minimum(u_int, k.V_GAL)

    # One shared grid in the *scaled* variable s = u / u_high, so every shell
    # gets the same relative sampling of its own integration range.
    s = np.linspace(0.0, 1.0, n_u)
    u = u_high[:, None] * s[None, :]                         # (n_shell, n_u)

    emin = 0.5 * m_X * u ** 2
    m_N = k.amu2GeV(k.ATOMIC_NUMBERS[element])
    mu = m_N * m_X / (m_N + m_X)
    emax = 2 * mu ** 2 * (u ** 2 + esc2[:, None]) / m_N

    inner = E_N * (np.exp(-emin / E_N) - np.exp(-emax / E_N))
    inner = np.where(emax > emin, inner, 0.0)

    integrand = inner * u * planet.f_cross(u)
    return np.trapz(integrand, u, axis=1)


def _kappa0_shell_integrals_dblquad(element, m_X, planet):
    """The same quantity via DarkCapPy's original two-dimensional quadrature.

    Kept for validation only -- it is some four orders of magnitude slower.
    """
    E_N = form_factor_energy(element)

    def integrand(E_R, u):
        return np.exp(-E_R / E_N) * u * planet.f_cross(u)

    out = np.zeros(len(planet.radius))
    for i in range(len(planet.radius)):
        u_high = min(u_intersection(element, m_X, planet.esc_vel2[i]), k.V_GAL)
        out[i] = integrate.dblquad(
            integrand, 0, u_high,
            lambda u: e_min(u, m_X),
            lambda u: e_max(element, m_X, planet.esc_vel2[i], u))[0]
    return out


def single_element_kappa0(element, m_X, alpha=k.ALPHA_EM, planet=None,
                          method='fast'):
    """kappa_0 for one element [GeV^5]."""
    planet = planet or earth()
    if method == 'fast':
        shell = _kappa0_shell_integrals_fast(element, m_X, planet)
    elif method == 'dblquad':
        shell = _kappa0_shell_integrals_dblquad(element, m_X, planet)
    else:
        raise ValueError("method must be 'fast' or 'dblquad', got %r" % method)

    radial_sum = np.sum(planet.number_density(element) * planet.radius ** 2
                        * shell * planet.delta_r)

    Z_N = k.N_PROTONS[element]
    m_N = k.amu2GeV(k.ATOMIC_NUMBERS[element])
    n_X = k.RHO_DM_LOCAL / m_X                       # GeV/cm^3 / GeV
    conversion = (5.06e13) ** -3 * (1.52e24)         # (cm^-3)(GeV^-2) -> s^-1
    cross_section_factors = 2 * (4 * np.pi) * alpha * Z_N ** 2 * m_N
    prefactor = (4 * np.pi) ** 2

    return n_X * conversion * prefactor * cross_section_factors * radial_sum


def kappa0(m_X, alpha=k.ALPHA_EM, planet=None, method='fast'):
    """kappa_0(m_X): the whole m_A- and epsilon-independent capture factor.

    The capture rate is then  C_cap = epsilon^2 alpha_X kappa_0 / m_A^4.
    """
    planet = planet or earth()
    return float(sum(single_element_kappa0(e, m_X, alpha, planet, method)
                     for e in planet.elements))


def capture_rate(m_X, m_A, epsilon, alpha_X, kappa_0):
    """Dark matter capture rate in the Earth [s^-1]."""
    return epsilon ** 2 * alpha_X * kappa_0 / m_A ** 4


################################################################
# alpha_X: the dark fine structure constant
################################################################

def alpha_thermal(m_X, m_A):
    """alpha_X fixed by the observed relic abundance through thermal freeze-out."""
    conversion = (5.06e13) ** 3 / (1.52e24)      # cm^3 s -> GeV^-2
    therm_avg_sigma_v = 2.2e-26                  # cm^3/s, arXiv:1602.01465
    r2 = (m_A / m_X) ** 2
    value = (conversion * therm_avg_sigma_v * (m_X ** 2 / np.pi)
             * (1 - 0.5 * r2) ** 2 / (1 - r2) ** 1.5)
    return np.sqrt(value)


def alpha_max(m_X):
    """The largest alpha_X allowed by CMB distortion bounds.

    The parameterisation alpha_X^max = 0.17 (m_X/TeV)^1.61 is the one used for
    the rate tables in the analysis note; it comes from Slatyer's CMB limit as
    quoted by Feng et al.
    """
    return 0.17 * (m_X / 1000.0) ** 1.61


def resolve_alpha_x(spec, m_X, m_A):
    """Turn the card's alpha_X entry ('thermal', 'max' or a number) into a value."""
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == 'thermal':
            return float(alpha_thermal(m_X, m_A)), 'thermal'
        if key == 'max':
            return float(alpha_max(m_X)), 'max'
        return float(spec), 'fixed'
    return float(spec), 'fixed'


################################################################
# Annihilation
################################################################

def v0(m_X):
    """Characteristic dark matter speed at the centre of the Earth (units of c)."""
    return np.sqrt(2 * k.T_CROSS / m_X)


def sigma_v_tree(m_X, m_A, alpha_X):
    """Tree-level annihilation cross section <sigma v> [GeV^-2]."""
    r2 = (m_A / m_X) ** 2
    return np.pi * (alpha_X / m_X) ** 2 * (1 - r2) ** 1.5 / (1 - 0.5 * r2) ** 2


def sommerfeld(v, m_X, m_A, alpha_X):
    """The Sommerfeld enhancement at relative speed v.

    The absolute value inside the cosine is DarkCapPy's kludge for the
    resonance region; it is kept so the numbers match.
    """
    a = v / (2 * alpha_X)
    c = 6 * alpha_X * m_X / (np.pi ** 2 * m_A)
    return (np.pi / a * np.sinh(2 * np.pi * a * c)
            / (np.cosh(2 * np.pi * a * c)
               - np.cos(2 * np.pi * np.abs(np.sqrt(np.abs(c - (a * c) ** 2))))))


def thermal_avg_sommerfeld(m_X, m_A, alpha_X):
    """Sommerfeld enhancement averaged over the Maxwell-Boltzmann core."""
    v_char = v0(m_X)

    def integrand(v):
        prefactor = 4 * np.pi / (2 * np.pi * v_char ** 2) ** 1.5
        return (prefactor * v ** 2 * np.exp(-0.5 * (v / v_char) ** 2)
                * sommerfeld(v, m_X, m_A, alpha_X))

    return integrate.quad(integrand, 0, 10 * v_char)[0]


def annihilation_rate(m_X, sigma_v, therm_avg_somm=1.0):
    """C_ann [s^-1]."""
    prefactor = (k.G_NAT * m_X * k.RHO_CROSS / (3 * k.T_CROSS)) ** 1.5
    return 1.52e24 * prefactor * sigma_v * therm_avg_somm


def equilibrium_time(c_cap, c_ann):
    """The capture/annihilation equilibration time [s]."""
    return 1.0 / np.sqrt(c_cap * c_ann)


def gamma_ann(c_cap, c_ann):
    """Annihilations per second today, from the solved rate equation."""
    tau = equilibrium_time(c_cap, c_ann)
    return 0.5 * c_cap * np.tanh(k.TAU_CROSS / tau) ** 2


def n_accumulated(c_cap):
    """Number of dark matter particles captured over the lifetime of the Earth."""
    return c_cap * k.PLANET_LIFE


################################################################
# Decay
################################################################

def decay_length(m_X, m_A, epsilon, BR):
    """Lab-frame decay length of the dark photon [cm].

    DarkCapPy's decayLength(), reproduced exactly:

        L = R_earth * BR * (3.6e-9/epsilon)^2 * (m_X/m_A) * (1/1000) * (1/m_A)

    The (m_X/m_A) factor is the boost gamma = E_A'/m_A, and the rest is
    gamma-independent, so the expression is  gamma * c * BR / Gamma_ee, i.e.
    the *total* decay length written in terms of the electron partial width.
    The BR argument is therefore BR(A' -> e+ e-), not the branching ratio of
    whatever channel is being generated.

    See total_decay_length() for the wrapper that makes that explicit, and
    rate.py for the 'darkcappy' compatibility convention.
    """
    return (k.R_CROSS * BR * (3.6e-9 / epsilon) ** 2 * (m_X / m_A)
            * (1.0 / 1000) * (1.0 / m_A))


def total_decay_length(m_X, m_A, epsilon, br_ee):
    """Total lab-frame decay length [cm], from the electron branching ratio.

    Convenience alias for decay_length() that names its argument correctly.
    Pass BR(A' -> e+ e-); the survival of the dark photon through the rock is
    governed by the total width, whatever channel is eventually generated.
    """
    return decay_length(m_X, m_A, epsilon, br_ee)


def decay_shell_fraction(L, r_inner, r_outer):
    """Fraction of dark photons that decay between radii r_inner and r_outer [cm].

    exp(-r_inner/L) - exp(-r_outer/L), i.e. the survival probability out to the
    bottom of the shell times the decay probability across it.
    """
    return np.exp(-r_inner / L) - np.exp(-r_outer / L)


def decay_fraction_darkcappy(L, effective_depth=1.0e5):
    """DarkCapPy's epsilonDecay(): the shell [R_earth, R_earth + depth].

    Kept for cross-checking against published DarkCapPy/IceCube numbers.  It
    puts the shell just *outside* the Earth, which for L >> depth is the same
    number as the physical one, and for short decay lengths is not.
    """
    return decay_shell_fraction(L, k.R_CROSS, k.R_CROSS + effective_depth)


################################################################
# The signal rate
################################################################

def flux_rate(gamma_annihilation, area_cm2, decay_fraction, BR,
              n_per_annihilation=2):
    """Rate of visible decays into the generation volume [s^-1].

    Each annihilation makes `n_per_annihilation` dark photons, emitted
    isotropically from the core.  The fraction of them aimed at the generation
    volume is its cross-sectional area over the area of the sphere at the
    Earth's surface; of those, `decay_fraction` decay inside the volume and a
    fraction BR of the decays go to the final state we generate.

    This is the same expression as DarkCapPy's iceCubeSignal(), with the
    IceCube effective area replaced by the area of the rock cylinder under the
    detector and the branching ratio made explicit.
    """
    solid_angle_fraction = area_cm2 / (4 * np.pi * k.R_CROSS ** 2)
    return (n_per_annihilation * gamma_annihilation * solid_angle_fraction
            * decay_fraction * BR)
