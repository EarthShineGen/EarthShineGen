"""Physical constants and unit conversions.

Ported from DarkCapPy/Configure/Constants.py and Conversions.py.  The numerical
values are unchanged, so EarthShineGen reproduces DarkCapPy rate for rate; only
the pandas dependency is gone.

Unless stated otherwise, masses are in GeV, lengths in cm, times in seconds and
velocities are dimensionless (units of c).
"""

import numpy as np

################################################################
# Unit conversions
################################################################


def amu2GeV(x):
    """Atomic mass units -> GeV."""
    return 0.938272 * x


def amu2g(x):
    """Atomic mass units -> grams."""
    return 1.66053892e-24 * x


def GeV2s(x):
    """Inverse GeV -> seconds."""
    return 1.52e24 * x


def GeV2cm(x):
    """Inverse GeV -> cm."""
    return 5.06e13 * x


def g2GeV(x):
    """Grams -> GeV."""
    return 5.62e23 * x


def yr2s(x):
    """Years -> seconds."""
    return 3.1536000e7 * x


################################################################
# Speeds and the galactic dark matter halo
################################################################

C_LIGHT = 3.0e10               # cm/s
G_NEWTON = 6.674e-11 * 100 ** 3 * 1000 ** -1   # cm^3 / (g s^2)

V_DOT = 220.0e5 / C_LIGHT      # Sun relative to the galactic centre
V_CROSS = 29.8e5 / C_LIGHT     # Earth relative to the Sun
V_GAL = 550.0e5 / C_LIGHT      # galactic escape velocity
U_0 = 245.0e5 / C_LIGHT        # characteristic speed of galactic DM
KURTOSIS = 2.5

RHO_DM_LOCAL = 0.3             # GeV/cm^3, local dark matter density

################################################################
# Earth
################################################################

PLANET_RADIUS = 6.371e8        # cm
PLANET_MASS = 5.972e27         # g
PLANET_LIFE = yr2s(4.5e9)      # s, 4.5 Gyr

R_CROSS = PLANET_RADIUS        # cm, alias used by the decay-length formulae
TAU_CROSS = PLANET_LIFE        # s, alias used by the equilibrium formulae

G_NAT = 6.71e-39               # GeV^-2, Newton's constant in natural units
RHO_CROSS = 5.67e-17           # GeV^4, density at the centre of the Earth
T_CROSS = 4.9134e-10           # GeV, temperature at the centre of the Earth

ALPHA_EM = 1.0 / 137.0

################################################################
# Particles
################################################################

MUON_MASS = 0.1056583745       # GeV
ELECTRON_MASS = 0.000510998950  # GeV

MUON_PDGID = 13
DARKPHOTON_PDGID = 4900022     # hidden-valley photon convention

################################################################
# Standard rock (used by the muon propagation)
################################################################

ROCK_DENSITY = 2.65            # g/cm^3
ROCK_A_IONIZATION = 2.0e-3     # GeV / (g/cm^2)
ROCK_B_RADIATIVE = 4.0e-6      # (g/cm^2)^-1
ROCK_X0 = 26.54                # radiation length of standard rock, g/cm^2


################################################################
# Atomic data (only the elements that appear in the PREM table)
################################################################

ATOMIC_NUMBERS = {
    'H1': 1., 'He3': 3., 'He4': 2., 'C12': 12., 'C13': 13.,
    'N14': 14., 'N15': 15., 'O16': 16., 'O17': 17., 'O18': 18.,
    'Ne': 20., 'Na': 23., 'Mg': 24., 'Al': 27., 'Si': 28.,
    'P': 30., 'S': 32., 'Cl': 35., 'Ar': 39., 'K': 39.,
    'Ca': 40., 'Sc': 45., 'Ti': 48., 'V': 51., 'Cr': 52.,
    'Mn': 55., 'Fe': 56., 'Co': 59., 'Ni': 59.,
}

N_PROTONS = {
    'H1': 1., 'He3': 2., 'He4': 2., 'C12': 6., 'C13': 6.,
    'N14': 7., 'N15': 7., 'O16': 8., 'O17': 8., 'O18': 8.,
    'Ne': 10., 'Na': 11., 'Mg': 12., 'Al': 13., 'Si': 14.,
    'P': 15., 'S': 16., 'Cl': 17., 'Ar': 18., 'K': 19.,
    'Ca': 20., 'Sc': 21., 'Ti': 22., 'V': 23., 'Cr': 24.,
    'Mn': 25., 'Fe': 26., 'Co': 27., 'Ni': 28.,
}

__all__ = [n for n in dir() if not n.startswith('_') and n != 'np']
