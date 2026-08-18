"""The Earth model: PREM shells, escape velocities, elemental number densities
and the dark matter velocity distribution.

This is a numpy port of DarkCapPy/Configure/PlanetData.py.  The original reads
its two csv files with pandas.  The generator is meant to run wherever numpy
and scipy do and nowhere else, so the readers here are hand-rolled.

The module builds its tables once, on first use, and caches them.
"""

import os

import numpy as np

from . import constants as k

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data')


def _read_prem(path):
    """Read PREM500_Mod.csv.

    Eight '#' lines, then a whitespace-separated header, then the shells.  The
    columns after the first six are elemental mass fractions, and the final
    'Intentionally_Blank' column is discarded -- the same slice
    (columns[6:-1]) that DarkCapPy takes.  Non-numeric entries ('None' in the
    Temp/Rho/Pres/Lumi columns) are read as NaN and never used.
    """
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip()]
    header = None
    rows = []
    for ln in lines:
        if ln.lstrip().startswith('#'):
            continue
        if header is None:
            header = ln.split()
            continue
        rows.append(ln.split())

    ncol = len(header)
    data = np.full((len(rows), ncol), np.nan)
    for i, row in enumerate(rows):
        for j in range(min(ncol, len(row))):
            try:
                data[i, j] = float(row[j])
            except ValueError:
                pass  # 'None' -> NaN

    columns = {name: data[:, j] for j, name in enumerate(header)}
    elements = list(header[6:-1])
    return columns, elements


def _read_veldist(path):
    """Read EarthDMVelDist.csv (a plain comma-separated file with a header)."""
    with open(path) as fh:
        header = fh.readline().strip().split(',')
        rows = [ln.strip().split(',') for ln in fh if ln.strip()]
    data = np.array([[float(v) for v in row] for row in rows])
    return {name: data[:, j] for j, name in enumerate(header)}


def _shell_thickness(radius):
    """deltaR[i] = radius[i] - radius[i-1], with deltaR[0] = radius[0].

    Matches DarkCapPy's deltaR_Func, which prepends a zero to the shifted list.
    """
    delta = np.empty_like(radius)
    delta[0] = radius[0]
    delta[1:] = radius[1:] - radius[:-1]
    return delta


def _shell_mass(enclosed_mass):
    """Mass of each shell from the enclosed-mass profile; the first shell is 0."""
    shell = np.empty_like(enclosed_mass)
    shell[0] = 0.0
    shell[1:] = enclosed_mass[1:] - enclosed_mass[:-1]
    return shell


def _shell_density(shell_mass, radius, delta_r):
    """rho[i] = m[i] / (4 pi r[i]^2 dr[i]); the r=0 kludge copies shell 1."""
    density = shell_mass / (4 * np.pi * radius ** 2 * delta_r)
    density[0] = density[1]
    return density


def _escape_velocity_squared(enclosed_mass, radius, delta_r):
    """v_esc^2(r)/c^2 for every shell.

    DarkCapPy sums from the shell outwards,
        v^2(i) = 2G/c^2 * ( sum_{j>=i} M(j) dr(j) / r(j)^2  +  M_tot/R_tot ),
    with v^2(0) set equal to v^2(1).  Written here as a reversed cumulative
    sum instead of the original O(N^2) double loop.
    """
    summand = enclosed_mass * delta_r / radius ** 2
    tail = np.cumsum(summand[::-1])[::-1]          # tail[i] = sum_{j>=i}
    constant = enclosed_mass.max() / radius.max()
    factor = 2.0 * k.G_NEWTON / k.C_LIGHT ** 2
    vesc2 = factor * (tail + constant)
    vesc2[0] = vesc2[1]                            # the r=0 shell copies its neighbour
    return vesc2


class Planet(object):
    """The tabulated Earth used by the capture-rate integrals."""

    def __init__(self, prem_path=None, veldist_path=None):
        prem_path = prem_path or os.path.join(DATA_DIR, 'PREM500_Mod.csv')
        veldist_path = veldist_path or os.path.join(DATA_DIR, 'EarthDMVelDist.csv')

        columns, elements = _read_prem(prem_path)
        self.elements = elements
        self.radius = columns['Radius'] * k.PLANET_RADIUS          # cm
        self.enclosed_mass = columns['Mass'] * k.PLANET_MASS       # g
        self.mass_fraction = {e: columns[e] for e in elements}

        self.delta_r = _shell_thickness(self.radius)
        self.shell_mass = _shell_mass(self.enclosed_mass)
        self.shell_density = _shell_density(self.shell_mass, self.radius,
                                            self.delta_r)
        self.esc_vel2 = _escape_velocity_squared(self.enclosed_mass,
                                                 self.radius, self.delta_r)

        vel = _read_veldist(veldist_path)
        self.vel_range = vel['Velocity_Range']
        self.vel_dist = vel['VelocityDist_Planet_Frame']

        self._num_density = {}

    def number_density(self, element):
        """Number density of `element` in each shell, in GeV^3."""
        if element not in self._num_density:
            mf = self.mass_fraction[element]
            self._num_density[element] = (
                mf * k.g2GeV(self.shell_density)
                / k.amu2GeV(k.ATOMIC_NUMBERS[element]))
        return self._num_density[element]

    def f_cross(self, u):
        """The dark matter speed distribution in the planet frame, interpolated.

        Linear interpolation on the tabulated grid, matching DarkCapPy's
        scipy.interpolate.interp1d(kind='linear').  Values outside the table
        are clamped to zero rather than raising, so integrands can be evaluated
        on a grid that runs past the last tabulated point.
        """
        return np.interp(u, self.vel_range, self.vel_dist, left=0.0, right=0.0)


_PLANET = None


def earth():
    """The shared Earth instance (built on first call)."""
    global _PLANET
    if _PLANET is None:
        _PLANET = Planet()
    return _PLANET


def branching_ratio_table(final_state='mumu'):
    """Read a branching-ratio csv and return (m_A grid [GeV], BR).

    'mumu' reads the dimuon curve digitised from Feng et al.; 'ee' reads the
    electron curve that ships with DarkCapPy.
    """
    name = {'mumu': 'br_mumu.csv', 'ee': 'br_ee.csv'}[final_state]
    with open(os.path.join(DATA_DIR, name)) as fh:
        fh.readline()                       # header, column names differ per file
        rows = [ln.strip().split(',') for ln in fh if ln.strip()]
    data = np.array([[float(v) for v in row] for row in rows])
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def branching_ratio(m_A, final_state='mumu'):
    """BR(A' -> final state) at dark photon mass m_A [GeV].

    Below the threshold of the tabulated grid the branching ratio is zero; for
    the dimuon channel that threshold is 2 m_mu = 0.211 GeV, and the digitised
    curve starts at 0.220 GeV.  Above the last tabulated point the last value
    is held.
    """
    x, y = branching_ratio_table(final_state)
    m_A = np.asarray(m_A, dtype=float)
    br = np.interp(m_A, x, y, left=0.0, right=y[-1])
    return np.where(m_A < x[0], 0.0, br)
