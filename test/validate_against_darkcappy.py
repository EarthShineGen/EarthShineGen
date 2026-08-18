#!/usr/bin/env python3
"""Check EarthShineGen's rate chain against the DarkCapPy package it ports.

Run it from anywhere, pointing at a DarkCapPy checkout:

    ./test/validate_against_darkcappy.py --darkcappy /path/to/DarkCapPy

It compares, for a few (m_X, m_A, epsilon) points:

  * the Earth model itself (shell radii, escape velocities, number densities)
  * kappa_0, both with EarthShineGen's analytic recoil integral and with the
    original two-dimensional quadrature
  * the Sommerfeld enhancement, <sigma v>, C_ann, the equilibrium time,
    Gamma_ann and the decay length

DarkCapPy needs pandas; EarthShineGen does not.  Run this from an environment
that has both.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from earthshinegen import constants as k          # noqa: E402
from earthshinegen import darkphoton as dp        # noqa: E402
from earthshinegen import planet as pl            # noqa: E402

TOLERANCE = 1e-3          # relative, for everything but the kappa_0 quadrature
KAPPA_TOLERANCE = 5e-3    # the two quadratures are different algorithms

_failures = []


def check(name, ours, theirs, tol=TOLERANCE):
    ours = float(ours)
    theirs = float(theirs)
    denom = abs(theirs) if theirs != 0 else 1.0
    rel = abs(ours - theirs) / denom
    status = 'ok  ' if rel <= tol else 'FAIL'
    if rel > tol:
        _failures.append(name)
    print('  %s %-34s ours %14.7g   DarkCapPy %14.7g   rel %8.2e'
          % (status, name, ours, theirs, rel))


def check_array(name, ours, theirs, tol=TOLERANCE):
    ours = np.asarray(ours, dtype=float)
    theirs = np.asarray(theirs, dtype=float)
    if ours.shape != theirs.shape:
        _failures.append(name)
        print('  FAIL %-34s shape %s vs %s' % (name, ours.shape, theirs.shape))
        return
    denom = np.where(theirs != 0, np.abs(theirs), 1.0)
    rel = np.max(np.abs(ours - theirs) / denom)
    status = 'ok  ' if rel <= tol else 'FAIL'
    if rel > tol:
        _failures.append(name)
    print('  %s %-34s max rel over %d entries %8.2e'
          % (status, name, ours.size, rel))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--darkcappy',
                    default=os.environ.get('DARKCAPPY_DIR', '../DarkCapPy'),
                    help='path to a DarkCapPy checkout '
                         '(https://github.com/agree019/DarkCapPy); '
                         'defaults to $DARKCAPPY_DIR, else ../DarkCapPy')
    ap.add_argument('--masses', type=float, nargs='+',
                    default=[100.0, 1000.0, 7000.0],
                    help='m_X points to test [GeV]')
    ap.add_argument('--skip-dblquad', action='store_true',
                    help='skip the slow original kappa_0 quadrature')
    args = ap.parse_args()

    sys.path.insert(0, args.darkcappy)
    try:
        import DarkCapPy.DarkPhoton as DP
    except ImportError as exc:
        raise SystemExit('could not import DarkCapPy from %s: %s\n'
                         '(it needs pandas; run this from an environment '
                         'that has it)'
                         % (args.darkcappy, exc))

    earth = pl.earth()

    print('\nEarth model')
    check_array('radius [cm]', earth.radius, np.asarray(DP.radius_List))
    check_array('deltaR [cm]', earth.delta_r, np.asarray(DP.deltaR_List))
    check_array('escape velocity^2', earth.esc_vel2, np.asarray(DP.escVel2_List))
    print('  ok   elements                          %s' % ', '.join(earth.elements))
    if list(earth.elements) != list(DP.element_List):
        _failures.append('element list')
        print('  FAIL element list                   %s vs %s'
              % (earth.elements, list(DP.element_List)))
    for element in earth.elements:
        check_array('n(%s) [GeV^3]' % element, earth.number_density(element),
                    np.asarray(DP.numDensity_Func(element)))

    print('\nVelocity distribution')
    u_test = np.linspace(0.0, 0.9 * earth.vel_range.max(), 25)
    check_array('f_cross(u)', earth.f_cross(u_test),
                np.asarray([DP.fCrossInterp(u) for u in u_test]))

    for m_X in args.masses:
        print('\nm_X = %g GeV' % m_X)

        ours_fast = dp.kappa0(m_X, method='fast')
        if not args.skip_dblquad:
            print('  (running the original quadrature, this takes a minute)')
            theirs = DP.kappa_0(m_X, k.ALPHA_EM)
            check('kappa_0', ours_fast, theirs, tol=KAPPA_TOLERANCE)
        else:
            theirs = None

        for m_A in (0.23, 0.5):
            print('  m_A = %g GeV' % m_A)
            alpha_th = dp.alpha_thermal(m_X, m_A)
            check('  alpha_thermal', alpha_th, DP.alphaTherm(m_X, m_A))

            for alpha_X, label in ((float(alpha_th), 'thermal'),
                                   (float(dp.alpha_max(m_X)), 'max')):
                somm = dp.thermal_avg_sommerfeld(m_X, m_A, alpha_X)
                check('  Sommerfeld (%s)' % label, somm,
                      DP.thermAvgSommerfeld(m_X, m_A, alpha_X))

                sv = dp.sigma_v_tree(m_X, m_A, alpha_X)
                check('  <sigma v> (%s)' % label, sv,
                      DP.sigmaVtree(m_X, m_A, alpha_X))

                c_ann = dp.annihilation_rate(m_X, sv, somm)
                check('  C_ann (%s)' % label, c_ann,
                      DP.cAnn(m_X, DP.sigmaVtree(m_X, m_A, alpha_X),
                              DP.thermAvgSommerfeld(m_X, m_A, alpha_X)))

                kappa = ours_fast if theirs is None else theirs
                eps = 1e-8
                c_cap = dp.capture_rate(m_X, m_A, eps, alpha_X, kappa)
                check('  C_cap (%s)' % label, c_cap,
                      DP.cCapQuick(m_X, m_A, eps, alpha_X, kappa))
                check('  tau_eq (%s)' % label,
                      dp.equilibrium_time(c_cap, c_ann),
                      DP.tau(c_cap, c_ann))
                check('  Gamma_ann (%s)' % label,
                      dp.gamma_ann(c_cap, c_ann), DP.gammaAnn(c_cap, c_ann))

            check('  decay length (BR=1)',
                  dp.decay_length(m_X, m_A, 1e-8, 1.0),
                  DP.decayLength(m_X, m_A, 1e-8, 1.0))
            check('  epsilonDecay (1 km)',
                  dp.decay_fraction_darkcappy(dp.decay_length(m_X, m_A, 1e-8, 1.0)),
                  DP.epsilonDecay(DP.decayLength(m_X, m_A, 1e-8, 1.0)))

    print('')
    if _failures:
        print('%d comparison(s) FAILED: %s' % (len(_failures),
                                               ', '.join(sorted(set(_failures)))))
        return 1
    print('all comparisons agree within tolerance')
    return 0


if __name__ == '__main__':
    sys.exit(main())
