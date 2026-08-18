"""The absolute rate: everything DarkCapPy contributes to an EarthShineGen run.

One call to compute() turns the dark sector parameters on the card into the
number of muon pairs per second decaying inside the generation volume.  The
Monte Carlo then measures what fraction of those reach the detector, and the
two are multiplied at the end of the run.

The chain is

    kappa_0(m_X)                        capture, stripped of m_A and epsilon
    C_cap  = eps^2 alpha_X kappa_0/m_A^4    captures per second
    C_ann                                annihilations, with Sommerfeld
    Gamma_ann = C_cap tanh^2(t/tau) / 2   annihilations per second today
    L                                    dark photon decay length
    f_shell                              fraction decaying in the volume
    BR(A' -> mu mu)                      visible fraction
    R = 2 Gamma_ann (A/4 pi R_E^2) f_shell BR     pairs per second in the volume
"""

import numpy as np

from . import card
from . import constants as k
from . import darkphoton as dp
from . import geometry as geo
from .planet import branching_ratio


def compute(p, verbose=False):
    """Evaluate the rate chain for a resolved parameter dict.

    Returns a dict of every intermediate quantity; nothing is printed here.
    """
    m_X, m_A, eps = p['m_X'], p['m_A'], p['epsilon']

    alpha_X, alpha_X_mode = dp.resolve_alpha_x(p['alpha_X'], m_X, m_A)

    br_mumu = float(branching_ratio(m_A, 'mumu'))
    br_ee = float(branching_ratio(m_A, 'ee'))

    # Capture
    kappa_0 = dp.kappa0(m_X, alpha=p['alpha_em'], method=p['kappa_method'])
    c_cap = dp.capture_rate(m_X, m_A, eps, alpha_X, kappa_0)
    n_captured = dp.n_accumulated(c_cap)

    # Annihilation
    somm = dp.thermal_avg_sommerfeld(m_X, m_A, alpha_X)
    sigma_v = dp.sigma_v_tree(m_X, m_A, alpha_X)
    c_ann = dp.annihilation_rate(m_X, sigma_v, somm)
    tau_eq = dp.equilibrium_time(c_cap, c_ann)
    gamma = dp.gamma_ann(c_cap, c_ann)

    # Decay.  In the 'total' convention the width is set by the electron
    # channel (which is what DarkCapPy's formula is normalised to) and the
    # dimuon branching ratio scales the yield.  In the 'darkcappy' convention
    # the dimuon branching ratio goes into the decay length instead and the
    # yield carries no explicit factor, reproducing the rate notebook.
    if p['decay_length_convention'] == 'total':
        L = dp.total_decay_length(m_X, m_A, eps, br_ee)
        yield_br = br_mumu
    else:
        L = dp.decay_length(m_X, m_A, eps, br_mumu)
        yield_br = 1.0

    disk_radius = resolve_disk_radius(p)
    depth_span_cm = abs(p['depth_max'] - p['depth_min']) * geo.CM_PER_M
    r_outer = k.R_CROSS - abs(p['depth_max']) * geo.CM_PER_M
    r_inner = k.R_CROSS - abs(p['depth_min']) * geo.CM_PER_M

    f_shell = float(dp.decay_shell_fraction(L, r_inner, r_outer))
    f_shell_darkcappy = float(dp.decay_fraction_darkcappy(L, depth_span_cm))

    area_cm2 = geo.cross_section_area_cm2(disk_radius)
    volume = geo.volume_m3(p['depth_min'], p['depth_max'], disk_radius)

    rate_in_volume = dp.flux_rate(gamma, area_cm2, f_shell, yield_br)

    handoff_radius, handoff_half_length = card.resolve_handoff(p)

    return {
        'm_X': m_X, 'm_A': m_A, 'epsilon': eps,
        'alpha_X': alpha_X, 'alpha_X_mode': alpha_X_mode,
        'alpha_em': p['alpha_em'],
        'br_mumu': br_mumu, 'br_ee': br_ee,
        'kappa_0': kappa_0,
        'C_cap': c_cap, 'n_captured': n_captured,
        'sommerfeld': somm, 'sigma_v_tree': sigma_v, 'C_ann': c_ann,
        'tau_eq': tau_eq, 'gamma_ann': gamma,
        'decay_length_cm': L,
        'decay_length_convention': p['decay_length_convention'],
        'f_decay_shell': f_shell,
        'f_decay_shell_darkcappy': f_shell_darkcappy,
        'yield_br': yield_br,
        'disk_radius_m': disk_radius,
        'area_cm2': area_cm2,
        'volume_m3': volume,
        'detector_radius_m': p['detector_radius'],
        'detector_half_length_m': p['detector_half_length'],
        'inner_detector_radius_m': p['inner_detector_radius'],
        'inner_detector_half_length_m': p['inner_detector_half_length'],
        'handoff_radius_m': handoff_radius,
        'handoff_half_length_m': handoff_half_length,
        'require_hit': p['require_hit'],
        'rate_in_volume_per_s': rate_in_volume,
        'livetime_years': p['livetime_years'],
    }


def resolve_disk_radius(p):
    """The generation cylinder radius, resolving 'auto'."""
    spec = p['disk_radius']
    if isinstance(spec, str) and spec.strip().lower() == 'auto':
        return float(geo.auto_disk_radius(p['depth_min']))
    return float(spec)


def observable_rate(rate_info, efficiency):
    """Fold the Monte Carlo acceptance into the rate.

    Returns per second, per year and per the card's live time.
    """
    per_second = rate_info['rate_in_volume_per_s'] * efficiency
    per_year = per_second * k.yr2s(1.0)
    return {
        'efficiency': efficiency,
        'rate_per_s': per_second,
        'rate_per_month': per_second * k.yr2s(1.0) / 12.0,
        'rate_per_year': per_year,
        'n_expected': per_year * rate_info['livetime_years'],
    }


def equivalent_cross_section_pb(n_expected, luminosity_fb):
    """A picobarn number for the LHE <init> block.

    The signal has no beam and no luminosity, so XSECUP has no physical
    meaning here.  Downstream tooling still reads it, so a self-consistent
    value is written: the one that reproduces the expected event count against
    a stated reference luminosity.  Quote the rate from the report, not this.
    """
    if luminosity_fb <= 0:
        return 0.0
    return n_expected / (luminosity_fb * 1000.0)


def format_report(rate_info, mc=None, params=None):
    """The human-readable rate report written next to the LHE file."""
    r = rate_info
    out = []
    add = out.append

    add('=' * 74)
    add('EarthShineGen rate report')
    add('=' * 74)
    add('')
    add('Dark sector')
    add('  m_X                          %14.6g GeV' % r['m_X'])
    add('  m_A                          %14.6g GeV' % r['m_A'])
    add('  epsilon                      %14.6g' % r['epsilon'])
    add('  alpha_X                      %14.6g   (%s)'
        % (r['alpha_X'], r['alpha_X_mode']))
    add('  BR(A'"'"' -> mu mu)             %14.6g' % r['br_mumu'])
    add('  BR(A'"'"' -> e e)               %14.6g' % r['br_ee'])
    add('')
    add('Capture and annihilation')
    add('  kappa_0                      %14.6g GeV^5' % r['kappa_0'])
    add('  C_cap                        %14.6g s^-1' % r['C_cap'])
    add('  captured over 4.5 Gyr        %14.6g particles' % r['n_captured'])
    add('  <sigma v> tree               %14.6g GeV^-2' % r['sigma_v_tree'])
    add('  Sommerfeld (thermal avg)     %14.6g' % r['sommerfeld'])
    add('  C_ann                        %14.6g s^-1' % r['C_ann'])
    add('  equilibration time           %14.6g s   (age of Earth %.3g s)'
        % (r['tau_eq'], k.TAU_CROSS))
    add('  Gamma_ann                    %14.6g s^-1' % r['gamma_ann'])
    add('')
    add('Detector')
    add('  outer cylinder               %14.6g m radius, %.6g m half length'
        % (r['detector_radius_m'], r['detector_half_length_m']))
    add('  inner cylinder               %14.6g m radius, %.6g m half length'
        % (r['inner_detector_radius_m'], r['inner_detector_half_length_m']))
    add('  hand-off surface             %14.6g m radius, %.6g m half length'
        % (r['handoff_radius_m'], r['handoff_half_length_m']))
    add('  acceptance requirement       %14s' % r['require_hit'])
    add('')
    add('Decay and geometry')
    add('  decay length L               %14.6g cm  (%.4g km)'
        % (r['decay_length_cm'], r['decay_length_cm'] / 1e5))
    add('  convention                   %14s' % r['decay_length_convention'])
    add('  generation cylinder radius   %14.6g m' % r['disk_radius_m'])
    add('  generation volume            %14.6g m^3  (%.4g km^3)'
        % (r['volume_m3'], r['volume_m3'] / 1e9))
    add('  intercepting area            %14.6g cm^2 (%.4g km^2)'
        % (r['area_cm2'], r['area_cm2'] / 1e10))
    add('  fraction decaying in volume  %14.6g' % r['f_decay_shell'])
    add('  (DarkCapPy epsilonDecay)     %14.6g' % r['f_decay_shell_darkcappy'])
    add('')
    add('Rate into the generation volume')
    add('  muon pairs                   %14.6g s^-1' % r['rate_in_volume_per_s'])
    add('                               %14.6g yr^-1'
        % (r['rate_in_volume_per_s'] * k.yr2s(1.0)))

    if mc is not None:
        add('')
        add('Monte Carlo acceptance')
        add('  thrown                       %14d' % mc['n_thrown'])
        add('  accepted                     %14d' % mc['n_accepted'])
        add('  efficiency                   %14.6g +/- %.3g'
            % (mc['efficiency'], mc['efficiency_error']))
        add('')
        add('Observable rate at the detector')
        add('  per second                   %14.6g' % mc['rate_per_s'])
        add('  per month                    %14.6g' % mc['rate_per_month'])
        add('  per year                     %14.6g' % mc['rate_per_year'])
        add('  in %.4g yr of live time      %14.6g events'
            % (r['livetime_years'], mc['n_expected']))

    if params is not None:
        add('')
        add('Run configuration')
        for key in sorted(params):
            add('  %-28s %s' % (key, params[key]))

    add('')
    add('=' * 74)
    return '\n'.join(out) + '\n'
