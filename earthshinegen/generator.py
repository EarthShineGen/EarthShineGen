"""The event loop.

Throws dark photon decays inside the rock volume under the detector, propagates
the two muons up through the overburden, keeps the ones that reach the detector
and writes them to LHE.  Events are produced in vectorised batches; the loop
keeps throwing until `n_events` have been accepted, so the requested number
always comes out even at parameter points with an acceptance of 10^-5.

Two output stages, selected by the card:

  stage = 'vertex'  the record is the decay itself: vertex at the A' decay
                    point in the rock, muon momenta undegraded.  Exact, and
                    the right thing to feed to an acceptance study.

  stage = 'detector'
                    the record is what arrives: the muon momenta after the
                    energy loss over their path through the rock, at the point
                    where they cross the hand-off cylinder.  That cylinder is
                    the detector's own outer surface by default, which makes
                    this a drop-in replacement for a fixed-momentum cosmic muon
                    generator whose vertices sit on the same target surface.

    A caveat that the format forces: the two muons cross the hand-off surface
    at two different points, and LHE has one vertex per event.  The event
    vertex is written at the midpoint of the two crossings and both crossings
    are written as comment lines (see lhe.py), so a consumer that wants the
    true two-vertex topology has the numbers and one that does not gets a
    sensible single vertex.  At the opening angles involved the two crossings
    are centimetres apart, but the analysis is a dimuon analysis, so the
    numbers are kept rather than thrown away.
"""

import sys
import time

import numpy as np

from . import card as card_mod
from . import constants as k
from . import eloss
from . import geometry as geo
from . import kinematics as kin
from . import rate as rate_mod
from .lhe import LHEWriter

BATCH = 20000


class Counters(object):
    """Bookkeeping for the acceptance, broken down by cut."""

    NAMES = ('thrown', 'decay_ok', 'reached_handoff', 'survived_eloss',
             'hit_requirement', 'momentum_cut', 'accepted')

    def __init__(self):
        for name in self.NAMES:
            setattr(self, name, 0)

    def as_dict(self):
        return dict((n, getattr(self, n)) for n in self.NAMES)

    def report(self):
        lines = ['Cut flow (per thrown dark photon decay):']
        base = max(self.thrown, 1)
        for name in self.NAMES:
            value = getattr(self, name)
            lines.append('  %-20s %12d  %10.6f' % (name, value, value / base))
        return '\n'.join(lines)


def throw_batch(n, p, rate_info, rng):
    """Generate one batch of decays and return everything the selection needs.

    Returns a dict of arrays, all of length n, plus the boolean masks.
    """
    m_X, m_A = p['m_X'], p['m_A']

    # --- decay points -----------------------------------------------------
    if p['depth_sampling'] == 'exponential':
        depths = geo.sample_depths_exponential(
            n, p['depth_min'], p['depth_max'], rate_info['decay_length_cm'], rng)
    else:
        depths = geo.sample_depths_uniform(n, p['depth_min'], p['depth_max'], rng)

    origins = geo.sample_origins(n, rate_info['disk_radius_m'], depths, rng)

    # --- the decay --------------------------------------------------------
    parents = kin.dark_photon_momenta(m_X, m_A, n, p['dm_model'], rng)
    mu1, mu2 = kin.two_body_decay(parents, m_A, k.MUON_MASS, rng)

    if p['dm_model'] == 'monoenergetic':
        mu1 = kin.force_monoenergetic(mu1, 0.5 * m_X, k.MUON_MASS)
        mu2 = kin.force_monoenergetic(mu2, 0.5 * m_X, k.MUON_MASS)

    return {'origins': origins, 'parents': parents, 'mu1': mu1, 'mu2': mu2}


def select_batch(batch, p, counters):
    """Propagate and apply the selection to one batch.

    Returns a dict of the per-event quantities for the accepted events.
    """
    origins = batch['origins']
    n = origins.shape[0]
    counters.thrown += n
    counters.decay_ok += n

    handoff_r, handoff_hl = card_mod.resolve_handoff(p)

    entry = {}
    reached = np.ones(n, dtype=bool)
    for tag, p4 in (('mu1', batch['mu1']), ('mu2', batch['mu2'])):
        pts, _ = geo.ray_cylinder_intersection(origins, p4[:, :3],
                                               handoff_r, handoff_hl)
        entry[tag] = pts
        reached &= ~np.isnan(pts[:, 0])
    counters.reached_handoff += int(reached.sum())

    # Path length through the rock, from the decay point to the hand-off
    # surface.  Muons that never reach it are given a NaN path, which the
    # energy loss turns into a stopped muon; they are already cut anyway.
    degraded = {}
    survived = np.ones(n, dtype=bool)
    for tag in ('mu1', 'mu2'):
        dist = geo.path_length(origins, entry[tag])
        dist = np.where(np.isfinite(dist), dist, 0.0)
        p4, alive = eloss.degrade(batch[tag], dist,
                                  model=p['eloss_model'],
                                  density=p['rock_density'])
        degraded[tag] = p4
        degraded[tag + '_distance'] = dist
        survived &= alive
    survived &= reached
    counters.survived_eloss += int(survived.sum())

    # Geometric requirement, evaluated on the propagated momenta at the
    # hand-off surface: the direction is unchanged by the average energy loss,
    # so the same straight line is used.
    if p['require_hit'] == 'none':
        hit = np.ones(n, dtype=bool)
    else:
        if p['require_hit'] == 'detector':
            radius, half_len = p['detector_radius'], p['detector_half_length']
        else:
            radius = p['inner_detector_radius']
            half_len = p['inner_detector_half_length']
        hits = []
        for tag in ('mu1', 'mu2'):
            pts, _ = geo.ray_cylinder_intersection(origins, batch[tag][:, :3],
                                                   radius, half_len)
            hits.append(~np.isnan(pts[:, 0]))
        hit = (hits[0] & hits[1]) if p['require_both_muons'] else (hits[0] | hits[1])
    hit &= survived
    counters.hit_requirement += int(hit.sum())

    # Momentum on arrival.  require_both_muons decides whether the pair or
    # just one muon has to clear the thresholds, the same convention as the
    # geometric requirement above.
    momentum_ok = {}
    for tag in ('mu1', 'mu2'):
        p3 = degraded[tag][:, :3]
        pmag = np.sqrt(np.sum(p3 * p3, axis=1))
        pt = np.sqrt(p3[:, 0] ** 2 + p3[:, 1] ** 2)
        momentum_ok[tag] = (pmag >= p['min_momentum']) & (pt >= p['min_pt'])

    if p['require_both_muons']:
        kinematic = momentum_ok['mu1'] & momentum_ok['mu2']
    else:
        kinematic = momentum_ok['mu1'] | momentum_ok['mu2']

    passes = hit & kinematic
    counters.momentum_cut += int(passes.sum())
    counters.accepted += int(passes.sum())

    idx = np.nonzero(passes)[0]
    return {
        'origins': origins[idx],
        'mu1_out': (degraded['mu1'] if p['stage'] == 'detector'
                    else batch['mu1'])[idx],
        'mu2_out': (degraded['mu2'] if p['stage'] == 'detector'
                    else batch['mu2'])[idx],
        'mu1_entry': entry['mu1'][idx],
        'mu2_entry': entry['mu2'][idx],
        'mu1_distance': degraded['mu1_distance'][idx],
        'mu2_distance': degraded['mu2_distance'][idx],
    }


def run(p, rate_info, rng, writer, log=sys.stdout):
    """Fill `writer` with p['n_events'] accepted events.  Returns the counters."""
    counters = Counters()
    target = p['n_events']
    max_trials = p['max_trials'] or 0
    start = time.time()
    written = 0
    last_report = start

    while written < target:
        n = BATCH
        batch = throw_batch(n, p, rate_info, rng)
        sel = select_batch(batch, p, counters)

        n_avail = sel['origins'].shape[0]
        n_take = min(n_avail, target - written)
        for i in range(n_take):
            if p['stage'] == 'detector':
                v1 = sel['mu1_entry'][i]
                v2 = sel['mu2_entry'][i]
                vertex = 0.5 * (v1 + v2)
            else:
                vertex = sel['origins'][i]
                v1 = v2 = None
            writer.write_event(sel['mu1_out'][i], sel['mu2_out'][i],
                               vertex_m=vertex, vertex1_m=v1, vertex2_m=v2,
                               decay_vertex_m=sel['origins'][i])
        written += n_take

        if max_trials and counters.thrown >= max_trials:
            log.write('reached max_trials = %d with %d events written\n'
                      % (max_trials, written))
            break

        now = time.time()
        if p['verbose'] and now - last_report > 10:
            log.write('  %d/%d written, %d thrown, %.1f s\n'
                      % (written, target, counters.thrown, now - start))
            log.flush()
            last_report = now

        if counters.thrown > 2e9:
            log.write('WARNING: 2e9 events thrown for %d accepted; the '
                      'acceptance at this point is too small to generate. '
                      'Stopping.\n' % written)
            break

    if p['verbose']:
        log.write('generation took %.1f s\n' % (time.time() - start))
    return counters


def efficiency_with_error(counters):
    """Binomial acceptance and its uncertainty."""
    n = max(counters.thrown, 1)
    eff = counters.accepted / n
    err = np.sqrt(max(eff * (1 - eff), 0.0) / n)
    return eff, err


def generate(p, log=sys.stdout):
    """Top-level entry point: rate, events, report.

    Returns (rate_info, mc_info, counters).
    """
    seed = p['seed']
    if not seed:
        seed = int(np.random.SeedSequence().entropy % 2 ** 31)
        p['seed'] = seed
        log.write('auto-generated seed: %d\n' % seed)
    rng = np.random.default_rng(seed)

    log.write('computing the capture and annihilation rate...\n')
    log.flush()
    t0 = time.time()
    rate_info = rate_mod.compute(p, verbose=p['verbose'])
    log.write('  kappa_0 = %.6g GeV^5   (%s method, %.2f s)\n'
              % (rate_info['kappa_0'], p['kappa_method'], time.time() - t0))
    log.write('  Gamma_ann = %.6g s^-1, L = %.6g cm, '
              'pairs in volume = %.6g s^-1\n'
              % (rate_info['gamma_ann'], rate_info['decay_length_cm'],
                 rate_info['rate_in_volume_per_s']))

    if rate_info['br_mumu'] <= 0:
        raise SystemExit('BR(A\' -> mu mu) is zero at m_A = %g GeV; the dimuon '
                         'channel opens at 2 m_mu = 0.211 GeV and the '
                         'tabulated curve starts at 0.220 GeV.' % p['m_A'])

    if p['dm_model'] != 'core':
        log.write("NOTE: the rate normalisation assumes the Feng et al. core "
                  "model, where the dark matter has thermalised into the "
                  "centre of the Earth.  dm_model = '%s' changes the "
                  "kinematics but NOT the normalisation, which DarkCapPy does "
                  "not provide for a crust distribution.  Treat the quoted "
                  "rate as core-model.\n" % p['dm_model'])

    header = _header(p, rate_info)
    with LHEWriter(p['output_file'], header,
                   xsec_pb=rate_info['rate_in_volume_per_s'] * k.yr2s(1.0),
                   xsec_err_pb=0.0, max_weight=1.0,
                   beam_energy=p['beam_energy'],
                   include_initial=bool(p['include_initial']),
                   include_mother=bool(p['include_mother'])) as writer:
        counters = run(p, rate_info, rng, writer, log=log)
        n_written = writer.n_events

        eff, err = efficiency_with_error(counters)
        mc = rate_mod.observable_rate(rate_info, eff)
        mc.update({'n_thrown': counters.thrown,
                   'n_accepted': counters.accepted,
                   'n_written': n_written, 'efficiency_error': err})

        # XSECUP has no meaning for a beamless signal, but downstream tooling
        # reads it, so it carries the observable rate in events per year of
        # live time -- the number the report quotes -- rather than a
        # placeholder.
        writer.update_cross_section(mc['rate_per_year'],
                                    mc['rate_per_year'] * err
                                    / max(eff, 1e-300))

    log.write('\n%s\n' % counters.report())
    log.write('\nwrote %d events to %s\n' % (n_written, p['output_file']))
    return rate_info, mc, counters


def _header(p, rate_info):
    """The <header> block: enough to reconstruct the run from the file alone."""
    from .lhe import header_block
    fields = dict((key, p[key]) for key in sorted(p))
    fields.update({
        'kappa_0_GeV5': '%.8g' % rate_info['kappa_0'],
        'C_cap_per_s': '%.8g' % rate_info['C_cap'],
        'Gamma_ann_per_s': '%.8g' % rate_info['gamma_ann'],
        'decay_length_cm': '%.8g' % rate_info['decay_length_cm'],
        'BR_A_to_mumu': '%.8g' % rate_info['br_mumu'],
        'f_decay_in_volume': '%.8g' % rate_info['f_decay_shell'],
        'pairs_in_volume_per_s': '%.8g' % rate_info['rate_in_volume_per_s'],
        'disk_radius_m': '%.8g' % rate_info['disk_radius_m'],
    })
    return header_block(fields)
