#!/usr/bin/env python3
"""EarthShineGen self-tests.

These need only numpy and scipy, so they run wherever the generator runs:

    ./test/run_tests.py

They cover the pieces that are new here rather than ported unchanged -- the
analytic two-body decay, the exponential depth sampling, the energy loss, the
geometry and the LHE record -- and finish by generating a small sample in every
model and stage to make sure nothing raises.  The comparison against DarkCapPy
lives in validate_against_darkcappy.py, which needs pandas.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from earthshinegen import constants as k          # noqa: E402
from earthshinegen import darkphoton as dp        # noqa: E402
from earthshinegen import eloss                   # noqa: E402
from earthshinegen import geometry as geo         # noqa: E402
from earthshinegen import kinematics as kin       # noqa: E402
from earthshinegen import planet as pl            # noqa: E402

_results = []


def test(fn):
    _results.append(fn)
    return fn


def approx(a, b, tol=1e-9, label=''):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
    rel = np.max(np.abs(a - b) / denom)
    if rel > tol:
        raise AssertionError('%s: max relative difference %.3e > %.3e'
                             % (label or 'values differ', rel, tol))


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------

@test
def test_two_body_decay_conserves_four_momentum():
    rng = np.random.default_rng(1)
    m_X, m_A = 7000.0, 0.23
    parent = kin.dark_photon_momenta(m_X, m_A, 5000, 'floating', rng)
    d1, d2 = kin.two_body_decay(parent, m_A, k.MUON_MASS, rng)
    approx(d1 + d2, parent, 1e-10, 'daughters do not sum to the parent')


@test
def test_two_body_decay_puts_muons_on_shell():
    rng = np.random.default_rng(2)
    parent = kin.dark_photon_momenta(7000.0, 0.23, 5000, 'core', rng)
    d1, d2 = kin.two_body_decay(parent, 0.23, k.MUON_MASS, rng)
    for d in (d1, d2):
        m2 = d[:, 3] ** 2 - np.sum(d[:, :3] ** 2, axis=1)
        approx(np.sqrt(np.abs(m2)), k.MUON_MASS * np.ones(len(d)), 1e-5,
               'muon is off its mass shell')


@test
def test_pair_invariant_mass_is_m_A():
    rng = np.random.default_rng(3)
    for m_A in (0.23, 0.5, 2.0):
        parent = kin.dark_photon_momenta(7000.0, m_A, 2000, 'floating', rng)
        d1, d2 = kin.two_body_decay(parent, m_A, k.MUON_MASS, rng)
        approx(kin.invariant_mass(d1, d2), m_A * np.ones(2000), 1e-5,
               'reconstructed m_A is wrong')


@test
def test_dark_photon_energy_is_the_dm_mass():
    rng = np.random.default_rng(4)
    m_X = 7000.0
    parent = kin.dark_photon_momenta(m_X, 0.23, 100, 'core', rng)
    approx(parent[:, 3], m_X * np.ones(100), 1e-12, 'E_A prime should be m_X')
    # and straight up
    assert np.all(parent[:, 0] == 0) and np.all(parent[:, 2] == 0), \
        'core model dark photons are not vertical'
    assert np.all(parent[:, 1] > 0), 'core model dark photons go down'


@test
def test_floating_model_is_upward_hemisphere():
    rng = np.random.default_rng(5)
    parent = kin.dark_photon_momenta(7000.0, 0.23, 20000, 'floating', rng)
    assert np.all(parent[:, 1] >= 0), 'floating model emits downward'
    cos_alpha = parent[:, 1] / np.linalg.norm(parent[:, :3], axis=1)
    # cos(alpha) should be uniform on [0, 1]
    assert abs(cos_alpha.mean() - 0.5) < 0.02, \
        'cos(alpha) mean %.4f is not 0.5' % cos_alpha.mean()


@test
def test_monoenergetic_forces_the_momentum():
    rng = np.random.default_rng(6)
    m_X = 7000.0
    parent = kin.dark_photon_momenta(m_X, 0.23, 1000, 'monoenergetic', rng)
    d1, _ = kin.two_body_decay(parent, 0.23, k.MUON_MASS, rng)
    forced = kin.force_monoenergetic(d1, 0.5 * m_X, k.MUON_MASS)
    approx(kin.momentum_magnitude(forced), 0.5 * m_X * np.ones(1000), 1e-12,
           'monoenergetic momentum is not m_X/2')
    m2 = forced[:, 3] ** 2 - np.sum(forced[:, :3] ** 2, axis=1)
    approx(np.sqrt(m2), k.MUON_MASS * np.ones(1000), 1e-6,
           'monoenergetic rescaling took the muon off shell')


@test
def test_opening_angle_scales_like_one_over_gamma():
    """The mean opening angle should track m_A / E_A prime."""
    rng = np.random.default_rng(7)
    m_A = 0.23
    angles = {}
    for m_X in (1000.0, 10000.0):
        parent = kin.dark_photon_momenta(m_X, m_A, 20000, 'core', rng)
        d1, d2 = kin.two_body_decay(parent, m_A, k.MUON_MASS, rng)
        angles[m_X] = np.median(kin.opening_angle(d1, d2))
    ratio = angles[1000.0] / angles[10000.0]
    assert 8 < ratio < 12, \
        'opening angle scaled by %.2f for a factor 10 in m_X' % ratio


# ---------------------------------------------------------------------------
# depth sampling
# ---------------------------------------------------------------------------

@test
def test_exponential_depths_stay_in_the_slab():
    rng = np.random.default_rng(8)
    for L in (1e3, 1e5, 4e5, 1e9, 1e12):
        d = geo.sample_depths_exponential(20000, -4000.0, -8.0, L, rng)
        assert d.min() >= -4000.0 - 1e-6 and d.max() <= -8.0 + 1e-6, \
            'depths left the slab for L = %g cm' % L


@test
def test_exponential_reduces_to_uniform_for_long_decay_lengths():
    rng = np.random.default_rng(9)
    d = geo.sample_depths_exponential(200000, -4000.0, -8.0, 1e14, rng)
    mean = d.mean()
    expected = -0.5 * (4000.0 + 8.0)
    assert abs(mean - expected) < 20.0, \
        'mean depth %.1f m, expected about %.1f m' % (mean, expected)


@test
def test_exponential_favours_the_deep_end():
    """For L comparable to the slab, decays pile up at the bottom.

    The analytic mean of p(d) proportional to exp(d/L) on [d_lo, d_hi] is
    checked directly rather than eyeballed.
    """
    rng = np.random.default_rng(10)
    L = 1.0e5                                  # cm, i.e. 1 km
    d_lo, d_hi = 8.0 * 100, 4000.0 * 100       # cm
    d = -geo.sample_depths_exponential(400000, -4000.0, -8.0, L, rng) * 100

    a, b = d_lo / L, d_hi / L
    # <d> = L * [ (b-1)e^b - (a-1)e^a ] / [ e^b - e^a ]
    num = (b - 1) * np.exp(b) - (a - 1) * np.exp(a)
    den = np.exp(b) - np.exp(a)
    expected = L * num / den

    assert abs(d.mean() - expected) / expected < 0.01, \
        'mean depth %.1f cm, analytic %.1f cm' % (d.mean(), expected)
    assert d.mean() > 0.5 * (d_lo + d_hi), \
        'the exponential did not favour the deep end'


# ---------------------------------------------------------------------------
# energy loss
# ---------------------------------------------------------------------------

@test
def test_eloss_matches_earthshine_reference():
    """propagate_running must reproduce EarthShine's propagate_muon_CMSSW.

    Same closed form, same coefficients; the only difference is the clamp at
    zero, so the comparison is restricted to muons that survive.
    """
    E = np.array([1e3, 3e3, 1e4, 5e4])
    dist = np.array([100.0, 500.0, 1000.0, 2000.0])
    ours = eloss.propagate_running(E, dist)

    l10 = np.log10(E)
    a = (1.91514 + 0.254957 * l10) / 1000.0
    b = (0.379763 + 1.69516 * l10 - 0.175026 * l10 ** 2) / 1.0e6
    eps = a / b
    ref = (E + eps) * np.exp(-b * dist * 100 * 2.65) - eps
    approx(ours, ref, 1e-12,
           'running-coefficient energy loss disagrees with the reference')


@test
def test_eloss_is_monotonic_and_bounded():
    E0 = 5000.0
    dist = np.linspace(0, 5000, 200)
    for model in ('running', 'constant'):
        E = eloss.propagate(E0 * np.ones_like(dist), dist, model=model)
        assert np.all(np.diff(E) <= 1e-9), '%s energy loss is not monotonic' % model
        assert E[0] <= E0 + 1e-9 and E[-1] >= 0.0, '%s energy left bounds' % model


@test
def test_eloss_none_is_a_no_op():
    E = np.array([100.0, 1000.0])
    approx(eloss.propagate(E, np.array([1000.0, 1000.0]), model='none'), E,
           0.0, "eloss_model 'none' changed the energy")


@test
def test_degrade_preserves_direction():
    rng = np.random.default_rng(11)
    p3 = rng.normal(size=(500, 3)) * 100
    E = np.sqrt(np.sum(p3 ** 2, axis=1) + k.MUON_MASS ** 2)
    p4 = np.column_stack([p3, E])
    out, alive = eloss.degrade(p4, 300.0 * np.ones(500))
    before = geo.unit(p4[alive][:, :3])
    after = geo.unit(out[alive][:, :3])
    approx(after, before, 1e-9, 'energy loss changed the direction')


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

@test
def test_vertical_ray_hits_the_cylinder_when_it_should():
    origins = np.array([[0.0, -1000.0, 0.0],      # dead centre, must hit
                        [100.0, -1000.0, 0.0],    # far outside in x, must miss
                        [0.0, -1000.0, 100.0]])   # far outside in z, must miss
    directions = np.tile([0.0, 1.0, 0.0], (3, 1))
    entry, exit_ = geo.ray_cylinder_intersection(origins, directions, 8.0, 15.0)
    assert not np.isnan(entry[0, 0]), 'central vertical ray missed'
    assert np.isnan(entry[1, 0]), 'ray outside the radius hit'
    assert np.isnan(entry[2, 0]), 'ray outside the length hit'
    approx(entry[0, 1], -8.0, 1e-12, 'entry point is not on the surface')
    approx(exit_[0, 1], 8.0, 1e-12, 'exit point is not on the surface')


@test
def test_entry_point_lies_on_the_surface():
    rng = np.random.default_rng(12)
    n = 2000
    origins = np.column_stack([rng.uniform(-30, 30, n),
                               -1000.0 * np.ones(n),
                               rng.uniform(-30, 30, n)])
    directions = np.column_stack([rng.normal(0, 0.02, n),
                                  np.ones(n),
                                  rng.normal(0, 0.02, n)])
    entry, _ = geo.ray_cylinder_intersection(origins, directions, 8.0, 15.0)
    hit = ~np.isnan(entry[:, 0])
    assert hit.sum() > 100, 'too few hits to test'
    r = np.hypot(entry[hit, 0], entry[hit, 1])
    on_barrel = np.abs(r - 8.0) < 1e-8
    on_cap = np.abs(np.abs(entry[hit, 2]) - 15.0) < 1e-8
    assert np.all(on_barrel | on_cap), 'entry point is not on the cylinder'


@test
def test_origins_fill_the_disk_uniformly():
    rng = np.random.default_rng(13)
    n = 200000
    depths = geo.sample_depths_uniform(n, -100.0, -8.0, rng)
    o = geo.sample_origins(n, 40.0, depths, rng)
    r = np.hypot(o[:, 0], o[:, 2])
    assert r.max() <= 40.0 + 1e-9, 'origin outside the disk'
    # for a uniform disk <r> = 2R/3
    assert abs(r.mean() - 2 * 40.0 / 3) < 0.2, \
        'mean radius %.3f, expected %.3f' % (r.mean(), 2 * 40.0 / 3)
    approx(o[:, 1], depths, 0.0, 'origin depth does not match the sample')


@test
def test_volume_and_area():
    approx(geo.volume_m3(-4000.0, -8.0, 40.0), 3992 * np.pi * 1600, 1e-12,
           'generation volume is wrong')
    approx(geo.cross_section_area_cm2(40.0), np.pi * 4000.0 ** 2, 1e-12,
           'intercepting area is wrong')


# ---------------------------------------------------------------------------
# rate chain
# ---------------------------------------------------------------------------

@test
def test_kappa0_scales_like_one_over_mx_at_high_mass():
    """kappa_0 carries an explicit n_X = rho/m_X, so it falls at large m_X."""
    a = dp.kappa0(1.0e5)
    b = dp.kappa0(1.0e6)
    assert b < a, 'kappa_0 grew with m_X at high mass'


@test
def test_capture_rate_scaling():
    kappa = 1.0e23
    base = dp.capture_rate(7000.0, 0.23, 1e-8, 1.0, kappa)
    approx(dp.capture_rate(7000.0, 0.23, 2e-8, 1.0, kappa), 4 * base, 1e-12,
           'C_cap does not scale as epsilon^2')
    approx(dp.capture_rate(7000.0, 0.46, 1e-8, 1.0, kappa), base / 16, 1e-12,
           'C_cap does not scale as 1/m_A^4')


@test
def test_decay_shell_fraction_is_a_probability():
    L = 7.0e9
    R = k.R_CROSS
    f = dp.decay_shell_fraction(L, R - 4.0e5, R)
    assert 0 < f < 1, 'decay fraction %g is not a probability' % f
    # a shell of thickness d, far inside the decay length, holds d/L of the flux
    approx(f, 4.0e5 / L * np.exp(-R / L), 1e-3,
           'thin-shell limit of the decay fraction is wrong')


@test
def test_branching_ratios_are_sane():
    assert pl.branching_ratio(0.15, 'mumu') == 0.0, \
        'nonzero dimuon BR below threshold'
    br = float(pl.branching_ratio(0.30, 'mumu'))
    assert 0.0 < br < 1.0, 'dimuon BR at 0.3 GeV is %g' % br


@test
def test_alpha_x_resolution():
    v, mode = dp.resolve_alpha_x('thermal', 7000.0, 0.23)
    assert mode == 'thermal' and v > 0
    v, mode = dp.resolve_alpha_x('max', 7000.0, 0.23)
    assert mode == 'max' and v > 0
    v, mode = dp.resolve_alpha_x('0.25', 7000.0, 0.23)
    assert mode == 'fixed' and abs(v - 0.25) < 1e-12


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def _run_generator(extra_args, tmpdir):
    out = os.path.join(tmpdir, 'events.lhe')
    cmd = [sys.executable, os.path.join(ROOT, 'EarthShineGen'),
           '--n_events', '25', '--seed', '4242',
           '--output_file', out, '--report_file', '',
           '--max_trials', '400000'] + extra_args
    proc = subprocess.run(cmd, cwd=tmpdir, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise AssertionError('EarthShineGen %s failed:\n%s'
                             % (' '.join(extra_args),
                                proc.stdout.decode('utf-8', 'replace')))
    return out


def _parse_lhe(path):
    """Return a list of events, each a list of (pdg, status, p4) tuples."""
    events = []
    current = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s == '<event>':
                current = []
                header_seen = False
                continue
            if s == '</event>':
                events.append(current)
                current = None
                continue
            if current is None or s.startswith('#'):
                continue
            fields = s.split()
            if not header_seen:
                header_seen = True
                continue
            current.append((int(fields[0]), int(fields[1]),
                            np.array([float(fields[6]), float(fields[7]),
                                      float(fields[8]), float(fields[9])]),
                            float(fields[10])))
    return events


@test
def test_end_to_end_every_model_and_stage():
    tmpdir = tempfile.mkdtemp(prefix='earthshinegen_test_')
    cases = [
        ['--dm_model', 'core', '--stage', 'detector'],
        ['--dm_model', 'core', '--stage', 'vertex'],
        ['--dm_model', 'floating', '--disk_radius', '400',
         '--depth_min', '-500', '--stage', 'detector'],
        ['--dm_model', 'monoenergetic', '--disk_radius', '400',
         '--depth_min', '-500', '--stage', 'detector'],
        ['--depth_sampling', 'uniform'],
        ['--eloss_model', 'constant'],
        ['--eloss_model', 'none'],
        ['--require_hit', 'inner_detector', '--depth_min', '-500'],
        ['--require_both_muons', '1'],
        ['--decay_length_convention', 'darkcappy'],
        ['--epsilon', '1e-6'],          # short decay length, exercises the CDF
    ]
    for case in cases:
        path = _run_generator(case, tmpdir)
        events = _parse_lhe(path)
        assert len(events) == 25, \
            '%s produced %d events, expected 25' % (' '.join(case), len(events))


@test
def test_lhe_record_is_self_consistent():
    tmpdir = tempfile.mkdtemp(prefix='earthshinegen_lhe_')
    path = _run_generator(['--dm_model', 'core', '--stage', 'vertex'], tmpdir)
    events = _parse_lhe(path)
    for ev in events:
        incoming = [p for p in ev if p[1] == -1]
        outgoing = [p for p in ev if p[1] == 1]
        assert len(outgoing) == 2, 'expected two final state muons'
        assert set(p[0] for p in outgoing) == {13, -13}, 'muon PDG IDs are wrong'

        if incoming:
            total_in = sum(p[2] for p in incoming)
            total_out = sum(p[2] for p in outgoing)
            approx(total_in, total_out, 1e-6,
                   'four-momentum is not conserved in the LHE record')

        for pdg, status, p4, mass in outgoing:
            m2 = p4[3] ** 2 - np.sum(p4[:3] ** 2)
            approx(np.sqrt(max(m2, 0.0)), mass, 1e-4,
                   'LHE mass field disagrees with the four-vector')
            approx(mass, k.MUON_MASS, 1e-4, 'muon mass in the record is wrong')

    # in the vertex stage the pair must still reconstruct the dark photon
    for ev in events:
        outgoing = [p[2] for p in ev if p[1] == 1]
        tot = outgoing[0] + outgoing[1]
        m = np.sqrt(max(tot[3] ** 2 - np.sum(tot[:3] ** 2), 0.0))
        approx(m, 0.23, 1e-4, 'the pair does not reconstruct m_A')


@test
def test_vertices_are_written_and_consistent():
    tmpdir = tempfile.mkdtemp(prefix='earthshinegen_vtx_')
    path = _run_generator(['--stage', 'detector'], tmpdir)
    n_vertex = n_mu1 = n_mu2 = n_decay = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith('#vertex '):
                n_vertex += 1
                y = float(line.split()[2])
                assert -7600 < y < 0, \
                    'detector-stage vertex at y = %.1f mm is not on the ' \
                    'hand-off surface below the detector' % y
            elif line.startswith('#vertex_mu1'):
                n_mu1 += 1
            elif line.startswith('#vertex_mu2'):
                n_mu2 += 1
            elif line.startswith('#decay_vertex'):
                n_decay += 1
                y = float(line.split()[2])
                assert -4.0e6 <= y <= -8.0e3 + 1, \
                    'decay vertex at y = %.1f mm is outside the volume' % y
    assert n_vertex == n_mu1 == n_mu2 == n_decay == 25, \
        'vertex comment lines are missing (%d, %d, %d, %d)'\
        % (n_vertex, n_mu1, n_mu2, n_decay)


@test
def test_detector_size_is_configurable():
    """A bigger detector must move the hand-off surface and raise the acceptance.

    The hand-off surface defaults to 'auto', meaning the detector's own outer
    cylinder, so setting one radius moves both and there is only one geometry
    to keep consistent.
    """
    tmpdir = tempfile.mkdtemp(prefix='earthshinegen_size_')

    def _vertex_radii(args):
        path = _run_generator(args, tmpdir)
        radii = []
        with open(path) as fh:
            for line in fh:
                if line.startswith('#vertex '):
                    x, y, _ = (float(v) for v in line.split()[1:4])
                    radii.append(np.hypot(x, y) / 1000.0)   # mm -> m
        return np.array(radii)

    small = _vertex_radii(['--detector_radius', '7.5',
                           '--detector_half_length', '15'])
    big = _vertex_radii(['--detector_radius', '30',
                         '--detector_half_length', '60',
                         '--depth_max', '-40'])

    # every vertex sits on its own hand-off cylinder (barrel or endcap, so the
    # radius in (x, y) is at most the cylinder radius)
    assert small.max() <= 7.5 + 1e-6, \
        'vertex outside the 7.5 m surface: %.4f m' % small.max()
    assert big.max() <= 30.0 + 1e-6, \
        'vertex outside the 30 m surface: %.4f m' % big.max()
    assert big.max() > 7.5, \
        'the 30 m detector produced no vertex beyond 7.5 m; the size did not ' \
        'take effect'


@test
def test_explicit_handoff_overrides_auto():
    from earthshinegen import card as card_mod
    v = card_mod.resolve(None, {'detector_radius': 10.0,
                                'detector_half_length': 20.0,
                                'depth_max': -25.0})
    assert card_mod.resolve_handoff(v) == (10.0, 20.0), \
        "'auto' did not follow the detector size"

    v = card_mod.resolve(None, {'detector_radius': 10.0,
                                'detector_half_length': 20.0,
                                'handoff_radius': '18',
                                'handoff_half_length': '25',
                                'depth_max': -25.0})
    assert card_mod.resolve_handoff(v) == (18.0, 25.0), \
        'an explicit hand-off size was not used'


@test
def test_detector_geometry_is_validated():
    from earthshinegen import card as card_mod

    def _expect_error(overrides, fragment):
        try:
            card_mod.resolve(None, overrides)
        except ValueError as exc:
            assert fragment in str(exc), \
                'wrong error for %r: %s' % (overrides, exc)
        else:
            raise AssertionError('no error raised for %r' % (overrides,))

    _expect_error({'inner_detector_radius': 20.0}, 'larger than')
    _expect_error({'inner_detector_half_length': 40.0}, 'larger than')
    _expect_error({'handoff_radius': '2.0'}, 'must enclose')
    _expect_error({'handoff_half_length': '3.0'}, 'must enclose')
    _expect_error({'detector_radius': -1.0}, 'must be positive')
    # the generation volume has to start below the hand-off surface
    _expect_error({'depth_max': -1.0}, 'inside the hand-off surface')


@test
def test_report_only_runs():
    tmpdir = tempfile.mkdtemp(prefix='earthshinegen_rep_')
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'EarthShineGen'), '--report-only',
         '--report_file', os.path.join(tmpdir, 'r.txt')],
        cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, proc.stdout.decode('utf-8', 'replace')
    text = proc.stdout.decode('utf-8', 'replace')
    for token in ('kappa_0', 'Gamma_ann', 'decay length', 'muon pairs'):
        assert token in text, 'report is missing %r' % token


@test
def test_card_round_trip():
    from earthshinegen import card as card_mod
    tmpdir = tempfile.mkdtemp(prefix='earthshinegen_card_')
    path = os.path.join(tmpdir, 'parameter.txt')
    card_mod.write_template(path)
    values = card_mod.resolve(path)
    for key, default in card_mod.DEFAULTS.items():
        assert str(values[key]) == str(default), \
            'card round trip changed %s: %r -> %r' % (key, default, values[key])


def main():
    failures = 0
    for fn in _results:
        name = fn.__name__
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print('FAIL  %s\n        %s' % (name, exc))
        else:
            print('ok    %s' % name)
    print('')
    print('%d/%d tests passed' % (len(_results) - failures, len(_results)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
