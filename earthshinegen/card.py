"""The parameter card.

BlackMax and Charybdis are driven by a positional parameter.txt that the
gridpack edits with `sed -i '8s/.*/${MD}/'`.  That works, but it means the
gridpack scripts encode line numbers and break the moment a comment is added.
EarthShineGen uses a `key  value` card instead, so the gridpack edits it by
name:

    sed -i "s|^m_X .*|m_X  ${MX}|" parameter.txt

Blank lines and everything after a '#' are ignored.  Every key is also exposed
as a command-line option, and the command line wins over the card.
"""

import os

# name -> (type, default, help)
SCHEMA = [
    # --- dark sector -----------------------------------------------------
    ('m_X', float, 7000.0,
     'dark matter mass [GeV]'),
    ('m_A', float, 0.23,
     'dark photon mass [GeV]'),
    ('epsilon', float, 1e-8,
     'kinetic mixing between the photon and the dark photon'),
    ('alpha_X', str, 'max',
     "dark fine structure constant: 'thermal', 'max', or a number"),
    ('alpha_em', float, 1.0 / 137.0,
     'electromagnetic fine structure constant'),

    # --- generation ------------------------------------------------------
    ('n_events', int, 1000,
     'number of accepted events to write'),
    ('seed', int, 0,
     'random seed; 0 draws one from the OS and records it'),
    ('dm_model', str, 'core',
     "'core', 'floating' or 'monoenergetic'"),
    ('depth_min', float, -4000.0,
     'deep edge of the generation volume [m], negative, below the detector'),
    ('depth_max', float, -8.0,
     'shallow edge of the generation volume [m], negative'),
    ('disk_radius', str, '40',
     "radius of the generation cylinder [m], or 'auto' for the angle-based value"),
    ('depth_sampling', str, 'exponential',
     "'exponential' (from the dark photon decay length) or 'uniform'"),
    ('max_trials', int, 0,
     'cap on thrown events; 0 means keep throwing until n_events are accepted'),

    # --- detector --------------------------------------------------------
    # The detector is two coaxial cylinders centred on the origin with their
    # axis along z, plus the surface the muons are delivered on.  Every size
    # is a free parameter; the defaults describe a general-purpose collider
    # detector and are not tied to any particular experiment.
    ('detector_radius', float, 7.5,
     'outer detector radius [m]'),
    ('detector_half_length', float, 15.0,
     'outer detector half length along z [m]'),
    ('inner_detector_radius', float, 1.0,
     'inner detector radius [m]'),
    ('inner_detector_half_length', float, 2.5,
     'inner detector half length along z [m]'),
    ('handoff_radius', str, 'auto',
     "radius of the surface the muons are delivered on [m], or 'auto' to use "
     "detector_radius"),
    ('handoff_half_length', str, 'auto',
     "half length of that surface along z [m], or 'auto' to use "
     "detector_half_length"),
    ('require_hit', str, 'detector',
     "acceptance requirement: 'none', 'detector' or 'inner_detector'"),
    ('require_both_muons', int, 0,
     'if 1, both muons must satisfy the acceptance requirement, not just one'),
    ('min_momentum', float, 10.0,
     'minimum muon momentum on arrival at the detector [GeV]'),
    ('min_pt', float, 0.0,
     'minimum muon transverse momentum on arrival at the detector [GeV]'),

    # --- propagation -----------------------------------------------------
    ('eloss_model', str, 'running',
     "'running' (a, b run with log10 E; PDG 2004 standard rock), 'constant' "
     "(fixed a, b) or 'none'"),
    ('rock_density', float, 2.65,
     'overburden density [g/cm^3]'),
    ('ms_model', str, 'none',
     "multiple scattering in the rock: 'none' or 'highland'.  'none' is the "
     "default so that existing samples reproduce; see the README before "
     "turning it on"),
    ('rock_radiation_length', float, 26.54,
     'radiation length of the overburden [g/cm^2]; 26.54 is standard rock'),

    # --- rate ------------------------------------------------------------
    ('decay_length_convention', str, 'total',
     "'total' (BR_ee sets the width, BR_mumu scales the yield) or "
     "'darkcappy' (reproduce the notebook: BR_mumu in the decay length, "
     "no explicit yield factor)"),
    ('livetime_years', float, 1.0,
     'live time the quoted rate refers to [yr]'),
    ('kappa_method', str, 'fast',
     "'fast' (analytic recoil integral) or 'dblquad' (DarkCapPy's quadrature)"),

    # --- output ----------------------------------------------------------
    ('stage', str, 'detector',
     "'detector' (propagate to the hand-off surface) or 'vertex' (decay point)"),
    ('output_file', str, 'events.lhe',
     'output LHE file'),
    ('report_file', str, 'earthshinegen_report.txt',
     "human-readable rate report; '' to skip"),
    ('include_initial', int, 1,
     'write two mock incoming particles into each event'),
    ('include_mother', int, 1,
     "write the A' as a status-2 intermediate particle"),
    ('beam_energy', float, 6800.0,
     'beam energy quoted in the <init> block [GeV]'),
    ('verbose', int, 0,
     'print per-stage diagnostics'),
]

DEFAULTS = dict((name, default) for name, _, default, _ in SCHEMA)
TYPES = dict((name, typ) for name, typ, _, _ in SCHEMA)
HELP = dict((name, text) for name, _, _, text in SCHEMA)


def read_card(path):
    """Parse a parameter card into a dict of typed values."""
    values = {}
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError('%s:%d: expected "key value", got %r'
                                 % (path, lineno, raw.rstrip()))
            key, value = parts[0], parts[1].strip()
            if key not in TYPES:
                raise ValueError('%s:%d: unknown parameter %r' % (path, lineno, key))
            values[key] = _convert(key, value, '%s:%d' % (path, lineno))
    return values


def _convert(key, value, where):
    typ = TYPES[key]
    try:
        if typ is str:
            return value
        return typ(value)
    except ValueError:
        raise ValueError('%s: %s expects %s, got %r'
                         % (where, key, typ.__name__, value))


def resolve(card_path=None, overrides=None):
    """Defaults, then the card, then the command line.

    Returns a plain dict with every key in SCHEMA present.
    """
    values = dict(DEFAULTS)
    if card_path:
        if not os.path.exists(card_path):
            raise IOError('parameter card not found: %s' % card_path)
        values.update(read_card(card_path))
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            values[key] = _convert(key, value, 'command line') \
                if isinstance(value, str) else value
    _validate(values)
    return values


def _validate(v):
    if v['m_A'] <= 0:
        raise ValueError('m_A must be positive')
    if v['m_X'] <= v['m_A']:
        raise ValueError('m_X (%g) must exceed m_A (%g)' % (v['m_X'], v['m_A']))
    if v['epsilon'] <= 0:
        raise ValueError('epsilon must be positive')
    if v['rock_radiation_length'] <= 0:
        raise ValueError('rock_radiation_length must be positive, got %g'
                         % v['rock_radiation_length'])
    if v['n_events'] <= 0:
        raise ValueError('n_events must be positive')
    if v['depth_min'] > v['depth_max']:
        v['depth_min'], v['depth_max'] = v['depth_max'], v['depth_min']
    if v['depth_max'] > 0:
        raise ValueError('the generation volume must sit below the detector; '
                         'depths are negative')
    for key, allowed in (('dm_model', ('core', 'floating', 'monoenergetic')),
                         ('depth_sampling', ('exponential', 'uniform')),
                         ('eloss_model', ('running', 'constant', 'none')),
                         ('ms_model', ('none', 'highland')),
                         ('stage', ('detector', 'vertex')),
                         ('kappa_method', ('fast', 'dblquad')),
                         ('require_hit', ('none', 'detector',
                                          'inner_detector')),
                         ('decay_length_convention', ('total', 'darkcappy'))):
        if v[key] not in allowed:
            raise ValueError('%s must be one of %s, got %r'
                             % (key, ', '.join(allowed), v[key]))

    _validate_detector(v)


def resolve_handoff(v):
    """The hand-off surface size, resolving 'auto' to the detector envelope.

    Returns (radius, half_length) in metres.  'auto' means the muons are
    delivered on the detector's own outer surface, so changing the detector
    size moves the hand-off surface with it and there is only one geometry to
    keep consistent.
    """
    def _one(key, fallback):
        spec = v[key]
        if isinstance(spec, str) and spec.strip().lower() == 'auto':
            return float(fallback)
        return float(spec)

    return (_one('handoff_radius', v['detector_radius']),
            _one('handoff_half_length', v['detector_half_length']))


def _validate_detector(v):
    """The three surfaces have to nest: inner <= detector <= hand-off."""
    for key in ('detector_radius', 'detector_half_length',
                'inner_detector_radius', 'inner_detector_half_length'):
        if v[key] <= 0:
            raise ValueError('%s must be positive, got %g' % (key, v[key]))

    if v['inner_detector_radius'] > v['detector_radius']:
        raise ValueError('inner_detector_radius (%g m) is larger than '
                         'detector_radius (%g m)'
                         % (v['inner_detector_radius'], v['detector_radius']))
    if v['inner_detector_half_length'] > v['detector_half_length']:
        raise ValueError('inner_detector_half_length (%g m) is larger than '
                         'detector_half_length (%g m)'
                         % (v['inner_detector_half_length'],
                            v['detector_half_length']))

    handoff_r, handoff_hl = resolve_handoff(v)
    if handoff_r <= 0 or handoff_hl <= 0:
        raise ValueError('the hand-off surface must have positive size, got '
                         'radius %g m, half length %g m' % (handoff_r, handoff_hl))
    if handoff_r < v['detector_radius'] or handoff_hl < v['detector_half_length']:
        raise ValueError(
            'the hand-off surface (radius %g m, half length %g m) must enclose '
            'the detector (radius %g m, half length %g m), otherwise the muons '
            'are delivered inside it'
            % (handoff_r, handoff_hl, v['detector_radius'],
               v['detector_half_length']))

    if abs(v['depth_max']) < handoff_r:
        raise ValueError(
            'the shallow edge of the generation volume (%g m) is inside the '
            'hand-off surface (radius %g m); decays have to happen below it'
            % (v['depth_max'], handoff_r))


def write_template(path):
    """Write a fully commented card containing every parameter at its default."""
    sections = [
        ('dark sector', ('m_X', 'm_A', 'epsilon', 'alpha_X', 'alpha_em')),
        ('generation', ('n_events', 'seed', 'dm_model', 'depth_min',
                        'depth_max', 'disk_radius', 'depth_sampling',
                        'max_trials')),
        ('detector', ('detector_radius', 'detector_half_length',
                      'inner_detector_radius', 'inner_detector_half_length',
                      'handoff_radius', 'handoff_half_length', 'require_hit',
                      'require_both_muons', 'min_momentum', 'min_pt')),
        ('propagation', ('eloss_model', 'rock_density', 'ms_model',
                         'rock_radiation_length')),
        ('rate', ('decay_length_convention', 'livetime_years',
                  'kappa_method')),
        ('output', ('stage', 'output_file', 'report_file', 'include_initial',
                    'include_mother', 'beam_energy', 'verbose')),
    ]
    with open(path, 'w') as fh:
        fh.write('# EarthShineGen parameter card\n'
                 '#\n'
                 '# One "key  value" pair per line.  Anything after a # is a\n'
                 '# comment.  Edit by name, e.g.\n'
                 '#     sed -i "s|^m_X .*|m_X  10000|" parameter.txt\n')
        for title, keys in sections:
            fh.write('\n#' + '-' * 70 + '\n# %s\n#%s\n' % (title, '-' * 70))
            for key in keys:
                fh.write('%-28s %-24s # %s\n'
                         % (key, DEFAULTS[key], HELP[key]))
