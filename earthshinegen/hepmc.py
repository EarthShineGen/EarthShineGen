"""HepMC output, in both the 2 and 3 flavours.

The reason this exists next to lhe.py: LHE has no field for a production
vertex, and the whole point of this signal is a pair of muons entering the
detector from below at two different, correlated points.  lhe.py works around
that by writing the positions as comment lines that a downstream producer has
to be written to read back.  HepMC carries a production vertex per particle, so
the topology is expressible directly.

Two output flavours, selected by `version`:

  '2'  HepMC 2, the `HepMC::IO_GenEvent` ASCII flavour.  Still the format most
       detector simulations read from a file, which is why it is the default.
  '3'  HepMC 3, the `HepMC::Asciiv3` flavour, which is what Rivet and the
       HepMC3 tools want.

Both are written by hand rather than through a binding, because the only
dependencies here are numpy and scipy.  The event is built once, in `_build`,
and the two writers differ only in how they lay it out.

Units are GeV and mm.  Positions arrive in metres and are converted here, the
same convention lhe.py uses for its comment lines.

The record, for `stage = detector` and the default `split` topology:

    V-1   at the A' decay point in the rock
            in   the A' (if include_mother), with no production vertex
            out  the two muons as produced
    V-2   where muon 1 crosses the hand-off surface
            in   muon 1 as produced
            out  muon 1 as it arrives, after the energy loss
    V-3   the same for muon 2

The A' deliberately has no production vertex.  The mock incoming pair that
lhe.py writes exists only because LHE demands an initial state; carrying it
into HepMC would force a production vertex, and the only place to put it would
be the decay point itself, giving the A' a zero-length flight path that means
nothing.  The real production point is wherever the dark matter annihilated --
for the core model the centre of the Earth -- and this generator does not model
that flight at all: it samples the decay point directly.

Four-momentum is deliberately not conserved at V-2 and V-3: that difference is
the energy the muon left in the rock.  HepMC does not check, and there is no
parton shower in this chain, so nothing rejects it.  The status-1 muons -- the
only ones a detector simulation will track -- therefore start exactly on the
hand-off surface with exactly the arriving momenta, which is the thing LHE
could not express.

With `topology = single` the record collapses to one vertex at the midpoint of
the two crossings, reproducing the LHE event one for one.  That is an escape
hatch and the reference for the LHE/HepMC equivalence test; it throws away the
per-muon geometry, which with `ms_model highland` is metres.

All vertex times are written as zero.  The muons really do arrive microseconds
after the decay, but a time offset that size is meaningless to a detector
simulation with a nanosecond readout window and would only get cut.
"""

from array import array

import numpy as np

from . import __version__
from . import constants as k

MM_PER_M = 1000.0

VERSIONS = ('2', '3')

HEPMC2_VERSION_LINE = 'HepMC::Version 2.06.10\n'
HEPMC2_START_KEY = 'HepMC::IO_GenEvent-START_EVENT_LISTING\n'
HEPMC2_END_KEY = 'HepMC::IO_GenEvent-END_EVENT_LISTING\n'

HEPMC3_VERSION_LINE = 'HepMC::Version 3.03.01\n'
HEPMC3_START_KEY = 'HepMC::Asciiv3-START_EVENT_LISTING\n'
HEPMC3_END_KEY = 'HepMC::Asciiv3-END_EVENT_LISTING\n'

# Status codes, in the sense a detector simulation reads them:
#
#   1  not decayed by the generator; it is tracked from its production vertex
#   2  decayed by the generator, but still propagated, and handed over with a
#      predefined decay if its end vertex is far enough out
#   3  decayed by the generator, and must NOT be propagated
#
# The intermediates here are 3, not the 2 a collider generator would use, and
# the difference is not cosmetic.  The A' and the muons as produced live at the
# decay point, a kilometre underground and far outside any detector volume, and
# their end vertices are metres off axis -- exactly the condition under which a
# status-2 particle gets taken as a primary and tracked from where it starts.
# Status 3 says what is actually true, that the generator has already done that
# propagation, so only the two status-1 muons at the hand-off surface are
# tracked.  The codes mean the same in HepMC 3.
STATUS_INTERMEDIATE = 3
STATUS_FINAL = 1

SIGNAL_PROCESS_ID = 1
ALPHA_QCD = 1.181000e-01
ALPHA_QED = 7.2973525e-03


def _mass(p4):
    m2 = p4[3] ** 2 - (p4[0] ** 2 + p4[1] ** 2 + p4[2] ** 2)
    return float(np.sqrt(max(m2, 0.0)))


def _mm(position_m):
    if position_m is None:
        return None
    return np.asarray(position_m, dtype=float) * MM_PER_M


class _Particle(object):
    """One particle, held until the layout is known."""

    __slots__ = ('barcode', 'pdg', 'p4', 'mass', 'status',
                 'end_vertex', 'production_vertex')

    def __init__(self, barcode, pdg, p4, mass, status):
        self.barcode = barcode
        self.pdg = pdg
        self.p4 = p4
        self.mass = mass
        self.status = status
        self.end_vertex = 0
        self.production_vertex = 0


class _Vertex(object):
    """One vertex, with everything coming in and everything going out."""

    __slots__ = ('barcode', 'position_mm', 'incoming', 'out')

    def __init__(self, barcode, position_mm):
        self.barcode = barcode
        self.position_mm = position_mm
        self.incoming = []
        self.out = []

    def add_in(self, particle):
        particle.end_vertex = self.barcode
        self.incoming.append(particle)

    def add_out(self, particle):
        particle.production_vertex = self.barcode
        self.out.append(particle)

    @property
    def orphans_in(self):
        """Incoming particles that were not produced anywhere in the record."""
        return [p for p in self.incoming if p.production_vertex == 0]


def _metadata_value(value):
    """One run-metadata value, in a form that survives a line-oriented parse."""
    text = str(value)
    return '""' if text.strip() == '' else text


class _HepMCWriterBase(object):
    """Everything the two flavours share: the event, not its serialisation."""

    # The cross-section line is padded to a fixed width so it can be
    # overwritten in place whatever the exponents turn out to be.
    XSEC_WIDTH = 64

    VERSION_LINE = None
    START_KEY = None
    END_KEY = None

    def __init__(self, path, metadata, xsec_pb, xsec_err_pb=0.0,
                 max_weight=1.0, topology='split',
                 muon_pdgids=(k.MUON_PDGID, -k.MUON_PDGID),
                 aprime_pdgid=k.DARKPHOTON_PDGID,
                 include_mother=True):
        self.path = path
        self.topology = topology
        self.muon_pdgids = muon_pdgids
        self.aprime_pdgid = aprime_pdgid
        self.include_mother = include_mother
        self.n_events = 0

        self._xsec = (xsec_pb, xsec_err_pb)
        # Byte offsets of the cross-section lines.  Both flavours put the cross
        # section in every event rather than in a header, hence a list rather
        # than the single offset lhe.py keeps.  'q' rather than a python list
        # so that a ten-million-event sample costs 80 MB, not 400.
        self._xsec_offsets = array('q')

        self._fh = open(path, 'w+')
        self._write_preamble(metadata or {})

    def _write_preamble(self, metadata):
        """The version line, the run metadata and the start key.

        Where the metadata goes differs between the flavours, so the two
        writers override the middle of this.
        """
        raise NotImplementedError

    # -- the cross-section line ------------------------------------------
    def _xsec_body(self, xsec_pb, xsec_err_pb):
        raise NotImplementedError

    def _xsec_line(self, xsec_pb, xsec_err_pb):
        line = self._xsec_body(xsec_pb, xsec_err_pb)
        if len(line) > self.XSEC_WIDTH:
            raise ValueError('cross section line does not fit: %r' % line)
        return line.ljust(self.XSEC_WIDTH) + '\n'

    def update_cross_section(self, xsec_pb, xsec_err_pb=0.0, max_weight=1.0):
        """Rewrite the per-event cross section, after the events are written."""
        line = self._xsec_line(xsec_pb, xsec_err_pb)
        here = self._fh.tell()
        for offset in self._xsec_offsets:
            self._fh.seek(offset)
            self._fh.write(line)
        self._fh.seek(here)
        self._xsec = (xsec_pb, xsec_err_pb)

    # -- event building ---------------------------------------------------
    def _build(self, p1, p2, vertex_mm, vertex1_mm, vertex2_mm,
               decay_vertex_mm, p1_raw, p2_raw):
        """Return (vertices, beams, signal_vertex_barcode, mA)."""
        split = (self.topology == 'split' and vertex1_mm is not None
                 and vertex2_mm is not None)
        # Where the muons are born.  In the split record that is the decay
        # point; otherwise everything happens at the single event vertex.
        origin_mm = decay_vertex_mm if (split and decay_vertex_mm is not None) \
            else vertex_mm

        P = p1_raw + p2_raw if split else p1 + p2
        mA = _mass(P)

        vertices = []
        next_vertex = -1

        decay = _Vertex(next_vertex, origin_mm)
        next_vertex -= 1
        vertices.append(decay)

        if self.include_mother:
            # The A' is written as an incoming particle of its own decay vertex
            # and given no production vertex.  That is the honest record: this
            # generator never models the A' flight.  It samples the decay point
            # directly, and the real production point is wherever the dark
            # matter annihilated -- for the core model the centre of the Earth,
            # thousands of km down and many decay lengths away.  Inventing a
            # production vertex at the decay point, which is what an LHE-style
            # mock initial state forces, would put a zero-length flight path in
            # the file and invite anyone downstream to measure it.
            decay.add_in(_Particle(0, self.aprime_pdgid, P, mA,
                                   STATUS_INTERMEDIATE))

        # The muons.  In the split record each one is produced at the decay
        # vertex, ends at its own crossing vertex, and is re-emitted there with
        # the momentum it actually arrives with.
        pairs = ((self.muon_pdgids[0], p1, p1_raw, vertex1_mm),
                 (self.muon_pdgids[1], p2, p2_raw, vertex2_mm))
        for pdg, arriving, produced, crossing_mm in pairs:
            if split:
                intermediate = _Particle(0, pdg, produced, _mass(produced),
                                         STATUS_INTERMEDIATE)
                decay.add_out(intermediate)

                crossing = _Vertex(next_vertex, crossing_mm)
                next_vertex -= 1
                vertices.append(crossing)
                crossing.add_in(intermediate)
                crossing.add_out(_Particle(0, pdg, arriving, _mass(arriving),
                                           STATUS_FINAL))
            else:
                decay.add_out(_Particle(0, pdg, arriving, _mass(arriving),
                                        STATUS_FINAL))

        # Number the particles in the order they are written.  Barcodes are
        # only labels and both readers resolve the topology through the vertex
        # ids, but a file whose particle lines count 1, 2, 3... is far easier
        # to read by eye and matches what every other generator produces.
        barcode = 0
        for vertex in vertices:
            for particle in vertex.orphans_in + vertex.out:
                barcode += 1
                particle.barcode = barcode

        return vertices, (0, 0), decay.barcode, mA

    def write_event(self, p1, p2, vertex_m, vertex1_m=None, vertex2_m=None,
                    decay_vertex_m=None, p1_raw=None, p2_raw=None,
                    weight=1.0):
        """Write one A' -> mu+ mu- event.

        p1, p2          the arriving four-momenta (px, py, pz, E) in GeV
        vertex_m        the single event vertex in metres
        vertex1_m, vertex2_m
                        where each muon crosses the hand-off surface
        decay_vertex_m  the A' decay point in the rock
        p1_raw, p2_raw  the muon four-momenta as produced, before the energy
                        loss; default to p1, p2 (the vertex stage, where the
                        record is the decay itself and nothing is degraded)
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        p1_raw = p1 if p1_raw is None else np.asarray(p1_raw, dtype=float)
        p2_raw = p2 if p2_raw is None else np.asarray(p2_raw, dtype=float)

        built = self._build(p1, p2, _mm(vertex_m), _mm(vertex1_m),
                            _mm(vertex2_m), _mm(decay_vertex_m),
                            p1_raw, p2_raw)
        self.n_events += 1
        self._write(weight, *built)

    def _write(self, weight, vertices, beams, signal_vertex, mA):
        raise NotImplementedError

    def close(self):
        self._fh.write(self.END_KEY)
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class HepMC2Writer(_HepMCWriterBase):
    """HepMC 2, the IO_GenEvent ASCII flavour."""

    VERSION_LINE = HEPMC2_VERSION_LINE
    START_KEY = HEPMC2_START_KEY
    END_KEY = HEPMC2_END_KEY

    def _write_preamble(self, metadata):
        # HepMC 2 has no header block and no run-info record.  Everything ahead
        # of the start key is skipped silently while IO_GenEvent looks for that
        # key, so the metadata goes there as plain comment lines.
        out = self._fh
        out.write(self.VERSION_LINE)
        out.write('# EarthShineGen run metadata\n')
        for key in sorted(metadata):
            out.write('#   %-28s = %s\n' % (key, metadata[key]))
        out.write(self.START_KEY)

    def _xsec_body(self, xsec_pb, xsec_err_pb):
        return 'C %+.9e %+.9e' % (xsec_pb, xsec_err_pb)

    @staticmethod
    def _particle_line(particle):
        # 15 significant digits for the same reason lhe.py uses them: these are
        # TeV muons with a 105 MeV mass, so E^2 - p^2 is a difference of two
        # numbers that agree to nine digits, and readers recover the mass from
        # the four-vector.
        p = particle.p4
        return ('P %d %d %+.15e %+.15e %+.15e %+.15e %+.15e %d 0 0 %d 0\n'
                % (particle.barcode, particle.pdg, p[0], p[1], p[2], p[3],
                   particle.mass, particle.status, particle.end_vertex))

    def _write(self, weight, vertices, beams, signal_vertex, mA):
        out = self._fh
        out.write('E %d -1 %+.15e %+.15e %+.15e %d %d %d %d %d 0 1 %+.15e\n'
                  % (self.n_events, mA, ALPHA_QCD, ALPHA_QED,
                     SIGNAL_PROCESS_ID, signal_vertex, len(vertices),
                     beams[0], beams[1], weight))
        out.write('N 1 "0"\n')
        out.write('U GEV MM\n')
        self._xsec_offsets.append(out.tell())
        out.write(self._xsec_line(*self._xsec))

        for vertex in vertices:
            orphans = vertex.orphans_in
            p = vertex.position_mm
            out.write('V %d 0 %+.15e %+.15e %+.15e %+.15e %d %d 0\n'
                      % (vertex.barcode, p[0], p[1], p[2], 0.0,
                         len(orphans), len(vertex.out)))
            # The incoming particles that have no production vertex come first,
            # then the outgoing ones; the reader counts them in that order.
            for particle in orphans:
                out.write(self._particle_line(particle))
            for particle in vertex.out:
                out.write(self._particle_line(particle))


class HepMC3Writer(_HepMCWriterBase):
    """HepMC 3, the Asciiv3 flavour.  For Rivet and the HepMC3 tools.

    Detector simulations are generally still on HepMC 2 for file input, so
    `hepmc_version 2` remains the default and this is for everything else.
    """

    VERSION_LINE = HEPMC3_VERSION_LINE
    START_KEY = HEPMC3_START_KEY
    END_KEY = HEPMC3_END_KEY

    def _write_preamble(self, metadata):
        # HepMC 3 has a real place for this: the run-info block, which sits
        # just inside the start key and holds the weight names, the generating
        # tool and a list of run-level attributes.  Comment lines would work in
        # the sense that ReaderAscii skips them, but it prints a warning for
        # every one of them, which for a few dozen metadata keys buries the
        # output of whatever is reading the file.
        out = self._fh
        out.write(self.VERSION_LINE)
        out.write(self.START_KEY)
        out.write('W 0\n')
        out.write('T EarthShineGen\\|%s\\|dark Earthshine dimuon generator\n'
                  % __version__)
        for key in sorted(metadata):
            out.write('A %s %s\n' % (key, _metadata_value(metadata[key])))

    def _xsec_body(self, xsec_pb, xsec_err_pb):
        # GenCrossSection is cross section, error, accepted events, attempted
        # events; -1 for the two counts means "not recorded", which is what the
        # library itself writes when they were never set.
        return ('A 0 GenCrossSection %.8e %.8e -1 -1'
                % (xsec_pb, xsec_err_pb))

    @staticmethod
    def _particle_line(particle):
        p = particle.p4
        return ('P %d %d %d %+.15e %+.15e %+.15e %+.15e %+.15e %d\n'
                % (particle.barcode, particle.production_vertex, particle.pdg,
                   p[0], p[1], p[2], p[3], particle.mass, particle.status))

    def _write(self, weight, vertices, beams, signal_vertex, mA):
        out = self._fh
        n_particles = sum(len(v.orphans_in) + len(v.out) for v in vertices)
        out.write('E %d %d %d\n' % (self.n_events, len(vertices), n_particles))
        out.write('U GEV MM\n')
        out.write('W %+.15e\n' % weight)
        self._xsec_offsets.append(out.tell())
        out.write(self._xsec_line(*self._xsec))

        for vertex in vertices:
            # A particle with no production vertex is written before the vertex
            # that consumes it; one that has a production vertex is written
            # after that vertex.  This is the order the library writes and the
            # order the reader expects.
            for particle in vertex.orphans_in:
                out.write(self._particle_line(particle))
            p = vertex.position_mm
            out.write('V %d 0 [%s] @ %+.15e %+.15e %+.15e %+.15e\n'
                      % (vertex.barcode,
                         ','.join(str(q.barcode) for q in vertex.incoming),
                         p[0], p[1], p[2], 0.0))
            for particle in vertex.out:
                out.write(self._particle_line(particle))


WRITERS = {'2': HepMC2Writer, '3': HepMC3Writer}


def HepMCWriter(path, metadata, version='2', **kwargs):
    """Open a writer for the requested HepMC flavour."""
    if str(version) not in WRITERS:
        raise ValueError("hepmc_version must be one of %s, got %r"
                         % (', '.join(sorted(WRITERS)), version))
    return WRITERS[str(version)](path, metadata, **kwargs)
