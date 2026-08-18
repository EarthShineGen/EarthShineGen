"""Les Houches Event output.

Adapted from EarthShine's event_writer.write_lhe, trimmed to what the gridpack
needs and extended so that the per-muon entry points survive into the file.

The LHE format has no field for a production vertex, which matters here because
the whole point of the signal is a pair of muons entering the detector from
below at a displaced, correlated position.  Three comment lines are written
inside every <event> block:

    #vertex        x y z     the vertex the event should be placed at
    #vertex_mu1    x y z     where muon 1 crosses the hand-off surface
    #vertex_mu2    x y z     where muon 2 crosses it
    #decay_vertex  x y z     the true A' decay point in the rock

LHE parsers ignore '#' lines, so the file stays valid for any reader.  A
downstream producer that wants the exact two-vertex topology can read the
per-muon lines back; one that does not will use the single event vertex, which
is written at the midpoint of the two muon entry points.

All positions are written in mm, the LHE/HepMC length unit.
"""

import numpy as np

from . import constants as k

MM_PER_M = 1000.0


def _particle_line(pdg, status, m1, m2, c1, c2, px, py, pz, e, m,
                   vtim=0.0, spin=9.0):
    # 16 significant digits rather than the customary 11.  The muons here carry
    # TeV energies and a 105 MeV mass, so E^2 - p^2 is a difference of two
    # numbers that agree to nine digits; at 11 digits the mass a reader
    # recovers from the four-vector is wrong by a per cent, which trips
    # momentum and mass consistency checks downstream.
    return ("    %9d %5d %5d %5d %5d %5d "
            "%+.15e %+.15e %+.15e %+.15e %+.15e %.4e %.4e\n"
            % (pdg, status, m1, m2, c1, c2, px, py, pz, e, m, vtim, spin))


def _mass(p4):
    m2 = p4[3] ** 2 - (p4[0] ** 2 + p4[1] ** 2 + p4[2] ** 2)
    return float(np.sqrt(max(m2, 0.0)))


def preamble(header_comment, xsec_pb, xsec_err_pb, max_weight,
             beam_energy=6800.0):
    """A minimal, spec-compliant header and <init> block.

    IDWTUP = 3 (unweighted, accept all) stops Pythia re-sampling weights.
    XSECUP carries the physical rate of the sample, converted to picobarns
    against the reference luminosity recorded in the header, so that the usual
    downstream normalisation machinery has something meaningful to use.  The
    header block itself carries the rate in its natural units, which is what
    you should actually quote.
    """
    return ('<LesHouchesEvents version="3.0">\n'
            '<header>\n'
            '%s'
            '</header>\n'
            '<init>\n'
            '    2212    2212  %.9e  %.9e    0    0  247000  247000    3    1\n'
            % (header_comment, beam_energy, beam_energy))


def init_process_line(xsec_pb, xsec_err_pb, max_weight):
    """The XSECUP/XERRUP/XMAXUP line.

    Written with fixed-width %.9e fields so it can be patched in place once the
    Monte Carlo acceptance is known (see LHEWriter.update_cross_section).
    """
    return '  %.9e  %.9e  %.9e    1\n' % (xsec_pb, xsec_err_pb, max_weight)


def header_block(params):
    """Render the run metadata as an XML-ish block for the LHE header."""
    lines = ['<EarthShineGen>\n']
    for key in sorted(params):
        lines.append('  %-28s = %s\n' % (key, params[key]))
    lines.append('</EarthShineGen>\n')
    return ''.join(lines)


class LHEWriter(object):
    """Streaming LHE writer: open, write events one at a time, close."""

    def __init__(self, path, header_comment, xsec_pb, xsec_err_pb=0.0,
                 max_weight=1.0, beam_energy=6800.0,
                 muon_pdgids=(k.MUON_PDGID, -k.MUON_PDGID),
                 aprime_pdgid=k.DARKPHOTON_PDGID,
                 include_initial=True, include_mother=True,
                 initial_pdgids=(11, -11)):
        self.path = path
        self.muon_pdgids = muon_pdgids
        self.aprime_pdgid = aprime_pdgid
        self.include_initial = include_initial
        self.include_mother = include_mother
        self.initial_pdgids = initial_pdgids
        self.n_events = 0

        self._fh = open(path, 'w+')
        self._fh.write(preamble(header_comment, xsec_pb, xsec_err_pb,
                                max_weight, beam_energy))
        # Remember where the process line starts so the cross section can be
        # corrected once the acceptance has been measured.  The replacement is
        # the same width, so it can be written in place.
        self._xsec_offset = self._fh.tell()
        self._fh.write(init_process_line(xsec_pb, xsec_err_pb, max_weight))
        self._fh.write('</init>\n')

    def update_cross_section(self, xsec_pb, xsec_err_pb=0.0, max_weight=1.0):
        """Rewrite XSECUP in place, after the events have been written."""
        here = self._fh.tell()
        self._fh.seek(self._xsec_offset)
        self._fh.write(init_process_line(xsec_pb, xsec_err_pb, max_weight))
        self._fh.seek(here)

    def write_event(self, p1, p2, vertex_m, vertex1_m=None, vertex2_m=None,
                    decay_vertex_m=None, weight=1.0):
        """Write one A' -> mu+ mu- event.

        p1, p2      four-momenta (px, py, pz, E) in GeV
        vertex_m    the event vertex in metres (detector frame)
        vertex1_m, vertex2_m, decay_vertex_m
                    optional extra positions recorded as comment lines
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        P = p1 + p2

        m1 = _mass(p1)
        m2 = _mass(p2)
        mA = _mass(P)
        pmag = float(np.sqrt(P[0] ** 2 + P[1] ** 2 + P[2] ** 2))
        if pmag <= 0.0:
            raise ValueError('event with zero total three-momentum')
        n_hat = P[:3] / pmag

        lines = []
        idx = 0
        mother = 0

        if self.include_initial:
            # Massless light-cone split of P: p_a + p_b = P exactly, both on
            # the massless shell, so momentum conservation survives the reader's
            # check.  Leptonic IDs need no colour lines.
            ea = 0.5 * (P[3] + pmag)
            eb = 0.5 * (P[3] - pmag)
            lines.append(_particle_line(self.initial_pdgids[0], -1, 0, 0, 0, 0,
                                        ea * n_hat[0], ea * n_hat[1],
                                        ea * n_hat[2], ea, 0.0))
            lines.append(_particle_line(self.initial_pdgids[1], -1, 0, 0, 0, 0,
                                        -eb * n_hat[0], -eb * n_hat[1],
                                        -eb * n_hat[2], eb, 0.0))
            idx += 2

        if self.include_mother:
            moms = (1, 2) if self.include_initial else (0, 0)
            lines.append(_particle_line(self.aprime_pdgid, 2, moms[0], moms[1],
                                        0, 0, P[0], P[1], P[2], P[3], mA))
            idx += 1
            mother = idx

        if self.include_mother:
            mu_moms = (mother, mother)
        elif self.include_initial:
            mu_moms = (1, 2)
        else:
            mu_moms = (0, 0)

        lines.append(_particle_line(self.muon_pdgids[0], 1, mu_moms[0],
                                    mu_moms[1], 0, 0,
                                    p1[0], p1[1], p1[2], p1[3], m1))
        lines.append(_particle_line(self.muon_pdgids[1], 1, mu_moms[0],
                                    mu_moms[1], 0, 0,
                                    p2[0], p2[1], p2[2], p2[3], m2))

        out = self._fh
        out.write('<event>\n')
        out.write('    %5d     1 %+.10e %.8e 7.2973525e-03 1.1810000e-01\n'
                  % (len(lines), weight, mA))
        out.writelines(lines)
        out.write(self._vertex_line('#vertex', vertex_m))
        if vertex1_m is not None:
            out.write(self._vertex_line('#vertex_mu1', vertex1_m))
        if vertex2_m is not None:
            out.write(self._vertex_line('#vertex_mu2', vertex2_m))
        if decay_vertex_m is not None:
            out.write(self._vertex_line('#decay_vertex', decay_vertex_m))
        out.write('</event>\n')
        self.n_events += 1

    @staticmethod
    def _vertex_line(tag, position_m):
        p = np.asarray(position_m, dtype=float) * MM_PER_M
        return '%s %.8e %.8e %.8e\n' % (tag, p[0], p[1], p[2])

    def close(self):
        self._fh.write('</LesHouchesEvents>\n')
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
