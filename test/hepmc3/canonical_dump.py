#!/usr/bin/env python3
"""Dump a HepMC 2 or HepMC 3 file in the same canonical form as dump_hepmc3.

    python3 canonical_dump.py events.hepmc3

One line per particle:

    <event> <id> <pdg> <status> <px> <py> <pz> <e> <m> <vx> <vy> <vz>

with the production vertex in mm, or "none".  Reading the file back with an
independent parser and getting the same lines the HepMC3 library gets is the
check that the hand-written writer produces valid HepMC; printing HepMC 2 and
HepMC 3 in the same form is what makes the two flavours comparable to each
other.

Pure python, no HepMC needed, so this half runs anywhere.
"""

import sys


def _fmt(values, vertex):
    numbers = ' '.join('%+.15e' % v for v in values)
    if vertex is None:
        return numbers + ' none'
    return numbers + ' ' + ' '.join('%+.15e' % v for v in vertex)


def dump_hepmc2(path):
    """P lines carry an end vertex; the production vertex is the enclosing V."""
    out = []
    listing = False
    event = 0
    pending = None
    position = None

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            if line.startswith('HepMC::IO_GenEvent-START_EVENT_LISTING'):
                listing = True
                continue
            if line.startswith('HepMC::IO_GenEvent-END_EVENT_LISTING'):
                break
            if not listing or not line:
                continue

            tag, rest = line[0], line[2:].split()
            if tag == 'E':
                event = int(rest[0])
                pending = None
            elif tag == 'V':
                position = [float(v) for v in rest[2:5]]
                pending = [int(rest[6]), int(rest[7]), 0]
            elif tag == 'P':
                n_in, n_out, seen = pending
                incoming = seen < n_in
                pending[2] += 1
                values = [float(v) for v in rest[2:7]]
                out.append('%d %s %s %s %s'
                           % (event, rest[0], rest[1], rest[7],
                              _fmt(values, None if incoming else position)))
    return out


def dump_hepmc3(path):
    """P lines carry the production vertex id directly."""
    out = []
    listing = False
    event = 0
    positions = {}

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            if line.startswith('HepMC::Asciiv3-START_EVENT_LISTING'):
                listing = True
                continue
            if line.startswith('HepMC::Asciiv3-END_EVENT_LISTING'):
                break
            if not listing or not line:
                continue

            tag, rest = line[0], line[2:].split()
            if tag == 'E':
                event = int(rest[0])
                positions = {}
            elif tag == 'V':
                # V <id> <status> [<in ids>] @ x y z t
                at = rest.index('@')
                positions[int(rest[0])] = [float(v) for v in rest[at + 1:at + 4]]
            elif tag == 'P':
                vertex = int(rest[1])
                values = [float(v) for v in rest[3:8]]
                out.append('%d %s %s %s %s'
                           % (event, rest[0], rest[2], rest[8],
                              _fmt(values,
                                   positions[vertex] if vertex else None)))
    return out


def dump(path):
    with open(path) as fh:
        head = fh.read(4096)
    if 'HepMC::Asciiv3' in head:
        return dump_hepmc3(path)
    if 'HepMC::IO_GenEvent' in head:
        return dump_hepmc2(path)
    raise SystemExit('%s is neither HepMC 2 nor HepMC 3' % path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print('\n'.join(dump(sys.argv[1])))
