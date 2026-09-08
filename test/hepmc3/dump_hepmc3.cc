// Dump a HepMC3 file with the real HepMC3 library, in a canonical form that
// can be diffed against what EarthShineGen thinks it wrote.
//
// EarthShineGen writes HepMC by hand -- its only dependencies are numpy and
// scipy -- so the thing worth checking is that the bytes it produces are what
// the library reads back.  This is the other half of that check;
// canonical_dump.py produces the same lines from the file directly.
//
//   ./check_with_hepmc3.sh events.hepmc3
//
// Output, one line per particle, fields separated by single spaces:
//
//   <event> <id> <pdg> <status> <px> <py> <pz> <e> <m> <vx> <vy> <vz>
//
// with the production vertex position in mm, or "none" for a particle that has
// no production vertex.  Momenta are printed at 15 significant digits, the
// precision EarthShineGen writes.

#include <cstdio>
#include <memory>
#include <string>

#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/Units.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: dump_hepmc3 <file.hepmc3>\n");
    return 2;
  }

  HepMC3::ReaderAscii reader(argv[1]);
  if (reader.failed()) {
    std::fprintf(stderr, "could not open %s\n", argv[1]);
    return 2;
  }

  int n_events = 0;
  while (!reader.failed()) {
    HepMC3::GenEvent evt(HepMC3::Units::GEV, HepMC3::Units::MM);
    reader.read_event(evt);
    if (reader.failed())
      break;
    ++n_events;

    for (const auto& p : evt.particles()) {
      const auto& mom = p->momentum();
      std::printf("%d %d %d %d %+.15e %+.15e %+.15e %+.15e %+.15e",
                  evt.event_number(), p->id(), p->pid(), p->status(),
                  mom.px(), mom.py(), mom.pz(), mom.e(),
                  p->generated_mass());
      // A particle the file declares with production vertex 0 -- a beam
      // particle -- does not come back with a null production_vertex().  The
      // reader hangs it off a placeholder vertex at the origin, and it does
      // that to files the HepMC3 library wrote itself, so it is the reader's
      // convention and not something about how this file was produced.  The
      // placeholder is the one with id 0; real vertices have negative ids.
      const auto& vtx = p->production_vertex();
      if (vtx && vtx->id() != 0) {
        const auto& pos = vtx->position();
        std::printf(" %+.15e %+.15e %+.15e\n", pos.x(), pos.y(), pos.z());
      } else {
        std::printf(" none\n");
      }
    }
  }
  reader.close();

  std::fprintf(stderr, "read %d events\n", n_events);
  return n_events > 0 ? 0 : 1;
}
