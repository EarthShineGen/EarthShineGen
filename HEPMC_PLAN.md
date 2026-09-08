# Plan: HepMC output for EarthShineGen, validated in CMSSW_20_1_0_pre3

> **Status: implemented and validated.** See [Outcome](#outcome) at the end for
> the decisions taken, what changed against this plan, and the results.


Branch: `hepmc-output` (created off `main`, commit `922ebb7` equivalent of EarthShineGen main).
Repo: `/home/users/tvami/EarthAsDM/EarthShineGen`
CMSSW: `/home/users/tvami/EarthAsDM/CMSSW_20_1_0_pre3/src` (arch `el8_amd64_gcc14`)

---

## 0. Why this is worth doing (and what format to target)

The LHE format has no vertex field. `earthshinegen/lhe.py` works around that by
writing `#vertex`, `#vertex_mu1`, `#vertex_mu2`, `#decay_vertex` as comment
lines inside each `<event>` block, and `gridpack/run3_fragment.py` documents at
length that a custom CMSSW producer would have to be written to put the muons
back where they belong. That whole workaround exists only because of the
format.

HepMC carries a production vertex per particle. The two-vertex topology that
this signal actually has -- two muons crossing the hand-off surface at two
different points, typically ~cm apart with `ms_model none` and ~0.5 m apart
with `ms_model highland` -- is expressible natively, and CMSSW consumes it with
no new C++ at all.

### Which HepMC flavour

Established by inspection of the release, not assumed:

* The only file-based generator input source in `CMSSW_20_1_0_pre3` is
  `MCFileSource` (`IOMC/Input/src/MCFileSource.cc`). It goes through
  `HepMCFileReader`, which opens the file with **`HepMC::IO_GenEvent`**
  (HepMC **2**, `IOMC/Input/src/HepMCFileReader.cc:62`) and produces
  `edm::HepMCProduct` + `GenEventInfoProduct`.
* A release-wide grep for `ReaderAscii` / `deduce_reader` / HepMC3 readers
  returns nothing: **there is no HepMC3 file input source in this release.**
  HepMC3 appears only inside hadronizer interfaces (Pythia8/Herwig7/Sherpa
  `*HepMC3Hadronizer`), `HepMC3Product`, and Rivet.
* Externals available: `hepmc` 2.06.10 and `hepmc3` 3.3.1.

=> **Primary target is HepMC2 ASCII (`IO_GenEvent`), because that is the one
CMSSW can read from a file today.** A HepMC3 ASCII writer is cheap to add on
top of the same event-building code and is useful for Rivet/standalone tools,
so it is included as a secondary, non-blocking deliverable.

Reference format (HepMC 2.06, from `IOMC/Input/test/*.hepmc` and
`HepMC/StreamInfo.h`):

```
HepMC::Version 2.06.10
HepMC::IO_GenEvent-START_EVENT_LISTING
E <evt> <mpi> <scale> <a_qcd> <a_qed> <sig_proc_id> <sig_vtx_bc> <n_vtx> <beam1_bc> <beam2_bc> <n_rnd> <n_wgt> [wgts]
N <n_wgt> "0"
U GEV MM
C <xsec_pb> <xsec_err_pb>
V <bc> <id> <x> <y> <z> <ctau> <n_orphans_in> <n_out> <n_wgt>
P <bc> <pdg> <px> <py> <pz> <E> <m> <status> <pol_th> <pol_ph> <end_vtx_bc> <n_flow>
...
HepMC::IO_GenEvent-END_EVENT_LISTING
```

Units are `GEV MM` -- the same mm convention `lhe.py` already uses for its
comment lines, so the `MM_PER_M = 1000.0` conversion carries over unchanged.

---

## 1. Scope decisions

**D1 (needs your call, recommendation given).** Default of the new
`output_format` card key:

* `lhe` -- byte-identical to today for every existing card.
* **`both` (recommended)** -- writes `events.lhe` and `events.hepmc`; existing
  workflows keep working, HepMC comes for free, cost is one extra file.
* `hepmc` -- HepMC only.

I will implement all three and default to `both` unless you say otherwise.
Nothing about the physics, the RNG stream or the accepted-event sequence
changes with this key, so `lhe` output stays bit-for-bit identical to `main`
in every case (this is asserted by a test, see 5.3).

**In scope:** the writer, the card keys, the wiring in `generator.py`, unit
tests, CMSSW read-back validation, docs.

**Explicitly out of scope (called out, not silently dropped):**
`ExternalLHEProducer` cannot consume HepMC. So the HepMC route is *not* a
gridpack drop-in: it is a standalone generation step whose output is fed to
`MCFileSource` as the CMSSW input source, with Pythia dropped from the chain
entirely (which is correct here -- the current fragment already switches every
Pythia stage off and uses it as a pass-through). I will ship a worked
`MCFileSource` cfg fragment and document the difference, but I will not
restructure the gridpack machinery on this branch.

---

## 2. New module: `earthshinegen/hepmc.py`

Mirrors `LHEWriter`'s public API exactly so `generator.py` can hold a list of
writers and not care which is which:

```python
class HepMCWriter:
    def __init__(self, path, params, xsec_pb, xsec_err_pb=0.0,
                 version='2', topology='split',
                 include_initial=True, include_mother=True, ...)
    def write_event(self, p1, p2, vertex_m, vertex1_m=None, vertex2_m=None,
                    decay_vertex_m=None, weight=1.0)
    def update_cross_section(self, xsec_pb, xsec_err_pb=0.0, max_weight=1.0)
    def close(self)          # + __enter__/__exit__
```

Pure Python + numpy, no new dependencies (consistent with the "numpy+scipy
only, nothing else in cvmfs" constraint that the rest of the package respects).

### 2.1 Topology

`hepmc_topology = split` (default when `stage = detector`):

| vertex | position | in | out |
|---|---|---|---|
| V-1 | A' decay point in the rock | A' (status 2) [+ the two mock initial particles if `include_initial`] | mu1, mu2 **status 2**, undegraded |
| V-2 | muon 1 entry point on the hand-off surface | mu1 undegraded | mu1 **status 1**, degraded |
| V-3 | muon 2 entry point | mu2 undegraded | mu2 **status 1**, degraded |

This is the honest record: V-2/V-3 stand for the accumulated energy loss and
multiple scattering over the rock path, and the two status-1 muons -- the only
things GEANT will track -- start exactly on the hand-off surface with exactly
the arriving momenta. Four-momentum is deliberately *not* conserved at V-2/V-3
(that is what energy loss means); HepMC has no conservation check, and Pythia
is not in this chain, so nothing rejects it.

`hepmc_topology = single`: one vertex, at the midpoint of the two entry points,
reproducing the LHE record one-for-one. Kept as an escape hatch and as the
thing the LHE-vs-HepMC equivalence test compares against.

`stage = vertex`: always a single vertex at the decay point, muons status 1,
undegraded -- there is no second surface to speak of.

Note: the `select_batch` return already carries `mu1_entry`/`mu2_entry` and
both the degraded and undegraded four-momenta, but `run()` currently only hands
the writer the degraded pair. `run()` will be extended to pass the undegraded
pair too (`batch['mu1']`/`batch['mu2']` at the accepted indices), which is a
small addition to the dict returned by `select_batch` (`mu1_raw`, `mu2_raw`).
The LHE writer ignores the extra arguments.

### 2.2 Barcodes, status codes, bookkeeping

* Vertex barcodes negative (`-1, -2, -3`), particle barcodes positive from 1.
* Status: `4` for the mock incoming particles (HepMC convention for beam
  particles; LHE used `-1`), `2` for the A' and for the intermediate
  undegraded muons, `1` for the final-state muons.
* `E` line: `n_vtx` and beam barcodes filled correctly; `signal_process_id`
  set to a fixed EarthShineGen id, weight list `1` weight = the event weight.
* `U GEV MM` written on every event.
* Momenta at the same 15-significant-digit precision `lhe.py` uses -- the
  reason there is the same (TeV muons with a 105 MeV mass; the recovered mass
  is wrong at the per-cent level at 11 digits, and `HepMCProduct`/G4 read the
  mass back).

### 2.3 Cross section and run metadata

* `C <xsec> <err>` per event carries the observable rate the same way `XSECUP`
  does in the LHE (events per year of live time). Because HepMC2 puts it in
  every event rather than in a header, `update_cross_section` records the byte
  offset of each fixed-width `C` line and patches them at `close()` -- the same
  seek-and-overwrite trick `LHEWriter.update_cross_section` already uses, just
  N times. (CMSSW's `MCFileSource` ignores the `C` line, so this is for
  standalone consumers; it is a dozen lines and keeps parity with the LHE path.)
* The full parameter/rate header that LHE carries in `<header>` has no HepMC2
  equivalent. It goes into a `HepMC::IO_GenEvent-COMMENT` block ahead of the
  event listing **if** the reader tolerates it -- this is verified in step 6.1
  before being relied on. Fallback if it does not: a sidecar
  `<output>.hepmc.info` file with the same content. Either way
  `earthshinegen_report.txt` is unaffected and remains the authoritative
  human-readable record.
* HepMC3 mode gets the same metadata as proper `A` (attribute) records, which
  is clean and needs no fallback.

---

## 3. Card changes (`earthshinegen/card.py`)

New keys in the `output` section of `SCHEMA`:

| key | type | default | meaning |
|---|---|---|---|
| `output_format` | str | `both` (see D1) | `lhe`, `hepmc` or `both` |
| `hepmc_file` | str | `events.hepmc` | HepMC output path |
| `hepmc_version` | str | `2` | `2` = `IO_GenEvent` ASCII (CMSSW-readable); `3` = HepMC3 ASCII |
| `hepmc_topology` | str | `split` | `split` (per-muon vertices) or `single` (LHE-equivalent) |

Plus: allowed-value checks in `_validate`, the keys added to the `output`
section of `write_template`, a regenerated `parameter.txt`, and the same
options exposed on the command line automatically (the `SCHEMA` loop in
`EarthShineGen` already does that with no change).

Validation rules: `hepmc_version 3` is accepted but warns that
`CMSSW_20_1_0_pre3` has no HepMC3 file source; `hepmc_topology split` with
`stage vertex` is silently equivalent to `single`.

---

## 4. Wiring (`earthshinegen/generator.py`)

* `generate()` builds a list of writers from `output_format` instead of the
  single `with LHEWriter(...)` block (a small `contextlib.ExitStack`, or an
  explicit try/finally to stay 3.6-compatible -- there are `cpython-36` pyc
  files in the tree, so I will not assume anything newer than 3.6).
* `run()` loops over writers per accepted event. Writer order does not touch
  the RNG, so the accepted-event sequence is unchanged.
* `update_cross_section` is called on every writer.
* The final log line reports every file written.

---

## 5. Tests (`test/run_tests.py`, same harness and style)

1. `test_hepmc_record_is_self_consistent` -- parse the written file back with a
   small in-file parser: the `E` line's vertex count matches the `V` lines,
   every `P` points at a declared end vertex or 0, barcodes are unique, the
   file opens and closes with the right keys.
2. `test_hepmc_momenta_and_masses` -- muon four-momenta on shell to 1e-9,
   pair invariant mass equals `m_A` for the `vertex` stage.
3. `test_hepmc_matches_lhe_event_for_event` -- run once with
   `output_format both`, `hepmc_topology single`, and assert the two files
   agree event for event on momenta and on the vertex position (mm).
4. `test_hepmc_split_topology_places_muons_at_their_entry_points` -- the
   status-1 muons' production vertices equal `#vertex_mu1` / `#vertex_mu2` from
   the LHE run at the same seed, and their energies are below the status-2
   parents' (energy loss went the right way).
5. `test_lhe_output_is_unchanged` -- generate with a fixed seed under
   `output_format lhe` and under `both`, and assert the LHE files are
   byte-identical. This is the no-regression guard for D1.
6. Extend `test_end_to_end_every_model_and_stage` to cover both formats.

---

## 6. CMSSW validation

All of this runs against the release plugins; **no CMSSW C++ has to be written
or compiled** for the validation itself. `IOMC/Input`, `GeneratorInterface`
and `SimDataFormats` are already checked out in the work area; if a step turns
out to need one more package it is `git cms-addpkg PhysicsTools/HepMCCandAlgos`
(for `ParticleListDrawer`) and/or `SimG4Core/Generator` (only for step 6.5).

The cfgs live in the EarthShineGen repo under `test/cmssw/` so they are version
controlled with the generator, and are copied/run from the CMSSW work area.

**6.1 Format acceptance.** Generate ~200 events, run

```
cmsRun test/cmssw/read_hepmc_cfg.py inputFiles=file:events.hepmc
```

with `source = MCFileSource`, `maxEvents = -1`, a `PoolOutputModule`.
Pass = all 200 events read, no exception, `HepMCProduct` in the output.
This is also where the `COMMENT`-block question from 2.3 is settled.

**6.2 Content check.** Add `genParticles` (`GenParticleProducer`, fed from the
`generator` `HepMCProduct`) and `ParticleListDrawer` to the same cfg. Pass =
the printed table shows, per event, the A' and the two muons with the right
PDG IDs, the right status codes, and momenta matching the generator's own
report to the printed precision.

**6.3 Vertex check -- the actual point of the exercise.** A tiny analyzer step
(or a FWLite/`edmDumpEventContent` + PyROOT read of the `HepMCProduct`)
comparing the status-1 muons' production vertices in the CMSSW event against
the `#vertex_mu1` / `#vertex_mu2` numbers written by the LHE writer at the same
seed. Pass = agreement to double precision in mm, i.e. the per-muon entry
points survived the round trip that LHE could not carry at all.

**6.4 Both stages.** Repeat 6.1-6.3 for `stage vertex` and `stage detector`,
and for `ms_model none` and `highland` (the case where the two entry points
are metres apart and the single-vertex approximation genuinely fails).

**6.5 GEN-SIM (stretch, flagged optional).** `cmsDriver.py` GEN-SIM step with
`MCFileSource`, `VtxSmeared` dropped (it would overwrite the per-event
vertices, which is the whole reason for this work), and cosmics-style
`SimG4Core` `Generator` settings (`ApplyPCuts = False`, `ApplyEtaCuts = False`)
so muons produced 7.5 m off-axis are not cut. Known risk: `SimG4Core`'s
`Generator` applies acceptance and world-volume checks to primaries; if it
rejects vertices on the hand-off cylinder that is a *downstream* configuration
issue, not a format issue, and I will report it as a finding rather than
silently expanding scope into simulation configuration.

Steps 6.1-6.4 are the deliverable "CMSSW can read it". 6.5 is best-effort.

---

## 7. Documentation

* `README.md`: a new output-format section; rewrite the "LHE has one vertex per
  event" bullet (line ~227) to say that HepMC solves it and how; add
  `hepmc.py` to the module map (line ~339).
* `gridpack/run3_fragment.py`: keep the LHE workaround text (still true for the
  LHE route), and add a pointer to the HepMC route.
* New `gridpack/mcfilesource_fragment.py`: the worked `MCFileSource`-based cfg,
  with the ExternalLHEProducer incompatibility stated up front.
* `earthshinegen/__init__.py` docstring: "writes Les Houches events" -> both.

---

## 8. Deliverables

```
earthshinegen/hepmc.py                  new writer (HepMC2 + HepMC3)
earthshinegen/card.py                   4 new keys, validation, template
earthshinegen/generator.py              multi-writer wiring
earthshinegen/__init__.py               docstring
parameter.txt                           regenerated
test/run_tests.py                       6 new/extended tests
test/cmssw/read_hepmc_cfg.py            MCFileSource read-back
test/cmssw/dump_hepmc_vertices.py       vertex round-trip check
gridpack/mcfilesource_fragment.py       CMSSW fragment for the HepMC route
README.md                               docs
HEPMC_PLAN.md                           this file
```

## 9. Acceptance criteria

1. `./test/run_tests.py` passes, including the byte-identical-LHE regression test.
2. `./EarthShineGen --output_format hepmc` produces a file that `cmsRun` +
   `MCFileSource` reads end to end with no exception.
3. The status-1 muon production vertices in the CMSSW event equal the entry
   points EarthShineGen generated, in mm, for both `ms_model` settings.
4. `ParticleListDrawer` shows the expected PDG IDs, statuses and momenta.
5. Nothing about the existing LHE output or the rate calculation changes.

## 10. Open questions for you

* **D1**: default `output_format` -- `both` (my recommendation), `lhe`, or `hepmc`?
* Do you want the HepMC3 writer on this branch at all, or HepMC2 only until
  CMSSW grows a HepMC3 source?
* Should step 6.5 (GEN-SIM) be attempted on this branch, or split out?

---

# Outcome

Answers given in the first round: `output_format` defaults to **`both`**;
HepMC 2 first; **step 6.5 done**.

A second round then added the **HepMC 3 writer** (`hepmc_version`, default
still `2` because that is the only one CMSSW reads from a file), moved every
CMSSW-specific file out to the
[InterfaceWithExperiments](https://github.com/EarthShineGen/InterfaceWithExperiments)
repository so this one stays experiment-neutral, and added GitHub Actions CI.
The file list below is the state after that.

## What changed against the plan

**The intermediate particles are status 3, not the status 2 the plan assumed.**
This was found while doing 6.5 and it is the one substantive correction.
`SimG4Core/Generators/src/Generator.cc` hands GEANT every status-2 particle
whose end vertex is outside the beampipe (`RDecLenCut`, 2.9 cm), as a primary
with a predefined decay. Our muons-as-produced have end vertices 7.5 m out, so
with status 2 GEANT would have started tracking them at the A' decay point, a
kilometre underground and outside any CMS volume. Status 3 is what CMSSW reads
as "decayed by the generator, do not propagate", which is exactly true here.
`test_hepmc_record_is_self_consistent` now asserts no particle is status 2.

**A related trap, documented in the GEN-SIM cfg rather than worked around.**
The same file computes
`fFiductialCuts = fPCuts || fPtransCut || fEtaCuts || fPhiCuts` and starts every
particle from `toBeAdded = !fFiductialCuts`. Turning *all* the cuts off -- the
obvious thing to do for muons entering 7.5 m off-axis -- therefore makes GEANT
track the entire record, mock beams and dark photon included. `ApplyPCuts`
stays on with the window opened instead.

**`hepmc_version` was dropped from the card**, since only HepMC 2 is written.
The other three keys landed as planned: `output_format`, `hepmc_file`,
`hepmc_topology`.

**The equivalence test compares from `<init>` onwards, not the whole file.**
The LHE `<header>` records the parameter card, which now mentions the output
format, so it necessarily differs between `output_format lhe` and `both`. The
events do not: verified byte for byte, and separately against `main`.

## Results

* `./test/run_tests.py`: **43/43**, including four new HepMC tests and the
  no-regression guard on the LHE output.
* LHE events byte-identical to `main` at the same seed (40 events, 49515 bytes).
* `test/cmssw/run_cmssw_validation.sh`: **6/6 configurations**, 50 events each.
  Momenta, statuses and per-particle production vertices match exactly against
  the `HepMCProduct` and to a part in 10^6 against `reco::GenParticle`
  (`Double32_t` on disk, so no tighter is possible).
* GEN-SIM: 20/20 events, GEANT taking exactly the two arriving muons as
  primaries at their own crossings; 20/20 leave muon-system hits, 18/20 leave
  tracker hits.

## Files, as built

In this repo (experiment-neutral):

```
earthshinegen/hepmc.py                    the writers, HepMC 2 and 3
earthshinegen/card.py                     output_format, hepmc_file,
                                          hepmc_version, hepmc_topology
earthshinegen/generator.py                _open_writers, multi-writer event loop
earthshinegen/lhe.py                      write_event takes (and ignores) p1_raw/p2_raw
earthshinegen/__init__.py, README.md      docs
parameter.txt                             regenerated
test/run_tests.py                         7 new tests, 2 extended
test/hepmc3/dump_hepmc3.cc                reads the file with the real library
test/hepmc3/canonical_dump.py             the same, parsed independently
test/hepmc3/check_with_hepmc3.sh          diffs the two
.github/workflows/tests.yml               CI
gridpack/*.sh                             LHE route pinned to output_format lhe
```

In the InterfaceWithExperiments repo (everything CMS):

```
cmssw/read_hepmc_cfg.py                   MCFileSource read-back
cmssw/check_hepmc_roundtrip.py            the numerical round trip
cmssw/run_read_validation.sh              the 6-configuration driver
cmssw/gensim_cfg.py                       GEN-SIM from a .hepmc file
cmssw/check_gensim.py                     SimTrack/SimVertex/SimHit checks
cmssw/run_gensim.sh                       the two steps plus the check
cmssw/fragment_mcfilesource.py            the CMSSW fragment for the HepMC route
docs/rpc-stepping.md                      the endcap step-limit finding
```

## Known, and left as findings

* Some muons are killed by GEANT's 20000-step limit in the endcap chambers.
  Measured afterwards: 3 of 200 at the release default, 0 of 200 at 200000,
  for 3.5% more CPU; 0 of 200 either way once the sample is selected on the
  inner detector. Written up, with a plan, in the interface repository's
  `docs/rpc-stepping.md`.
* `SimG4Core` records the *first* vertex of the event as the SimEvent
  collision point, which for the split topology is the A' decay point deep in
  the rock. It is metadata only, and arguably the honest answer.
* The `require_hit detector` default only asks that the muon reach the hand-off
  surface -- which is the detector's own outer cylinder, so it is satisfied by
  construction and most such muons clip the outside without entering CMS. A
  GEN-SIM sample wants `--require_hit inner_detector --require_both_muons 1`
  and a hand-off half length matched to the real detector.
