import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import *

# EarthShineGen writes two final-state muons and nothing else: no colour, no
# beam remnant, no underlying event.  Pythia is kept in the chain only because
# it is what ExternalLHEProducer feeds, so everything that would dress the
# event is switched off and it acts as a pass-through.
generator = cms.EDFilter(
    "Pythia8HadronizerFilter",
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'PartonLevel:ISR = off',
            'PartonLevel:FSR = off',
            'PartonLevel:MPI = off',
            'PartonLevel:Remnants = off',
            'HadronLevel:all = off',
            'Check:event = off',       # the mock initial state is not a pp collision
            'Beams:frameType = 4',
        ),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CP5Settings',
            'processParameters',
        )
    ),
    comEnergy = cms.double(13600.),
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1),
)

ProductionFilterSequence = cms.Sequence(generator)

# ---------------------------------------------------------------------------
# THE VERTEX
# ---------------------------------------------------------------------------
# Read this before running the detector stage through GEN-SIM.
#
# EarthShineGen's 'detector' stage puts each muon pair on the hand-off surface
# around the detector (by default the detector's own outer cylinder, whose size
# is set on the parameter card), travelling inward and upward -- the same kind
# of target surface CosMuoGenSource uses.  That position is per event and it is
# the whole point of the signal, so the standard beamspot smearing is wrong
# here: it would put both muons at the interaction point, flying outward.
#
# LHE has no vertex field, so the position travels in comment lines inside each
# <event> block, which ExternalLHEProducer preserves in
# LHEEventProduct::comments():
#
#     #vertex        x y z    midpoint of the two muon entry points  [mm]
#     #vertex_mu1    x y z    where muon 1 crosses the surface       [mm]
#     #vertex_mu2    x y z    where muon 2 crosses the surface       [mm]
#     #decay_vertex  x y z    the true A' decay point in the rock    [mm]
#
# Two ways to use them:
#
#   1. Replace VtxSmeared with a small EDProducer that reads '#vertex' out of
#      the LHEEventProduct comments and sets the HepMC event vertex from it.
#      This is a few dozen lines and gives the single-vertex topology.
#
#   2. For the exact two-vertex topology, consume '#vertex_mu1' and
#      '#vertex_mu2' and place each muon at its own point.  The two crossings
#      are centimetres apart at these opening angles, so this only matters if
#      the analysis is sensitive to the muon separation at entry.
#
# Until such a producer exists, run with
#
#     process.VtxSmeared = cms.EDProducer("HLLHCEvtVtxGenerator")  # NO
#
# replaced by no smearing at all,
#
#     from Configuration.StandardSequences.VtxSmeared import VtxSmeared
#     process.VtxSmeared.src = ...   # or drop VtxSmeared from the path
#
# and treat the resulting sample as unplaced.  The 'vertex' stage
# (--stage vertex) writes the decay point instead and is the right input for
# acceptance studies that do their own geometry.
