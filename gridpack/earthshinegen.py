import FWCore.ParameterSet.Config as cms

# Generic gridpack: the parameter point is passed through `args`, after the
# tarball path.  Order matches runcmsgrid_earthshinegen_generic.sh:
#     m_X [GeV], m_A [GeV], epsilon, alpha_X, dm_model
externalLHEProducer = cms.EDProducer(
    'ExternalLHEProducer',
    args = cms.vstring(
        '/cvmfs/cms.cern.ch/phys_generator/gridpacks/test_earthasdm/'
        'earthshinegen_gridpack_run3.tar.xz',
        '7000',      # m_X [GeV]
        '0.23',      # m_A [GeV]
        '1e-8',      # epsilon
        'max',       # alpha_X: 'thermal', 'max', or a number
        'core',      # dm_model: 'core', 'floating', 'monoenergetic'
    ),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(6),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath(
        'GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'),
    # EarthShineGen is single threaded per invocation; concurrent generation
    # splits the request across streams, each with its own seed.
    generateConcurrently = cms.untracked.bool(True),
)
