#!/bin/bash
#
# Generic EarthShineGen gridpack driver.
#
# ExternalLHEProducer always passes nevents, seed and ncpu as the first three
# arguments; anything after that comes from the `args` vstring in the fragment
# and is the parameter point.
#
#   $1 nevents
#   $2 random seed
#   $3 number of cpus                (unused: the generator is vectorised and
#                                     single threaded, and ExternalLHEProducer
#                                     already runs one process per stream)
#   $4 m_X       [GeV]
#   $5 m_A       [GeV]
#   $6 epsilon
#   $7 alpha_X   ('thermal', 'max' or a number)
#   $8 dm_model  ('core', 'floating' or 'monoenergetic')
#
echo "Input arguments: $@"

nevt=${1}
echo "%MSG-EarthShineGen number of events requested = $nevt"

rnum=${2}
echo "%MSG-EarthShineGen random seed used for the run = $rnum"

ncpu=${3}
echo "%MSG-EarthShineGen number of cpus = $ncpu"

MX=${4}
MA=${5}
EPSILON=${6}
ALPHAX=${7}
DMMODEL=${8}

source /cvmfs/cms.cern.ch/cmsset_default.sh
if [ ! -d VERCMSSW/src ]; then
    cmsrel VERCMSSW
fi
cd VERCMSSW/src
cmsenv
cd -

cd EarthShineGen

sed -i "s|^n_events .*|n_events                     ${nevt}|"        parameter.txt
sed -i "s|^seed .*|seed                         ${rnum}|"            parameter.txt
sed -i "s|^m_X .*|m_X                          ${MX}|"               parameter.txt
sed -i "s|^m_A .*|m_A                          ${MA}|"               parameter.txt
sed -i "s|^epsilon .*|epsilon                      ${EPSILON}|"      parameter.txt
sed -i "s|^alpha_X .*|alpha_X                      ${ALPHAX}|"       parameter.txt
sed -i "s|^dm_model .*|dm_model                     ${DMMODEL}|"     parameter.txt

echo "----- parameter.txt -----"
grep -v '^\s*#' parameter.txt | grep -v '^\s*$'
echo "-------------------------"

echo "Running the event generation"
# cmsgrid_final.lhe is the name ExternalLHEProducer expects; the
# generator's own default is the neutral events.lhe.
./EarthShineGen --output_file cmsgrid_final.lhe --report_file earthshinegen_report.txt
echo "Event generation finished"

cat earthshinegen_report.txt

mv cmsgrid_final.lhe ../cmsgrid_final.lhe
cd ../

ls -l cmsgrid_final.lhe
