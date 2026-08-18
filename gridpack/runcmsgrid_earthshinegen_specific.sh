#!/bin/bash
#
# Specific EarthShineGen gridpack driver.
#
# The parameter point is already baked into parameter.txt, so the only things
# that vary job to job are the event count and the seed.  ExternalLHEProducer
# always passes them first.
#
#   $1 nevents
#   $2 random seed
#   $3 number of cpus  (unused; the generator is vectorised and single threaded)
#
echo "Input arguments: $@"

nevt=${1}
echo "%MSG-EarthShineGen number of events requested = $nevt"

rnum=${2}
echo "%MSG-EarthShineGen random seed used for the run = $rnum"

ncpu=${3}
echo "%MSG-EarthShineGen number of cpus = $ncpu"

source /cvmfs/cms.cern.ch/cmsset_default.sh
if [ ! -d VERCMSSW/src ]; then
    cmsrel VERCMSSW
fi
cd VERCMSSW/src
cmsenv
cd -

cd EarthShineGen

sed -i "s|^n_events .*|n_events                     ${nevt}|" parameter.txt
sed -i "s|^seed .*|seed                         ${rnum}|"     parameter.txt

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
