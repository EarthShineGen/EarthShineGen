#!/bin/bash
#
# Build an EarthShineGen gridpack.
#
# Follows the same shape as blackmax_gridpack.sh: pick a CMSSW release, stage
# the generator into a work directory, drop in a runcmsgrid.sh and tar the lot
# up with xz.  EarthShineGen is pure Python on top of numpy and scipy, both of
# which every recent CMSSW release ships, so there is nothing to compile.
#
# Usage
#   ./earthshinegen_gridpack.sh run3
#       generic gridpack: the mass point is passed at run time, as
#       ExternalLHEProducer arguments
#
#   ./earthshinegen_gridpack.sh run3 <m_X> <m_A> <epsilon> <alpha_X> <dm_model>
#       specific gridpack: the parameters are baked into parameter.txt and the
#       runcmsgrid script takes only nevents/seed/ncpu
#
# Examples
#   ./earthshinegen_gridpack.sh run3
#   ./earthshinegen_gridpack.sh run3 7000 0.23 1e-8 max core
#
set -e

SCRIPTDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRCDIR=$(dirname "$SCRIPTDIR")

if [ "$1" = "run2ul" ]; then
    ISRUN2UL=1
    echo "Running for run2ul condition"
elif [ "$1" = "run3" ]; then
    ISRUN2UL=0
    echo "Running for run3 condition"
else
    echo "Invalid condition: expected run2ul or run3 as the first argument"
    exit 1
fi

if [[ $# -eq 6 ]]; then
    echo "Generating gridpack for a specific parameter point"
    GENERIC=0
    MX=$2
    MA=$3
    EPSILON=$4
    ALPHAX=$5
    DMMODEL=$6
elif [[ $# -eq 1 ]]; then
    GENERIC=1
else
    echo "Invalid number of arguments: expected 1 or 6"
    exit 1
fi

# Set CMSSW version
if [ $ISRUN2UL -eq 1 ]; then
    VERCMSSW=CMSSW_10_6_38
else
    VERCMSSW=CMSSW_14_1_0_pre4
fi

# Set up the environment.  The release is only needed so that the gridpack is
# built and validated against the same python/numpy the job will see.
source /cvmfs/cms.cern.ch/cmsset_default.sh
if [ -r $VERCMSSW/src ] ; then
    echo "release $VERCMSSW already exists"
else
    scram p CMSSW $VERCMSSW
fi
cd $VERCMSSW/src
eval `scram runtime -sh`
cd -

rm -rf gridpack_workdir
mkdir gridpack_workdir
cd gridpack_workdir

# Stage the generator
mkdir EarthShineGen
cp    "$SRCDIR/EarthShineGen"        EarthShineGen/
cp -r "$SRCDIR/earthshinegen"        EarthShineGen/
cp -r "$SRCDIR/data"                 EarthShineGen/
cp    "$SRCDIR/README.md"            EarthShineGen/ 2>/dev/null || true
chmod +x EarthShineGen/EarthShineGen

# A fresh card at the defaults, then the point baked in if this is a specific
# gridpack.  The card is edited by key, never by line number.
./EarthShineGen/EarthShineGen --write-card EarthShineGen/parameter.txt

if [ $GENERIC -eq 1 ]; then
    echo "Generating a generic gridpack"
    cp "$SCRIPTDIR/runcmsgrid_earthshinegen_generic.sh" runcmsgrid.sh
else
    echo "Generating a gridpack for m_X=$MX m_A=$MA epsilon=$EPSILON alpha_X=$ALPHAX model=$DMMODEL"
    sed -i "s|^m_X .*|m_X                          ${MX}|"           EarthShineGen/parameter.txt
    sed -i "s|^m_A .*|m_A                          ${MA}|"           EarthShineGen/parameter.txt
    sed -i "s|^epsilon .*|epsilon                      ${EPSILON}|"  EarthShineGen/parameter.txt
    sed -i "s|^alpha_X .*|alpha_X                      ${ALPHAX}|"   EarthShineGen/parameter.txt
    sed -i "s|^dm_model .*|dm_model                     ${DMMODEL}|" EarthShineGen/parameter.txt
    cp "$SCRIPTDIR/runcmsgrid_earthshinegen_specific.sh" runcmsgrid.sh
fi

sed -i "s|VERCMSSW|${VERCMSSW}|g" runcmsgrid.sh
chmod +x runcmsgrid.sh

# Smoke test before packing: a gridpack that cannot make ten events is not
# worth shipping.
echo "Smoke test..."
( cd EarthShineGen && ./EarthShineGen --n_events 10 --seed 1 \
    --output_file /dev/null --report_file '' --max_trials 2000000 >/dev/null )
echo "Smoke test passed"

# Drop the byte code the smoke test just produced: it is tied to this python
# version, and the job recompiles anyway.
find EarthShineGen -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

if [[ $ISRUN2UL -eq 1 ]]; then
    TARBALL=earthshinegen_gridpack_run2ul.tar.xz
else
    TARBALL=earthshinegen_gridpack_run3.tar.xz
fi
# Packed from inside the work directory so the tarball unpacks flat, the same
# way the BlackMax and Charybdis gridpacks do.
tar -cJpf "../$TARBALL" runcmsgrid.sh EarthShineGen
cd ../
echo "Wrote $(pwd)/$TARBALL"
ls -lh "$TARBALL"
