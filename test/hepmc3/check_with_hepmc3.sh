#!/bin/bash
#
# Read an EarthShineGen HepMC 3 file back with the real HepMC3 library and
# check it says what we think it says.
#
#   ./test/hepmc3/check_with_hepmc3.sh [file.hepmc3]
#
# With no argument a small sample is generated first.  Needs HepMC3 on the
# system, found in this order: HepMC3-config on PATH, HEPMC3_DIR, a cvmfs CMSSW
# external, a distribution install (libhepmc3-dev).  On a machine with cvmfs,
# `cmsenv` in any CMSSW area is enough.
#
# With no HepMC3 anywhere this exits 0 with a SKIP, so it can be dropped into a
# test run on a machine that does not have it.  Set REQUIRE_HEPMC3=1 -- as CI
# does -- to make a missing library a failure instead, otherwise a job with a
# broken install goes green having checked nothing.
#
# This is the half of the HepMC 3 check that needs the library.  The other half
# -- structure, and agreement with the HepMC 2 output of the same run -- is in
# test/run_tests.py and needs nothing but numpy.
#
set -eu

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
WORK=$(mktemp -d -t earthshinegen_hepmc3_XXXXXX)
trap 'rm -rf "${WORK}"' EXIT

CONFIG=$(command -v HepMC3-config || true)
if [ -z "${CONFIG}" ] && [ -n "${HEPMC3_DIR:-}" ]; then
    CONFIG="${HEPMC3_DIR}/bin/HepMC3-config"
fi
if [ -z "${CONFIG}" ] || [ ! -x "${CONFIG}" ]; then
    # cvmfs.  The arch has to be pinned: the tree carries ppc64le and aarch64
    # builds too, and picking one of those gets you an "incompatible library,
    # cannot find -lHepMC3" from the linker rather than anything informative.
    # CMSSW spells x86_64 as amd64 in its arch strings; uname does not.
    case "$(uname -m)" in
        x86_64) cms_machine=amd64 ;;
        *)      cms_machine=$(uname -m) ;;
    esac
    for arch in "${SCRAM_ARCH:-}" "el9_${cms_machine}_gcc14" "el8_${cms_machine}_gcc14"; do
        [ -n "${arch}" ] || continue
        CONFIG=$(ls -d "/cvmfs/cms.cern.ch/${arch}/external/hepmc3/"*/bin/HepMC3-config \
                 2>/dev/null | tail -1 || true)
        [ -n "${CONFIG}" ] && break
    done
fi

if [ -n "${CONFIG}" ] && [ -x "${CONFIG}" ]; then
    CXXFLAGS=$(${CONFIG} --cxxflags)
    LDFLAGS=$(${CONFIG} --ldflags)
    RUNPATH="$(${CONFIG} --prefix)/lib64:$(${CONFIG} --prefix)/lib"
    echo "using HepMC3 $(${CONFIG} --version) from $(${CONFIG} --prefix)"
elif [ -f /usr/include/HepMC3/GenEvent.h ]; then
    # A distribution install (Debian/Ubuntu libhepmc3-dev) ships the headers
    # and the library but not HepMC3-config.
    CXXFLAGS=""
    LDFLAGS="-lHepMC3"
    RUNPATH=""
    echo "using the system HepMC3 from /usr/include/HepMC3"
elif [ -n "${REQUIRE_HEPMC3:-}" ]; then
    echo "ERROR: REQUIRE_HEPMC3 is set but no HepMC3 installation was found" >&2
    exit 1
else
    echo "SKIP: no HepMC3 installation found (set HEPMC3_DIR)" >&2
    exit 0
fi

if [ $# -ge 1 ]; then
    FILE=$1
else
    FILE="${WORK}/events.hepmc3"
    python3 "${ROOT}/EarthShineGen" \
        --n_events 25 --seed 4242 --output_format hepmc --hepmc_version 3 \
        --hepmc_file "${FILE}" --report_file '' --max_trials 400000 \
        --ms_model highland > "${WORK}/gen.log" 2>&1
fi

# shellcheck disable=SC2086
g++ -std=c++17 -O1 ${CXXFLAGS} "${HERE}/dump_hepmc3.cc" \
    -o "${WORK}/dump_hepmc3" ${LDFLAGS}

LD_LIBRARY_PATH="${RUNPATH}:${LD_LIBRARY_PATH:-}" \
    "${WORK}/dump_hepmc3" "${FILE}" > "${WORK}/from_library.txt"
python3 "${HERE}/canonical_dump.py" "${FILE}" > "${WORK}/from_file.txt"

if diff -u "${WORK}/from_file.txt" "${WORK}/from_library.txt" > "${WORK}/diff.txt"
then
    echo "HepMC3 read back $(wc -l < "${WORK}/from_library.txt") particles, all"
    echo "matching the file byte for byte at 15 significant digits"
else
    echo "MISMATCH between the file and what HepMC3 read from it:" >&2
    head -40 "${WORK}/diff.txt" >&2
    exit 1
fi
