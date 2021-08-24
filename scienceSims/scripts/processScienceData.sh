#!/usr/bin/env bash

DATADIR="./process-science"
RERUN="science"
CORES=10
CLEANUP=true
DEVELOPER=false
usage() {
    echo "Run the PFS 2D pipeline code on the science exposures" 1>&2
    echo "" 1>&2
    echo "Usage: $0 [-d DATADIR] [-r <RERUN>] [-c CORES] [-n] WORKDIR" 1>&2
    echo "" 1>&2
    echo "    -d <DATADIR> : path to raw data (default: ${DATADIR})" 1>&2
    echo "    -r <RERUN> : rerun name to use (default: ${RERUN})" 1>&2
    echo "    -c <CORES> : number of cores to use (default: ${CORES})" 1>&2
    echo "    -n : don't cleanup temporary products" 1>&2
    echo "    -D : developer mode (--clobber-config --no-versions)" 1>&2
    echo "    WORKDIR : directory to use for work"
    echo "" 1>&2
    exit 1
}

while getopts "c:d:Dnr:" opt; do
    case "${opt}" in
        c)
            CORES=${OPTARG}
            ;;
        d)
            DATADIR=${OPTARG}
            ;;
        D)
            DEVELOPER=true
            ;;
        n)
            CLEANUP=false
            ;;
        r)
            RERUN=${OPTARG}
            ;;
        h | *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))
WORKDIR=$1; shift
if [ -z "$WORKDIR" ] || [ -n "$1" ]; then
    usage
fi
HERE=$(unset CDPATH && cd "$(dirname "$0")" && pwd)

set -evx

# Set up the data repo and ingest all data
mkdir -p $WORKDIR
mkdir -p $WORKDIR/CALIB
echo "lsst.obs.pfs.PfsMapper" > $WORKDIR/_mapper
ingestPfsImages.py $WORKDIR $DATADIR/PFFA*.fits


# Run the pipeline on brn
generateCommands.py $WORKDIR \
    $HERE/../config/science.yaml \
    $WORKDIR/pipeline_on_brn.sh \
    --rerun=$RERUN/pipeline/brn \
    --blocks=pipeline_on_brn \
    -j $CORES $develFlag

sh $WORKDIR/pipeline_on_brn.sh

# Run the pipeline on bmn
generateCommands.py $WORKDIR \
    $HERE/../config/science.yaml \
    $WORKDIR/pipeline_on_bmn.sh \
    --rerun=$RERUN/pipeline/bmn \
    --blocks=pipeline_on_bmn \
    -j $CORES $develFlag

sh $WORKDIR/pipeline_on_bmn.sh
