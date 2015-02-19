#!/bin/bash


function setup_CVMFS() {
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
    source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
}


function setup_ROOT_cvmfs()
{
    # 5.34.25-x86_64-slc6-gcc48-opt #
    export rootVersion=6.02.05-x86_64-slc6-gcc48-opt
    source ${ATLAS_LOCAL_ROOT_BASE}/packageSetups/atlasLocalROOTSetup.sh --rootVersion ${rootVersion}
}

function setup_ROOT_sfu()
{
    # setup ROOT, Python etc.
    #export ROOT_VERSION=5.34.09.hsg7 #5.34.07
    # export ROOT_VERSION=5.34.10 # skims
    #export ROOT_VERSION=5.34.head
    # export ROOT_VERSION=5.34.14
    export ROOT_VERSION=5.34.18 # new TMVA
    source /atlas/software/bleedingedge/setup.sh
}

function setup_PYTHON()
{
    # setup Python user base area
    export PATH=${HOME}/.local/bin${PATH:+:$PATH}

    # setup Python module installed in Noel's area
    echo "-- Quentin, you've got to do your own setup !"
    export PYTHONUSERBASE=/cluster/data10/endw/local/sl5
    export PATH=${PYTHONUSERBASE}/bin${PATH:+:$PATH}
}


source /atlas/software/bleedingedge/setup.sh
echo 'Start by CVMFS'
setup_CVMFS
echo 'Then, ROOT'
setup_ROOT_cvmfs
echo 'And, python'
setup_PYTHON