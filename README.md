# Table of Content

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)

# Introduction
Package to study ATLAS performances for tau leptons
In the following README, I put a series of instruction to perform the different steps of the ATLAS tauID tuning

# Setup 
## Setup
Copy the following lines in a setup script (change the python and root version to more adequate version if needed)
This setup script should be sourced everytime you log in.
```bash
#!/bin/bash

if [ -z "$PYTHON_VERSION" ]
then
    PYTHON_VERSION=2.7.4-x86_64-slc6-gcc48
fi

if [ -z "$ROOT_VERSION" ]
then
    ROOT_VERSION=6.02.12-x86_64-slc6-gcc48-opt
fi

function setup_CVMFS() {
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
    source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh --quiet
}

function setup_python()
{
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
    source ${ATLAS_LOCAL_ROOT_BASE}/packageSetups/atlasLocalPythonSetup.sh ${PYTHON_VERSION} --quiet
}

function setup_ROOT()
{
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
    source ${ATLAS_LOCAL_ROOT_BASE}/packageSetups/atlasLocalROOTSetup.sh --rootVersion ${ROOT_VERSION} --skipConfirm --quiet
}

echo "------------> CVMFS"
setup_CVMFS
echo "------------> Python"
setup_python
which  python
echo "------------> ROOT"
setup_ROOT
which root
```

## Install
### virtual environment
```bash
virtualenv imaging_ve
source imaging_ve/bin/activate
```
### root setup
you need a working setup of ROOT 6.

### dependencies
```bash
pip install pip --upgrade
pip install theano
pip install keras
pip install pydot
pip install h5py
pip install sklearn
pip install scikit-image
pip install matplotlib
pip install root_numpy
pip install rootpy
pip install tabulate
```

## Downloading and setup of the tauperf project
The set of commands below allows you to download and use the tauperf package. If you want to contribute to the project and also make your work available publicly you need to fork it on github and work with your own copy of tauperf.

```bash
git clone https://github.com/qbuat/tauperf.git
cd tauperf
git checkout -b imaging origin/imaging
source setup.sh
```
# Usage
## Initial setup at each login
1. Source the initial setup [script](#setup)
1. Go to the `tauperf` directory
1. Source the `setup.sh` script in tauperf

## TO BE WRITTEN
