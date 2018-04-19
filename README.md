# Table of Content

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)

# Introduction
Package to study ATLAS performances for tau leptons. This branch is dedicated to the development of tau imaging algorithms.

# Setup 
## Getting started on lxplus
```bash
git clone https://github.com/qbuat/tauperf.git
cd tauperf
git checkout -b imaging origin/imaging
source setup_lxplus.sh
```
NB: training seems to be broken on lxplus, still trying to figure out why...

## Install using a virtual environment
TODO: provide recipe for anaconda

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
pip install pydot_ng
pip install h5py
pip install tables
pip install sklearn
pip install scikit-image
pip install matplotlib
pip install root_numpy
pip install rootpy
pip install tabulate
```
### tauperf project: imaging branch
```bash
git clone https://github.com/qbuat/tauperf.git
cd tauperf
git checkout -b imaging origin/imaging
```
# Usage
## Creating your own setup script
1. Copy the file [setup.sh](setup.sh)
1. Edit the ROOT setup
1. Edit the variables `DATA_AREA` and `VE_PATH` 

## Data (as of Apr. 18th, 2018)
### Flat root ntuples:
```
user.qbuat.mc16_13TeV.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight.recon.ESD.e5468_s3170_r9466.tauid.v2_OUT
```

### root and hdf5 files with selected tau candidates:
```
/eos/atlas/user/q/qbuat/IMAGING/v13/test/output.selected.root
/eos/atlas/user/q/qbuat/IMAGING/v13/test/output.selected.h5
```

### hdf5 files containing the formated images
```
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_1p0n.h5
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_1p1n.h5
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_1p2n.h5
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_3p0n.h5
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_3p1n.h5
```

## Processing/training/testing
see the [workflow] (doc/workflow.md)
