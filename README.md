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

## Data (as of Nov. 6th, 2016)
### Flat root ntuples:
```
user.qbuat.mc15_13TeV.361108.PowPy8EvtGen_AZNLOCTEQ6L1_Ztautau.recon.ESD.e3601_s2650_s2183_r7823.tauid.v6_OUT
```
### hdf5 file:
```
/afs/cern.ch/user/q/qbuat/work/public/tau_imaging/tauid_ntuples/v6/output_selected.h5
```
### npy images
```
/afs/cern.ch/user/q/qbuat/work/public/tau_imaging/tauid_ntuples/v6/images_new_*.npy
```


## Processing/training/testing
NB: to be written
