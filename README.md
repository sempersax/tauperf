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

## Installing the dependencies
### setuptools (if needed)
```bash
wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
python ez_setup.py --user
```
### rootpy
```bash
git clone https://github.com/rootpy/rootpy.git
cd rootpy
python setup.py install --user
cd .. 
 ```
### prettytable 
 ```bash
 git clone https://github.com/qbuat/prettytable.git
 cd prettytable/
 python setup.py install --user
 cd ..
 ```
## Downloading and setup of the tauperf project
The set of commands below allows you to download and use the tauperf package. If you want to contribute to the project and also make your work available publicly you need to fork it on github and work with your own copy of tauperf.

```bash
git clone https://github.com/qbuat/tauperf.git
cd tauperf
source setup.sh
```
# Usage
## Initial setup at each login
1. Source the initial setup [script](#setup)
1. Go to the `tauperf` directory
1. Source the `setup.sh` script in tauperf

## Samples organisation
for each dataset, you need to merge all the files into a single rootfile using the `hadd` functionality from root.
The code uses the following naming scheme

| Sample |  rootfile name             |
|--------|:--------------------------:|
| data   | data.root                  |
| Ztautau| Ztautau.root               |
| JZ slices | jz1w.root, .., jz7w.root|
 
 Once you have organized the samples properly (ie created the merged rootfile for all of them), you need to put them in a specific directory 
 and specify this directory in the variables *NTUPLE_PATH* in `tauperf/__init__.py`
 
## Applying weights
In the analysis, several weights can be computed.
 
| Weight     | Sample    | Comment                           |
|------------|:---------:|:----------------------------------|
| pileup     | Z, JZ     | already calculated in the ntuples |
| pt         | data, JZ  | reweight bkg pt distribution to signal        |
|anti-pileup | data      | optional (just for testing)       |
| bdt score  | all       | after training, apply bdt score      |

To apply those weights, it is more efficient to run on the individual files (before hadd) rather than the merged files. The code will 
use all of the cpus of the machine you are running on to apply weights to several files in parallel. Of course if you run on a single core machine
this will not make any difference ;-).

The scripts `apply-pt-weight` or `apply-bdt-weights-from-xml` can be used in the following way
```bash
# Option 1
apply-pt-weight file1.root,file2.root
# Option 2
apply-pt-weight *.root
```
Each processed file will be copied with a prefix (`weighted` or `scored`) and the weight will be attached.

## Computing pt weights
### Command
```bash
pt-weight --categories plotting_hlt --level hlt
```
### Output
```bash
INFO:rootpy.plotting.style] using ROOT style 'ATLAS(shape=rect)'
INFO:tauperf.analysis] Use Z->tautau simulation
INFO:tauperf.samples.sample] tau: weights are None
INFO:tauperf.analysis] Use data for bkg
INFO:tauperf.samples.sample] jet: weights are None
INFO:tauperf.analysis] Analysis object is instantiated
INFO:tauperf.analysis]
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] 1prong_hlt category
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] Signal cuts: ((hlt_ntracks==1)&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5))
INFO:tauperf.analysis] Background cuts: (((hlt_ntracks==1)&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(met<100000.)
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:pt-weight] RatioPlot('RatioPlot_JLQSK3hnDVXjbFXTKKZo6i')
INFO:ROOT.TCanvas.Print] png file plots/pt_weight_1prong_hlt.png has been created
INFO:tauperf.analysis]
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] multiprongs_hlt category
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] Signal cuts: (((hlt_ntracks>1)&&(hlt_ntracks<4))&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5))
INFO:tauperf.analysis] Background cuts: ((((hlt_ntracks>1)&&(hlt_ntracks<4))&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(met<100000.)
INFO:pt-weight] RatioPlot('RatioPlot_mjfRWU5SAswsYBCp2pFS5J')
INFO:ROOT.TCanvas.Print] png file plots/pt_weight_multiprongs_hlt.png has been created
```
In addition to the two plots, a file `pt_weights.root` has been created. If you want to use this file to recompute the pt weights, you need to copy/move it over to the `cache` directory.

 
 
