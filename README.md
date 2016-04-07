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

## Make plots of the discriminating variables
### Command
```bash
# Make plots of the discri variables in the plot directory
plot-features --trigger --categories plotting_hlt --level hlt
plot-features --trigger --categories plotting_hlt --level hlt --logy
```

### Output
```bash
INFO:rootpy.plotting.style] using ROOT style 'ATLAS(shape=rect)'
INFO:tauperf.analysis] Use Z->tautau simulation
INFO:tauperf.samples.sample] tau: weights are pu_weight
INFO:tauperf.analysis] Use data for bkg
INFO:tauperf.samples.sample] jet: weights are pt_weight
INFO:tauperf.analysis] Analysis object is instantiated
INFO:tauperf.analysis]
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] 1prong_hlt category
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] Signal cuts: (((hlt_ntracks==1)&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1)
INFO:tauperf.analysis] Background cuts: ((((hlt_ntracks==1)&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1))&&(met<100000.)
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:plot-features] ChPiEMEOverCaloEMECorrected
INFO:ROOT.TH1D.Chi2TestX] There is a bin in h1 with less than 10 effective events.

...........

INFO:plot-features] RatioPlot('RatioPlot_mhpjSPYVutCvApQPfvjrqG')
INFO:ROOT.TCanvas.Print] png file plots/features/hlt_EMPOverTrkSysPCorrected_1prong_hlt.png has been created
INFO:tauperf.analysis]
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] multiprongs_hlt category
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] Signal cuts: ((((hlt_ntracks>1)&&(hlt_ntracks<4))&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1)
INFO:tauperf.analysis] Background cuts: (((((hlt_ntracks>1)&&(hlt_ntracks<4))&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1))&&(met<100000.)
INFO:plot-features] ChPiEMEOverCaloEMECorrected
INFO:ROOT.TH1D.Chi2TestX] There is a bin in h1 with less than 10 effective events.

...........................

INFO:ROOT.TCanvas.Print] png file plots/features/hlt_trFlightPathSigCorrected_multiprongs_hlt.png has been created
```
## Training
### Making training sample
For training we need to separate the data events in two trees based on the oddity of the event number. We then train 2 bdts, and then compute the BDT score for all events using cross-validation.
```bash
prepare-train-test-trees path_to_file/data.root
```
Once the training tree is prepared, you need to rename the newly created `training.data.root` into `data.root` and to put it in another directory with the Ztautau training sample.

### Training command

```bash
nohup ./train --trigger --level hlt --features features --categories training_hlt &
nohup ./train --trigger --level hlt --features features_pileup_corrected --categories training_hlt &
```

### Output

Too long to be pasted here. Sorry you will have to run it by yourself.

## ROC curve and BDT score
### Command
```bash
roc --trigger --level hlt --categories plotting_hlt
```
### Output
```bash
INFO:rootpy.plotting.style] using ROOT style 'ATLAS(shape=rect)'
INFO:tauperf.analysis] Use Z->tautau simulation
INFO:tauperf.samples.sample] tau: weights are pu_weight
INFO:tauperf.analysis] Use data for bkg
INFO:tauperf.samples.sample] jet: weights are pt_weight
INFO:tauperf.analysis] Analysis object is instantiated
INFO:tauperf.analysis]
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] 1prong_hlt category
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] Signal cuts: (((hlt_ntracks==1)&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1)
INFO:tauperf.analysis] Background cuts: ((((hlt_ntracks==1)&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1))&&(met<100000.)
INFO:tauperf.plotting.roc] create the workers
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:tauperf.plotting.roc] --> Calculate the total yields
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:ROOT.TCanvas.Print] png file plots/roc_cat_1prong_hlt_score_hlt_bdtjetscore.png has been created
INFO:ROOT.TH1F.Chi2TestX] There is a bin in h1 with less than 10 effective events.

INFO:ROOT.TH1F.Chi2TestX] There is a bin in h2 with less than 10 effective events.

INFO:ROOT.TH1F.Chi2TestX] There is a bin in h1 with less than 10 effective events.

INFO:ROOT.TH1F.Chi2TestX] There is a bin in h2 with less than 10 effective events.

INFO:ROOT.TCanvas.Print] png file plots/scores_cat_1prong_hlt_score_hlt_bdtjetscore.png has been created
INFO:tauperf.analysis]
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] multiprongs_hlt category
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] Signal cuts: ((((hlt_ntracks>1)&&(hlt_ntracks<4))&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1)
INFO:tauperf.analysis] Background cuts: (((((hlt_ntracks>1)&&(hlt_ntracks<4))&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1))&&(met<100000.)
INFO:tauperf.plotting.roc] create the workers
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/data.root ...
INFO:tauperf.plotting.roc] --> Calculate the total yields
INFO:ROOT.TCanvas.Print] png file plots/roc_cat_multiprongs_hlt_score_hlt_bdtjetscore.png has been created
INFO:ROOT.TCanvas.Print] png file plots/scores_cat_multiprongs_hlt_score_hlt_bdtjetscore.png has been created

INFO:roc] ========================================
+-----------------+--------------------+-------------------+--------------------------------+
|     Category    |        cut         | signal efficiency | background rejection (1/eff_b) |
+-----------------+--------------------+-------------------+--------------------------------+
|    1prong_hlt   | hlt_is_loose == 1  |   0.994968817167  |         1.27444764759          |
|    1prong_hlt   | hlt_is_medium == 1 |   0.966977125922  |         1.99173577424          |
|    1prong_hlt   | hlt_is_tight == 1  |   0.913640944907  |         2.90462017437          |
| multiprongs_hlt | hlt_is_loose == 1  |   0.863020318252  |         5.04476978046          |
| multiprongs_hlt | hlt_is_medium == 1 |   0.687497912695  |         13.3721745079          |
| multiprongs_hlt | hlt_is_tight == 1  |   0.556898397836  |          26.214977795          |
+-----------------+--------------------+-------------------+--------------------------------+
INFO:roc] ========================================
```

## Efficiency and rejection plots for a given working point
### Command
```bash
working-point-picker --categories plotting_hlt --trigger --level hlt
```
### Output
```bash

```
## Pt-dependent cut 
### Command
```bash
cut-value-picker --categories plotting_hlt --level hlt --trigger
```
### Output
```bash
INFO:rootpy.plotting.style] using ROOT style 'ATLAS(shape=rect)'
INFO:tauperf.analysis] Use Z->tautau simulation
INFO:tauperf.samples.sample] tau: weights are pu_weight
INFO:tauperf.analysis] Use data for bkg
INFO:tauperf.samples.sample] jet: weights are pt_weight
INFO:tauperf.analysis] Analysis object is instantiated
[(25, 26), (26, 28), (28, 30), (30, 32), (32, 34), (34, 36), (36, 38), (38, 40), (40, 42), (42, 44), (44, 46), (46, 48), (48, 50), (50, 54), (54, 58), (58, 62), (62, 66), (66, 70), (70, 78), (78, 86), (86, 94), (94, 102), (102, 110), (110, 150), (150, 200), (200, 300)]
INFO:tauperf.analysis]
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] 1prong_hlt category
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] Signal cuts: (((hlt_ntracks==1)&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1)
INFO:tauperf.analysis] Background cuts: ((((hlt_ntracks==1)&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1))&&(met<100000.)
loose
INFO:tauperf.samples.db] opening /cluster/warehouse/qbuat/tauid_ntuples/v3/new_Z_training_sample/Ztautau.root ...
medium
tight
0 	25 	0.28860 	0.36610 	0.47190
25 	26 	0.28860 	0.36610 	0.47190
26 	28 	0.28360 	0.36440 	0.46830
28 	30 	0.28610 	0.36340 	0.46690
30 	32 	0.28090 	0.36020 	0.46610
32 	34 	0.27990 	0.35640 	0.46560
34 	36 	0.27420 	0.35400 	0.46160
36 	38 	0.27430 	0.35300 	0.46560
38 	40 	0.26590 	0.35360 	0.46400
40 	42 	0.27200 	0.35550 	0.46550
42 	44 	0.27010 	0.35670 	0.46170
44 	46 	0.26980 	0.34850 	0.45810
46 	48 	0.26560 	0.35300 	0.45860
48 	50 	0.26980 	0.35120 	0.45740
50 	54 	0.25990 	0.34730 	0.45820
54 	58 	0.26530 	0.34190 	0.45880
58 	62 	0.26820 	0.35210 	0.46170
62 	66 	0.26790 	0.34590 	0.46490
66 	70 	0.25710 	0.34500 	0.46970
70 	78 	0.27330 	0.36420 	0.47070
78 	86 	0.27830 	0.37430 	0.47370
86 	94 	0.28560 	0.37650 	0.47460
94 	102 	0.29020 	0.36540 	0.48130
102 	110 	0.25120 	0.34180 	0.47700
110 	150 	0.27160 	0.37750 	0.47620
150 	200 	0.24980 	0.31960 	0.47110
200 	300 	0.22750 	0.30190 	0.46080
300 	100000.0 	0.22750 	0.30190 	0.46080
INFO:tauperf.analysis]
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] multiprongs_hlt category
INFO:tauperf.analysis] ========================================
INFO:tauperf.analysis] Signal cuts: ((((hlt_ntracks>1)&&(hlt_ntracks<4))&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1)
INFO:tauperf.analysis] Background cuts: (((((hlt_ntracks>1)&&(hlt_ntracks<4))&&(hlt_pt>25000))&&((off_pt>20000.)&&(abs(off_eta)<2.5)))&&(hlt_matched_to_offline==1))&&(met<100000.)
loose
medium
tight
0 	25 	0.53000 	0.58970 	0.63360
25 	26 	0.53000 	0.58970 	0.63360
26 	28 	0.51960 	0.59470 	0.64320
28 	30 	0.51150 	0.59160 	0.64240
30 	32 	0.51060 	0.59130 	0.64460
32 	34 	0.51130 	0.59400 	0.64540
34 	36 	0.50880 	0.59220 	0.64530
36 	38 	0.50450 	0.59300 	0.64680
38 	40 	0.50200 	0.59130 	0.64610
40 	42 	0.50140 	0.58960 	0.64510
42 	44 	0.49620 	0.58930 	0.64530
44 	46 	0.49320 	0.58760 	0.64420
46 	48 	0.48470 	0.58460 	0.64280
48 	50 	0.48710 	0.58740 	0.64660
50 	54 	0.48320 	0.58630 	0.64430
54 	58 	0.47960 	0.58520 	0.64690
58 	62 	0.48290 	0.59260 	0.65100
62 	66 	0.49270 	0.59720 	0.65720
66 	70 	0.48820 	0.60700 	0.66070
70 	78 	0.50340 	0.60490 	0.66370
78 	86 	0.50110 	0.61710 	0.67260
86 	94 	0.49620 	0.60860 	0.66790
94 	102 	0.49410 	0.61090 	0.67350
102 	110 	0.53720 	0.62800 	0.67900
110 	150 	0.48300 	0.60690 	0.66700
150 	200 	0.44750 	0.58350 	0.65060
200 	300 	0.46480 	0.58810 	0.65830
300 	100000.0 	0.46480 	0.58810 	0.65830
```


