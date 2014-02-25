#!/bin/bash
###These commands set up the Grid Environment for your job:
#PBS -N EF-efficiencies
#PBS -q short
#PBS -M quentin_buat@sfu.ca
#PBS -m abe
#PBS -V

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
cd /cluster/data12/qbuat/EventFilterTauID_dev
localSetupROOT 5.34.10-x86_64-slc5-gcc4.3 --skipConfirm
cd packages ;source RootCore/scripts/setup.sh; cd ..
export PYTHONPATH=$PYTHONPATH:./

./ID_Training input_lists/v10/input_Ztautau_14TeV_mu4060_v10.txt input_lists/v10/input_JF17_14TeV_mu4060_v10.txt variables_list/variables_quentin_bdt_preselection.txt blurp1 --ID presel --cat all --data_type 14TeV_offline
# ./run_efficiencies.sh 6 -1
# ./run_roc.sh 6 200000
