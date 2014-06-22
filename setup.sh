setupATLAS
localSetupROOT 5.34.18-x86_64-slc5-gcc4.3
current_dir=${PWD}
cd ../packages ;source RootCore/scripts/setup.sh; cd ${current_dir}

export PYTHONPATH=$PYTHONPATH:./
export PYTHONPATH=$PYTHONPATH:/cluster/data12/qbuat/EventFilterTauID_dev

