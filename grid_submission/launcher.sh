#!/bin/bash

inputfilelist=${1}
stream=${2} # 'signal', 'background', 'data'

echo "Current directory: ${PWD}"
initdir=${PWD}

tarballs=(analysis rootpy)

for package in ${tarballs[*]}
do
  echo "Start unpackaging ${package}"
  tar -xvf ${package}.tar.gz
  echo "${package} unpacked !"
done

echo "Setup setuptools"
python ez_setup.py --user

# echo "Setup rootcore packages"
# cd packages ;source RootCore/scripts/setup.sh; cd ${initdir}

echo "Set python path"
export PYTHONPATH=$PYTHONPATH:./

echo "Create list of input files"
listoffiles="input_files.txt"
python ./skim/CreateFileList.py $inputfilelist $listoffiles

while read filename; do
    echo $filename
    inputfile=`echo ${filename}| awk -F"/" '{print $NF}'`
    outputfile=`echo $inputfile| awk -F".root" '{print $1"_skimmed.root"}'`
    echo "############# SKIMMING/SLIMMING #######################"
    ./skimmer ${filename} ${outputfile} $stream
    echo "############# CREATE TRAINING/TESTING TREES ###########"
    ./split-skim ${outputfile}
done < $listoffiles

cp $listoffiles $initdir/
cp *.root $initdir/

echo "End of the script"
