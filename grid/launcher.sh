#!/bin/bash

inputfilelist=${1}
stream=${2} # 'signal', 'background', 'data'

echo "Current directory: ${PWD}"
initdir=${PWD}

tarballs=(analysis)

for package in ${tarballs[*]}
do
  echo "Start unpackaging ${package}"
  tar -xvf ${package}.tar.gz
  echo "${package} unpacked !"
done

echo "Create list of input files"
listoffiles="input_files.txt"
./create-inputlist $inputfilelist $listoffiles

if [[ -f ${listoffiles} ]]
then 
    while read filename 
    do
	echo $filename
	inputfile=`echo ${filename}| awk -F"/" '{print $NF}'`
	outputfile=`echo $inputfile| awk -F".root" '{print $1"_skimmed.root"}'`
	echo "############# SKIMMING/SLIMMING #######################"
	./skim-maker ${filename} ${outputfile} $stream
	echo "############# CREATE TRAINING/TESTING TREES ###########"
	./skim-split ${outputfile}
    done < $listoffiles

cp $listoffiles $initdir/
cp *.root $initdir/

echo "End of the script"
