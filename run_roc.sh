#!/bin/bash

VERSION=${1}
EVENTS=${2}
INPUT_LIST=./input_lists
OUTPUT_DIR=./rocs

run_one_roc()
{
    input_file_sig=${1}
    input_file_bkg=${2}
    sample_type=${3}
    output_label=`echo ${input_file_sig}| awk -F".txt" '{print $1}' | awk -F"input" '{print $2}'`
    output_file=${OUTPUT_DIR}/roc${output_label}.root
    echo ${output_file}
    python roc_curve.py ${INPUT_LIST}/v${VERSION}/${input_file_sig} ${INPUT_LIST}/v${VERSION}/${input_file_bkg} ${output_file} ${sample_type} -N ${4}
}
B
run_one_roc input_Ztautau_14TeV_mu40_v${VERSION}.txt input_JF17_14TeV_mu40_v${VERSION}.txt '14TeV' ${EVENTS}
run_one_roc input_Ztautau_14TeV_mu60_v${VERSION}.txt input_JF17_14TeV_mu60_v${VERSION}.txt '14TeV' ${EVENTS}

