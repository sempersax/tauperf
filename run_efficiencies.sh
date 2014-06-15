#!/bin/bash

VERSION=${1}
EVENTS=${2}
INPUT_LIST=./input_lists
OUTPUT_DIR=./efficiencies

run_one_efficiency()
{
    input_file=${1}
    sample_type=${2}
    output_label=`echo ${input_file}| awk -F".txt" '{print $1}' | awk -F"input" '{print $2}'`
    output_file=efficiencies_noroc${output_label}.root
    echo ${output_file}
    ./eff-calculator_new ${INPUT_LIST}/v${VERSION}/${input_file} ${OUTPUT_DIR}/${output_file}
}

run_one_presel_eff()
{
    input_file=${1}
    sample_type=${2}
    output_label=`echo ${input_file}| awk -F".txt" '{print $1}' | awk -F"input" '{print $2}'`
    output_file=efficiencies_presel${output_label}.root
    ./eff-calculator_new ${INPUT_LIST}/v${VERSION}/${input_file} ${OUTPUT_DIR}/${output_file}

}

run_one_presel_eff input_Ztautau_14TeV_all_v${VERSION}.txt  
run_one_presel_eff input_Ztautau_14TeV_mu20_v${VERSION}.txt 
run_one_presel_eff input_Ztautau_14TeV_mu40_v${VERSION}.txt 
run_one_presel_eff input_Ztautau_14TeV_mu60_v${VERSION}.txt 
run_one_presel_eff input_Ztautau_14TeV_mu80_v${VERSION}.txt 
run_one_presel_eff input_JF17_14TeV_mu40_v${VERSION}.txt    
run_one_presel_eff input_JF17_14TeV_mu60_v${VERSION}.txt    
# run_one_presel_eff input_JF17_14TeV_all_v${VERSION}.txt 'background_14TeV' ${EVENTS}
