#!/bin/bash

TAR_BALL=analysis.tar.gz
SKIM_DIR=../skim_tools
SUBSTRUCTURE_DIR=../substructure_tools

# PY_SCRIPT_1=../D3PD_slimmer
PY_SCRIPT_1=../new_skimmer
PY_SCRIPT_2=../TrainingTesting_Sample

echo "-- Tarball the analysis --"
tar cvzf ${TAR_BALL} ${SKIM_DIR} ${SUBSTRUCTURE_DIR} ${PY_SCRIPT_1} ${PY_SCRIPT_2}
echo "-- Tarball rootpy --"
tar czf rootpy.tar.gz ../rootpy
echo "-- done ! --"
