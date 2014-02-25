#!/bin/bash

TAR_BALL=analysis.tar.gz
HELPER_DIR=../helpers
SKIM_DIR=../skim_tools
SUBSTRUCTURE_DIR=../substructure_tools

PY_SCRIPT_1=../D3PD_slimmer
PY_SCRIPT_2=../TrainingTesting_Sample

echo "-- Tarball the analysis --"
tar cvzf ${TAR_BALL} ${HELPER_DIR} ${SKIM_DIR} ${SUBSTRUCTURE_DIR} ${PY_SCRIPT_1} ${PY_SCRIPT_2}
echo "-- Tarball rootpy --"
tar czf rootpy.tar.gz ../rootpy
echo "-- done ! --"
