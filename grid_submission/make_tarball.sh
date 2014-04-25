#!/bin/bash

TAR_BALL=analysis.tar.gz
SKIM_DIR=../skim

# PY_SCRIPT_1=../D3PD_slimmer
PY_SCRIPT_1=../skimmer
PY_SCRIPT_2=../split-skim

echo "-- Tarball the analysis --"
tar cvzf ${TAR_BALL} ${SKIM_DIR} ${PY_SCRIPT_1} ${PY_SCRIPT_2}
echo "-- Tarball rootpy --"
tar czf rootpy.tar.gz ../rootpy
echo "-- done ! --"
