#!/bin/bash

echo "specific to SFU GPU cluster"
source /software/root-6.06.08/bin/thisroot.sh

export VE_PATH=/home/qbuat/imaging/imaging_ve
export DATA_AREA=/cluster/warehouse/qbuat/crackpot

source ${VE_PATH}/bin/activate


SOURCE_TAUPERF_SETUP="${BASH_SOURCE[0]:-$0}"
DIR_TAUPERF_SETUP="$( dirname "$SOURCE_TAUPERF_SETUP" )"

while [ -h "$SOURCE_TAUPERF_SETUP" ]
do 
  SOURCE_TAUPERF_SETUP="$(readlink "$SOURCE_TAUPERF_SETUP")"
  [[ $SOURCE_TAUPERF_SETUP != /* ]] && SOURCE_TAUPERF_SETUP="$DIR_TAUPERF_SETUP/$SOURCE_TAUPERF_SETUP"
  DIR_TAUPERF_SETUP="$( cd -P "$( dirname "$SOURCE_TAUPERF_SETUP"  )" && pwd )"
  echo $SOURCE_TAUPERF_SETUP
  echo $DIR_TAUPERF_SETUP
done
DIR_TAUPERF_SETUP="$( cd -P "$( dirname "$SOURCE_TAUPERF_SETUP" )" && pwd )"

echo $DIR_TAUPERF_SETUP
echo "sourcing ${SOURCE_TAUPERF_SETUP}..."

export PATH=${DIR_TAUPERF_SETUP}${PATH:+:$PATH}
export PYTHONPATH=${DIR_TAUPERF_SETUP}${PYTHONPATH:+:$PYTHONPATH}

# specific to the gpu cluster
export THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.8'
git pull origin imaging