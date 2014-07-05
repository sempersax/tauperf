#!/bin/bash
# Author: Quentin Buat
# Inspired by https://github.com/htautau/hhntup/blob/master/grid-setup.sh
shopt -s expand_aliases

function print_help() {
    echo "Usage : $0 [clean|local|unpack|build|worker]"
    exit
}

if [[ $# -eq 0 ]]
then
    print_help
fi

BASE=${PWD}
github_deps=packages.github
repo='http://linuxsoft.cern.ch/cern/slc5X/updates/x86_64/RPMS/'

PYTHON_VERS_MAJOR=2.7
PYTHON_VERS=2.7.2
ROOT_VERS=5.34.14
SETUPTOOLS_VERS=2.2

ROOT_VERSION_CVMFS=5.34.14-x86_64-slc5-gcc4.3
PYTHON_VERSION_CVMFS=2.6.5-x86_64-slc5-gcc43

USE_CVMFS=true

# debugging grid issues
echo "Python site imported from:"
python -c "import site; print site.__file__"
# clear PYTHONPATH
unset PYTHONPATH


function download_from_github() {
    cd ${BASE}
    GIT_USER=${1}
    PACKAGE=${2}
    TAG=${3}
    if [[ ! -e ${PACKAGE}.tar.gz ]]
    then
        if ! wget --no-check-certificate -O ${PACKAGE}.tar.gz https://github.com/${GIT_USER}/${PACKAGE}/tarball/${TAG}
        then
            echo "Failed to download package ${PACKAGE} from github"
            exit 1
        fi
    fi
}

function unpack_github_tarball() {
    cd ${BASE}
    GIT_USER=${1}
    PACKAGE=${2}
    if [[ ! -e ${PACKAGE} ]]
    then
        if tar -pzxf ${PACKAGE}.tar.gz
        then
            rm -f ${PACKAGE}.tar.gz
            mv ${GIT_USER}-${PACKAGE}-* ${PACKAGE}
        else
            exit 1
        fi
    fi
}

function install_python_package() {
    cd ${BASE}
    PACKAGE=${1}
    if [[ -d ${PACKAGE} ]]
    then
        echo "Installing ${PACKAGE}..."
        cd ${PACKAGE}
        cp ${BASE}/setuptools-${SETUPTOOLS_VERS}.tar.gz .
        if ! python setup.py install --user
        then
            echo "Failed to install package ${PACKAGE}"
            exit 1
        fi
        cd ..
    fi
}

function setup_root() {
    source ${BASE}/root/bin/thisroot.sh
}

function setup_python() {
    export PYTHONPATH=${BASE}/python/lib/python${PYTHON_VERS_MAJOR}/site-packages${PYTHONPATH:+:$PYTHONPATH}
    export LD_LIBRARY_PATH=${BASE}/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
}

function determine_python() {
    export PYTHON_VERSION=`python -c "import distutils.sysconfig; print distutils.sysconfig.get_python_version()"`
    export PYTHON_LIB=`python -c "import distutils.sysconfig; import os; print os.path.dirname(distutils.sysconfig.get_python_lib(standard_lib=True))"`
    export PYTHON_INCLUDE=`python -c "import distutils.sysconfig; import os; print os.path.dirname(distutils.sysconfig.get_python_inc())"`/python${PYTHON_VERSION}
    echo "Python version is "${PYTHON_VERSION}
    echo "Python lib is located in "${PYTHON_LIB}
    echo "Python include path is "${PYTHON_INCLUDE}
    export PYTHONUSERBASE=${BASE}/user-python
    export PATH=${PYTHONUSERBASE}/bin${PATH:+:$PATH}
    export LD_LIBRARY_PATH=$PYTHON_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LDFLAGS="-L${PYTHON_LIB} ${LDFLAGS}"
    export CPPFLAGS="-I${PYTHON_INCLUDE} ${CPPFLAGS}"
}

function setup_CVMFS() {
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
    source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
}

function setup_ROOT_CVMFS() {
    source ${ATLAS_LOCAL_ROOT_BASE}/packageSetups/atlasLocalROOTSetup.sh \
        --skipConfirm --rootVersion=${ROOT_VERSION_CVMFS}
}

function setup_python_CVMFS() {
    source ${ATLAS_LOCAL_ROOT_BASE}/packageSetups/atlasLocalPythonSetup.sh \
        --pythonVersion=${PYTHON_VERSION_CVMFS}
}

case "${1}" in
clean)

    echo "Cleaning up..."
    rm -rf user-python 
    if [[ -f ${github_deps} ]]
    then
        while read line
        do
            line=($line)
            package=${line[1]}
            rm -rf ${package}
            rm -f ${package}.tar.gz
        done < ${github_deps}
    fi
    rm -rf ${BASE}/setuptools-${SETUPTOOLS_VERS}
    rm -f ${BASE}/setuptools-${SETUPTOOLS_VERS}.tar.gz
    ;;

local)
    
    if [[ -f ${github_deps} ]]
    then
	while read line
        do
            download_from_github $line
        done < ${github_deps}
    fi
    # get setuptools
    if [[ ! -e setuptools-${SETUPTOOLS_VERS}.tar.gz ]]
    then
        wget --no-check-certificate https://pypi.python.org/packages/source/s/setuptools/setuptools-${SETUPTOOLS_VERS}.tar.gz
    fi
    ;;

unpack)
    
    if [[ -f ${github_deps} ]]
    then
        while read line
        do 
            unpack_github_tarball $line
        done < ${github_deps}
    fi
    ;;

build)
    
    if $USE_CVMFS
    then
        setup_CVMFS
        setup_python_CVMFS
        setup_ROOT_CVMFS
    fi
    
    determine_python
    export ROOTPY_NO_EXT=1
 
    if [[ ! -e user-python ]]
    then
        echo "Creating user python area"
        mkdir -p user-python/lib/python${PYTHON_VERSION}/site-packages/
        mkdir user-python/bin
    fi

    # install setuptools
    tar -zxvf ${BASE}/setuptools-${SETUPTOOLS_VERS}.tar.gz
    cd setuptools-${SETUPTOOLS_VERS}
    python setup.py install --user
    cd -

    if [[ -f ${github_deps} ]]
    then
        while read line
        do 
            unpack_github_tarball $line
            line=($line)
            install_python_package ${line[1]}
        done < ${github_deps}
    fi
    ;;

worker)
    
    if $USE_CVMFS
    then
        setup_CVMFS
        setup_python_CVMFS
        setup_ROOT_CVMFS
    fi

    if [[ -e python ]]
    then
        setup_python
    fi
    if [[ -e root ]]
    then
        setup_root
    fi
    
    determine_python 
    export ROOTPY_GRIDMODE=true

    # source user setup script
    if [[ -f grid.setup ]]
    then
        source grid.setup
    fi
    ;;

*)
    
    print_help
    ;;
esac
