#!/bin/bash

VERSION=${1}

# ------------------------------
prun_command()
{
    IN=${1}
    OUT=${2}
    STREAM=${3}
    echo 'prun --exec "./launcher.sh %IN '${STREAM}'" --inDS='${IN}'/ --outDS='${OUT}'/ 
	--rootVer=5.34/10 --cmtConfig=x86_64-slc5-gcc43-opt --nGBPerJob=MAX 
	--extFile=analysis.tar.gz,packages.tar.gz,rootpy.tar.gz,setuptools-2.2.tar.gz,ez_setup.py 
	--useRootCore --noCompile
	--site=SFU-LCG2_LOCALGROUPDISK
	--outputs="*skimmed.root"'
    prun --exec "./launcher.sh %IN ${STREAM}" --inDS=${IN}/ --outDS=${OUT}/ \
	--rootVer=5.34/10 --cmtConfig=x86_64-slc5-gcc43-opt --nGBPerJob=MAX \
	--extFile=analysis.tar.gz,rootpy.tar.gz,setuptools-2.2.tar.gz,ez_setup.py,create-inputlist \
	--useRootCore --noCompile \
	--site=SFU-LCG2_LOCALGROUPDISK \
	--outputs="*skimmed.root"


### --site=SFU-LCG2_LOCALGROUPDISK
}
# ------------------------------
submit_onejob()
{
    IN=${1}
    STREAM=${2}
    VERSION=${3}
    OUT=${IN}_TauPerfSkim_v${VERSION}
    user=`echo ${OUT}| awk -F"." '{print $1}'`
    name=`echo ${OUT}| awk -F"." '{print $2}'`
    user_name=${user}.${name}
    echo ${user_name}
    if [ "${user_name}" != "user.qbuat" ]; then 
	OUT=user.qbuat.${OUT}
    fi
    prun_command ${IN} ${OUT} ${STREAM}
}


echo "-------- Signal sample jobs ---------"
submit_onejob user.qbuat.mc12_14TeV.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1836_s1715_s1691_r4738_D3PD_v3 'signal_14TeV' ${VERSION}
submit_onejob user.qbuat.mc12_14TeV.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1836_s1715_s1691_r4739_D3PD_v3 'signal_14TeV' ${VERSION} 
submit_onejob user.qbuat.mc12_14TeV.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1836_s1715_s1691_r4740_D3PD_v2 'signal_14TeV' ${VERSION} 
submit_onejob user.qbuat.mc12_14TeV.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1836_s1715_s1691_r4741_D3PD_v2 'signal_14TeV' ${VERSION} 
prun_command user.mhodgkin.TauPi0Rec_D3PD.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1176_s1479_s1470_r3553_tid00999074_00.v06-00 \
    user.qbuat.user.mhodgkin.TauPi0Rec_D3PD.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1176_s1479_s1470_r3553.v06-00_TauPerfSkim_v${VERSION} 'signal_8TeV'


echo "-------- Background sample jobs ---------"
submit_onejob user.qbuat.mc12_14TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.ESD.e1313_s1682_s1691_r4710_D3PD_v1 'background_14TeV' ${VERSION}
submit_onejob user.qbuat.mc12_14TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.ESD.e1313_s1682_s1691_r4711_D3PD_v2 'background_14TeV' ${VERSION}

# echo "-------- Data sample jobs ---------"
# submit_onejob user.qbuat.data12_8TeV.periodA.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodB.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodC.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodD.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodE.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodG.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodH.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodI.physics_JetTauEtmiss.PhysCont.DESD_CALJET.t0pro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodJ.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodL.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1 'data' ${VERSION}
# submit_onejob user.qbuat.data12_8TeV.periodM.physics_JetTauEtmiss.PhysCont.DESD_CALJET.t0pro14_v01_D3PD_v1 'data' ${VERSION}


