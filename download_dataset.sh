#!/bin/bash

VERSION=${1}
ROOTFILES_DIR=rootfiles
##-----------------------------------------------
untar_dataset(){
    final_dir=${1}
    for i in `ls --color=never | grep root.tgz`
      do 
      echo $i
      tar xvzf $i;rm $i
      mv *.root ../${final_dir}
    done
}

##-----------------------------------------------
get_dataset(){
    input_dataset=${1}
    final_dir=${input_dataset}_merged
    dq2-get -f "*.root.tgz" ${1}/
    mkdir -p ${final_dir}
    for dir in `ls | grep user.qbuat | grep -v merged`
      do  
      echo $dir; cd $dir
      untar_dataset ${final_dir} 
      cd ..; rm -r $dir
    done
}

cd ${PWD}/${ROOTFILES_DIR}/
mkdir -p v${VERSION}; cd v${VERSION};


get_dataset user.qbuat.mc12_14TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.ESD.e1313_s1682_s1691_r4710_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.mc12_14TeV.129160.Pythia8_AU2CTEQ6L1_perf_JF17.recon.ESD.e1313_s1682_s1691_r4711_D3PD_v2_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.mc12_14TeV.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1836_s1715_s1691_r4738_D3PD_v3_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.mc12_14TeV.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1836_s1715_s1691_r4739_D3PD_v3_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.mc12_14TeV.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1836_s1715_s1691_r4740_D3PD_v2_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.mc12_14TeV.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1836_s1715_s1691_r4741_D3PD_v2_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.user.mhodgkin.TauPi0Rec_D3PD.147818.Pythia8_AU2CTEQ6L1_Ztautau.recon.ESD.e1176_s1479_s1470_r3553.v06-00_TauPerfSkim_v${VERSION} 

get_dataset user.qbuat.data12_8TeV.periodA.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodB.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodC.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodD.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodE.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodG.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodH.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodI.physics_JetTauEtmiss.PhysCont.DESD_CALJET.t0pro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodJ.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodL.physics_JetTauEtmiss.PhysCont.DESD_CALJET.repro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
get_dataset user.qbuat.data12_8TeV.periodM.physics_JetTauEtmiss.PhysCont.DESD_CALJET.t0pro14_v01_D3PD_v1_TauPerfSkim_v${VERSION}
