from rootpy.tree import Cut

from .offline import Category_Preselection
from .features import *
from .centfrac import CentFrac_Cut
# All basic cut definitions are here

# OFFLINE_L1_MATCHED = Cut('l1_matched_to_offline != -1')
OFFLINE_HLT_MATCHED = Cut('hlt_matched_to_offline != -1')
# L1_TAUCLUS = Cut('l1_tauclus>=12000')

ONEPRONG = Cut('hlt_ntracks == 1')
TWOPRONG = Cut('hlt_ntracks == 2')
THREEPRONG = Cut('hlt_ntracks == 3')
MULTIPRONG = Cut('hlt_ntracks > 1') & Cut('hlt_ntracks < 4')

# preselection cuts
HLT_PT_CUT = Cut('hlt_pt > 25000')
HLT_PRESEL_PT_CUT = Cut('hlt_presel_pt > 25000.')
HLT_PRESEL_CENTFRAC = Cut('hlt_centFrac > CentFrac_Cut(hlt_presel_pt)')
HLT_PRESEL_CALO = HLT_PRESEL_PT_CUT & HLT_PRESEL_CENTFRAC

FAST_TRACK_CORE = Cut('0 < hlt_fasttrack_Ncore < 4') 
FAST_TRACK_ISO = Cut('hlt_fasttrack_Niso4 < 2')
FAST_TRACK = FAST_TRACK_CORE & FAST_TRACK_ISO

# HLT_PRESEL = HLT_PRESEL_CALO & FAST_TRACK
HLT_PRESEL = HLT_PT_CUT #& Cut('hlt_pt < 150000.')


FEATURES_CUTS_ONEPRONG = (
    Cut('hlt_pt < 150000.')
    & Cut('hlt_centFrac > -1110')
    & Cut('hlt_innerTrkAvgDist > -1110')
    & Cut('hlt_ipSigLeadTrk > -999')
    & Cut('hlt_etOverPtLeadTrk > 1./3.')
    & Cut("hlt_ChPiEMEOverCaloEME > -10.")
    & Cut("hlt_ChPiEMEOverCaloEME < 10.")
    & Cut("hlt_EMPOverTrkSysP >= 0.")
    & Cut("hlt_ptRatioEflowApprox < 10.")
    & Cut("hlt_mEflowApprox < 10000.")
)

FEATURES_CUTS_ONEPRONG_PU = (
    Cut('hlt_pt < 150000.')
    & Cut('hlt_centFracCorrected > -1110')
    & Cut('hlt_innerTrkAvgDistCorrected > -1110')
    # & Cut('hlt_ipSigLeadTrkCorrected < 999')
    & Cut('hlt_etOverPtLeadTrkCorrected > 1./3.')
    & Cut("hlt_ChPiEMEOverCaloEMECorrected > -10.")
    & Cut("hlt_ChPiEMEOverCaloEMECorrected < 10.")
    & Cut("hlt_EMPOverTrkSysPCorrected >= 0.")
    & Cut("hlt_ptRatioEflowApproxCorrected < 10.")
    & Cut("hlt_mEflowApproxCorrected < 10000.")
    # & Cut("10 < averageintpercrossing< 20")
)


FEATURES_CUTS_THREEPRONG = (
    Cut('hlt_pt < 150000.')
    & Cut('hlt_centFrac > -1110')
    & Cut('hlt_innerTrkAvgDist > -1110')
    & Cut('hlt_dRmax > -1110')
    & Cut('hlt_trFlightPathSig > -1110')
    & Cut('hlt_massTrkSys >= 0')
    & Cut('hlt_etOverPtLeadTrk > 1./3.')
    & Cut("hlt_ChPiEMEOverCaloEME > -10.")
    & Cut("hlt_ChPiEMEOverCaloEME < 10.")
    & Cut("hlt_EMPOverTrkSysP >= 0.")
    & Cut("hlt_ptRatioEflowApprox < 10.")
    & Cut("hlt_mEflowApprox < 10000.")
)

FEATURES_CUTS_THREEPRONG_PU = (
    Cut('hlt_pt < 150000.')
    & Cut('hlt_centFracCorrected > -1110')
    & Cut('hlt_innerTrkAvgDistCorrected > -1110')
    & Cut('hlt_dRmaxCorrected > -1110')
    & Cut('hlt_trFlightPathSigCorrected > -1110')
    & Cut('hlt_massTrkSysCorrected >= 0')
    & Cut('hlt_etOverPtLeadTrkCorrected > 1./3.')
    & Cut("hlt_ChPiEMEOverCaloEMECorrected > -10.")
    & Cut("hlt_ChPiEMEOverCaloEMECorrected < 10.")
    & Cut("hlt_EMPOverTrkSysPCorrected >= 0.")
    & Cut("hlt_ptRatioEflowApproxCorrected < 10.")
    & Cut("hlt_mEflowApproxCorrected < 10000.")
)


class Category_1P_HLT(Category_Preselection):
    name = '1prong_hlt'
    label = '#tau_{had} (1P HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = ONEPRONG & HLT_PRESEL
    features = features_1p
    cuts_features = FEATURES_CUTS_ONEPRONG
    features_pileup_corrected = features_1p_pileup_corrected
    cuts_features_pileup_corrected = FEATURES_CUTS_ONEPRONG_PU
    eff_target = {
        'loose': 0.99,
        'medium': 0.97,
        'tight': 0.9}

class Category_2P_HLT(Category_Preselection):
    name = '2prongs_hlt'
    label = '#tau_{had} (2P HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = TWOPRONG & HLT_PRESEL

class Category_HLT(Category_Preselection):
    name = 'HLT taus'
    label = '#tau_{had} (HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = HLT_PRESEL

class Category_3P_HLT(Category_Preselection):
    name = '3prongs_hlt'
    label = '#tau_{had} (3P HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = THREEPRONG & HLT_PRESEL
    features_cut = FEATURES_CUTS_THREEPRONG
    features = features_mp
    cuts_features = FEATURES_CUTS_THREEPRONG
    features_pileup_corrected = features_mp_pileup_corrected
    cuts_features_pileup_corrected = FEATURES_CUTS_THREEPRONG_PU

class Category_MP_HLT(Category_Preselection):
    name = 'multiprongs_hlt'
    label = '#tau_{had} (MP HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = MULTIPRONG & HLT_PRESEL
    eff_target = {
        'loose': 0.85,
        'medium': 0.70,
        'tight': 0.55}


