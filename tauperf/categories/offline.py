from rootpy.tree import Cut

from .base import Category
from .features import *
# All basic cut definitions are here

L1_TAUCLUS = Cut('l1_tauclus>=12000')
L1_ISOL = Cut('(l1_emisol < 1000 * (2.0 + 0.10 * l1_tauclus / 1000.) && l1_tauclus <= 60000.) || l1_tauclus > 60000.')
OFFLINE_L1_MATCHED = Cut('l1_matched_to_offline != -1')
OFFLINE_HLT_MATCHED = Cut('hlt_matched_to_offline != -1')

ONEPRONG = Cut('off_ntracks == 1')
TWOPRONG = Cut('off_ntracks == 2')
THREEPRONG = Cut('off_ntracks == 3')
MULTIPRONG = Cut('off_ntracks > 1')
OFF_ETA = Cut('abs(off_eta) < 2.5')

# common preselection cuts
PRESELECTION = (
    OFF_ETA
    # OFFLINE_L1_MATCHED
    # & OFFLINE_HLT_MATCHED
    # & L1_TAUCLUS & L1_ISOL
)

FEATURES_CUTS_ONEPRONG = (
    Cut('off_centFrac > -1110')
    & Cut('off_innerTrkAvgDist > -1110')
    & Cut('off_ipSigLeadTrk > -999')
    & Cut('off_etOverPtLeadTrk > 1./3.')
    & Cut("off_ChPiEMEOverCaloEME > -10.")
    & Cut("off_ChPiEMEOverCaloEME < 10.")
    & Cut("off_EMPOverTrkSysP >= 0.")
    & Cut("off_ptRatioEflowApprox < 10.")
    & Cut("off_mEflowApprox < 10000.")
)

FEATURES_CUTS_THREEPRONG = (
    Cut('off_centFrac > -1110')
    & Cut('off_innerTrkAvgDist > -1110')
    & Cut('off_dRmax > -1110')
    & Cut('off_trFlightPathSig > -1110')
    & Cut('off_massTrkSys >= 0')
    & Cut('off_etOverPtLeadTrk > 1./3.')
    & Cut("off_ChPiEMEOverCaloEME > -10.")
    & Cut("off_ChPiEMEOverCaloEME < 10.")
    & Cut("off_EMPOverTrkSysP >= 0.")
    & Cut("off_ptRatioEflowApprox < 10.")
    & Cut("off_mEflowApprox < 10000.")
)




class Category_NoCut(Category):
    name = 'NoCut'
    label = '#tau_{had}'
    cuts = Cut('off_pt > 20000. && off_pt < 80000.')

class Category_Preselection(Category):
    name = 'inclusive'
    label = '#tau_{had}'
    common_cuts = PRESELECTION # & Cut('off_pt > 30000.')


class Category_1P(Category_Preselection):
    name = '1prong'
    label = '#tau_{had} (1P)'
    common_cuts = Category_Preselection.common_cuts
    cuts = ONEPRONG
    features_cuts = FEATURES_CUTS_ONEPRONG
    features_pileup_corrected = features_1p_pileup_corrected

class Category_2P(Category_Preselection):
    name = '2prongs'
    label = '#tau_{had} (2P)'
    common_cuts = Category_Preselection.common_cuts
    cuts = TWOPRONG
    features_pileup_corrected = features_mp_pileup_corrected

class Category_3P(Category_Preselection):
    name = '3prongs'
    label = '#tau_{had} (3P)'
    common_cuts = Category_Preselection.common_cuts
    cuts = THREEPRONG
    features_cuts = FEATURES_CUTS_THREEPRONG
    features_pileup_corrected = features_mp_pileup_corrected

class Category_MP(Category_Preselection):
    name = 'multiprongs'
    label = '#tau_{had} (MP)'
    common_cuts = Category_Preselection.common_cuts
    cuts = MULTIPRONG
    features_pileup_corrected = features_mp_pileup_corrected
