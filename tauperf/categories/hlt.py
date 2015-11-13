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
HLT_PRESEL = OFFLINE_HLT_MATCHED & HLT_PT_CUT

class Category_1P_HLT(Category_Preselection):
    name = '1prong_hlt'
    label = '#tau_{had} (1P HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = ONEPRONG & HLT_PRESEL
    features = features_1p
    features_pileup_corrected = features_1p_pileup_corrected

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

class Category_MP_HLT(Category_Preselection):
    name = 'multiprongs_hlt'
    label = '#tau_{had} (MP HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = MULTIPRONG & HLT_PRESEL
    features = features_mp
    features_pileup_corrected = features_mp_pileup_corrected
