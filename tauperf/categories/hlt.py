from rootpy.tree import Cut

from .offline import Category_Preselection
from .features import *
# All basic cut definitions are here


# OFFLINE_L1_MATCHED = Cut('l1_matched_to_offline != -1')
# OFFLINE_HLT_MATCHED = Cut('hlt_matched_to_offline != -1')
# L1_TAUCLUS = Cut('l1_tauclus>=12000')

ONEPRONG = Cut('hlt_ntracks == 1')
TWOPRONG = Cut('hlt_ntracks == 2')
THREEPRONG = Cut('hlt_ntracks == 3')
MULTIPRONG = Cut('hlt_ntracks > 1')

HLT_PT_CUT = Cut('hlt_pt > 25000')


class Category_1P_HLT(Category_Preselection):
    name = '1prong_hlt'
    label = '#tau_{had} (1P HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = ONEPRONG & HLT_PT_CUT
    features = features_1p

class Category_2P_HLT(Category_Preselection):
    name = '2prongs_hlt'
    label = '#tau_{had} (2P HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = TWOPRONG & HLT_PT_CUT

class Category_3P_HLT(Category_Preselection):
    name = '3prongs_hlt'
    label = '#tau_{had} (3P HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = THREEPRONG & HLT_PT_CUT

class Category_MP_HLT(Category_Preselection):
    name = 'multiprongs_hlt'
    label = '#tau_{had} (MP HLT)'
    common_cuts = Category_Preselection.common_cuts
    cuts = MULTIPRONG & HLT_PT_CUT
    features = features_mp
