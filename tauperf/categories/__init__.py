from .offline import *
from .hlt import *

CATEGORIES = {
    'offline': [
        Category_Preselection,
        Category_1P,
        Category_2P,
        Category_3P,
        Category_MP,
        ],
    'hlt': [
        Category_Preselection,
        Category_1P_HLT,
        Category_2P_HLT,
        Category_3P_HLT,
        Category_MP_HLT,
        Category_HLT,
        ],
    'plotting': [
        Category_Preselection,
        Category_1P,
        Category_MP,
        ],
    'plotting_hlt': [
        Category_1P_HLT,
        Category_MP_HLT,
        # Category_HLT,
        ],
}
