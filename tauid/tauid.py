# root/rootpy imports
from rootpy.tree import Tree
from rootpy import asrootpy
# local imports
from skim.mixins import TauCategories
from .decision import DecisionTool
from . import VARIABLES
from . import log; log=log[__name__]

def get_IDtools():
    ID_Tools = {}
    ID_Tools['presel_3'] = TauIDTool({"all":{'name':'BDT',
                                             'weight_file':'weights_prod/presel_fullvarlist_michel3_all_14TeV_offline_BDT_AlekseyParams.weights.xml',
                                             'variables_list': VARIABLES['presel_3'],
                                             'training': 'training_old',
                                             'cutval': 0.389722714377}})
    ID_Tools['presel_q'] = TauIDTool({"all":{'name':'BDT',
                                             'weight_file':'weights_prod/presel_fullvarlist_quentin_all_14TeV_offline.weights.xml',
                                             'variables_list': VARIABLES['presel_q'],
                                             'training': 'training',
                                                   'cutval': 0.463663626155}})
    ID_Tools['full'] = TauIDTool({'1p': {'name': 'BDT',
                                         'variables_list': VARIABLES['full_1p'],
                                         'training': 'training',
                                         'cutval': 0.499492869572,
                                         'weight_file': 'weights_prod/test_1p_14TeV_offline_full_BDT.weights.xml'},
                                  'mp': {'name': 'BDT',
                                         'variables_list': VARIABLES['full_mp'],
                                         'cutval': 0.5,
                                         'training': 'training',
                                         'weight_file': 'weights_prod/test_mp_14TeV_offline_full_BDT.weights.xml'}})
    ID_Tools['and'] = TauIDTool({'1p': {'name': 'BDT',
                                        'variables_list': VARIABLES['and_1p'],
                                        'cutval': 0.548691,
                                        'training': 'training_and',
                                        'weight_file': 'weights_prod/andrew_bdt_11/sp.xml'},
                                 'mp': {'name': 'BDT',
                                        'variables_list': VARIABLES['and_mp'],
                                        'cutval': 0.637151,
                                        'training': 'training_and',
                                        'weight_file': 'weights_prod/andrew_bdt_11/mp.xml'}})
    return ID_Tools

class TauIDTool(object):
    """
    TODO: add description
    """
    def __init__(self, DT_inputs_list):
        """ A class to handle the Tau ID decision based on several BDT"""
        self._DT = {}
        self._score = -9999
        for key, DT_inputs in DT_inputs_list.items():
            self._DT[key] = DecisionTool(DT_inputs['name'], DT_inputs['weight_file'],
                                         DT_inputs['variables_list'], DT_inputs['cutval'],
                                         training = DT_inputs['training'])

    def ToolKey(self, tau):
        bdt_cat = set(tau.idcat) & set(self._DT.keys())
        if len(bdt_cat)!=1:
            raise RuntimeError('Need exactly one category in common')
        return list(bdt_cat)[0]

    def Decision(self, tau):
        tool_to_use = self.ToolKey(tau)
        log.debug('Tool to use: {0}'.format(tool_to_use))
        Decision = self._DT[tool_to_use].Decision(tau)
        self._score = self._DT[tool_to_use].score
        return Decision
    
    @property
    def score(self):
        return self._score

    # ----------------------------------------------------------------
    def SetCutValues(self, cutvalues):
        for input_key in self._DT:
            if input_key in cutvalues:
                self._DT[input_key].cutval = cutvalues[input_key]
            else:
                raise RuntimeError('The input key does not contain a cut value !')
