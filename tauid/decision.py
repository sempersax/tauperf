# --> python imports
from array import array
# --> ROOT/rootpy imports
from ROOT import TMVA
from rootpy.extern import ordereddict
# local imports
from . import log; log = log[__name__]

class DecisionTool:
    """
    TODO: add description
    """
    def __init__(self,
                 tree,
                 name,
                 weight_file,
                 variables_list,
                 cutval,
                 training_name = 'training'):
        """ A class to handle the decision of the BDT"""
        TMVA.Tools.Instance()
        self._reader = TMVA.Reader()
        self._tree   = tree
        self._variables_list = variables_list
        self._variables = ordereddict.OrderedDict()
        for var in self._variables_list:
            self._variables[var['name']]=[var['training'], var[training_name], array('f', [0.])]
        self._cutvalue = -1
        self._bdtscore = -9999
        self._name = name
        self._training_name = training_name
        log.info('SetReader({0}, {1}, {2})'.format(name, weight_file, variables_list))
        self.SetReader(name, weight_file)
        self.SetCutValue(cutval)
        
    # --------------------------------------------
    def SetReader(self, name, weight_file):
        log.info('Set the {0} with {1}'.format(name, weight_file))
        for _, var in self._variables.items():
            log.info(var)
            self._reader.AddVariable(var[1], var[2])
        self._reader.BookMVA(name, weight_file)

    # --------------------------
    def SetCutValue(self, val):
        self._cutvalue = val

    # -------------------------------------------------
    def BDTScore(self):
        for _, var in self._variables.items():
            var[2] = getattr(self._tree,var[0])
        return self._reader.EvaluateMVA(self._name)

    # --------------------------------------------
    def Decision(self):
        self._bdtscore = self.BDTScore()
        if self._bdtscore>=self._cutvalue:
            return True
        else:
            return False
    
    # ----------------------
    def GetBDTScore(self):
        self._bdtscore = self.BDTScore()
        return self._bdtscore
