# --> python imports
from array import array

# --> ROOT imports
from ROOT import TMVA

# --> rootpy imports
from rootpy.extern import ordereddict

class DecisionTool:
    """
    TODO: add description
    """
    def __init__(self,
                 tree,
                 name,
                 weight_file,
                 variables_list,
                 cutval):
        """ A class to handle the decision of the BDT"""
        TMVA.Tools.Instance()
        self._reader = TMVA.Reader()
        self._tree   = tree
        self._variables = {}
        self._cutvalue = -1
        self._bdtscore = -9999
        self._name = name
        self._var_file = variables_list
        self.SetReader(name, weight_file, variables_list)
        self.SetCutValue(cutval)
        
    # --------------------------------------------
    @property
    def SetReader(self, name, weight_file, variables_list):
        self._variables = self.InitVariables(variables_list)
        for varName, var in self._variables.iteritems():
            self._reader.AddVariable(varName,var[1])
        self._reader.BookMVA(name, weight_file)
        
    # ----------------------
    @property
    def InitVariables(self, variables_list):
        variables = ordereddict.OrderedDict()
        for var in variables_list:
            variables[var['name']] = [var['training'], array(var['type'],[0])]
        return variables

    # --------------------------
    def SetCutValue(self, val):
        self._cutvalue = val

    # -------------------------------------------------
    def BDTScore(self):
        for varName, var in self._variables.iteritems():
            var[1][0] = getattr(self._tree,var[0])
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
