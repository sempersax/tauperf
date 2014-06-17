from ROOT import TMVA
from array import array
from rootpy.extern import ordereddict


class DecisionTool:


    def __init__(self,tree,name,weight_file,var_file,cutval):
        """ A class to handle the decision of the BDT"""
        TMVA.Tools.Instance()
        self._reader = TMVA.Reader()
        self._tree   = tree
        self._variables = {}
        self._cutvalue = -1
        self._bdtscore = -9999
        self._name = name
        self._weight_file = weight_file
        self._var_file = var_file
        self.SetReader(self._name,self._weight_file,self._var_file)
        self.SetCutValue(cutval)
        
    # --------------------------
    def SetCutValue(self,val):
        self._cutvalue = val

    # --------------------------------------------
    def SetReader(self,name,weight_file,var_file):
        self._variables = self.InitVariables(var_file)
        for varName, var in self._variables.iteritems():
            self._reader.AddVariable(varName,var[1])
        self._reader.BookMVA(name,weight_file)
        
    # ----------------------
    def InitVariables(self,var_file):

        variables = ordereddict.OrderedDict()
        file = open(var_file,'r')
        
        for line in file:
            if "#" in line: continue
            words = line.strip().split(',')
            variables[ words[0] ] = [ words[1],array( 'f',[0.]) ]
        return variables

    # -------------------------------------------------
    def BDTScore(self):
        for varName, var in self._variables.iteritems():
            var[1][0] = getattr(self._tree,var[0])
            log.info('{0}: {1}'.format(varName, var[1][0])
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
