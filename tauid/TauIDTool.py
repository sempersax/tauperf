# from EFTau_Category import Category
from skim.mixins import TauCategories
from DecisionTool import DecisionTool

from rootpy.tree import Tree
from rootpy import asrootpy

class TauIDTool:
    """
    TODO: add description
    """
    # ----------------------------------------------------------------
    def __init__(self,tree,DT_inputs_list):
        """ A class to handle the Tau ID decision based on several BDT"""
        self._tree = asrootpy(tree)
        self._DT = {}
        for key, DT_inputs in DT_inputs_list.items():
            self._DT[key] = DecisionTool(self._tree,
                                         DT_inputs_list.name,
                                         DT_inputs_list.weight_file,
                                         DT_inputs_list.variables_list,
                                         DT_inputs_list.cutval)
    
    # ----------------------------------------------------------------
    @property
    def ToolKey(self):
        categories = []
        tree.define_object(name='tau', prefix='off', mix=TauCategories)
        tau_idcat = tree.tau.idcat
        bdt_cat = set(tau_idcat) & set(self._DT.keys())
        if len(bdt_cat)!=1:
            raise RuntimeError('Need exactly one category in common')
        return list(bdt_cat)[0]
    # ----------------------------------------------------------------
    def Decision(self):
        tool_to_use = self.ToolKey()
        Decision = self._DT[tool_to_use].Decision()
        return Decision
    
    # ----------------------------------------------------------------
    def BDTScore(self):
        tool_to_use = self.ToolKey()
        BDT_Score = self._DT[tool_to_use].GetBDTScore()
        return BDT_Score

    # ----------------------------------------------------------------
    def SetCutValues(self,cutvalues):
        for input_key in self._DT:
            if input_key in cutvalues:
                self._DT[input_key].SetCutValue(cutvalues[input_key])
            else:
                raise RuntimeError('The input key does not contain a cut value !')
