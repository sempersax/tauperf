from EFTau_Category import Category
from DecisionTool import DecisionTool


class TauIDTool:

    # ----------------------------------------------------------------
    def __init__(self,tree,DT_inputs_list):
        """ A class to handle the Tau ID decision based on several BDT"""
        self._tree = tree
        self._DT_inputs_list = DT_inputs_list
        self._DT = {}
        for input_key in self._DT_inputs_list:
            self._DT[input_key] = DecisionTool( self._tree,
                                                self._DT_inputs_list[input_key][0],
                                                self._DT_inputs_list[input_key][1],
                                                self._DT_inputs_list[input_key][2],
                                                self._DT_inputs_list[input_key][3] )
    
    # ----------------------------------------------------------------
    def ToolKey(self):
        categories = []
        tau_cat = Category(self._tree)
        bdt_cat = set( tau_cat.ID_cat ) & set( self._DT.keys() )
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
