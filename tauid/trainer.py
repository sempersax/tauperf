from ROOT import TMVA


class trainer(TMVA.Factory):

    def __init__(self, factory_name,outputFile):
        ROOT.TMVA.Factory.__init__(self,
                                   factory_name,
                                   outputFile,
                                   "V:!Silent:Color:DrawProgressBar")
        
    def SetVariablesFromList(self, variables_list):
        for var in variables_list:
            self.AddVariable(var['name'], var['training'], '', var['type'])

    def BookBDT(self,
                nEventsMin=10,
                NTrees=100,
                MaxDepth=8,
                nCuts=200,
                NNodesMax=100000):

        params  = ["PruneBeforeBoost=False"]
        params += ["SeparationType=GiniIndex"]
        params += ["BoostType=AdaBoost"]
        params += ["PruneMethod=NoPruning"]
        params += ["UseYesNoLeaf=False"]
        params += ["AdaBoostBeta=0.2"]
        params += ["DoBoostMonitor"]
        params += ["MaxDepth={0}".format(MaxDepth)]
        params += ["nCuts=[0}".format(nCuts)]
        params += ["NNodesMax={0}".format(NNodesMax)]
        params += ["nEventsMin={0}".format(nEventsMin)]
        params += ["NTrees={0}".format(NTrees)]
        print params

        method_name = "BDT_Train"
        params_string = "!H:V"
        for param in params:
            params_string+= ":"+param
        self.BookMethod(ROOT.TMVA.Types.kBDT,
                        method_name,
                        params_string)

        
