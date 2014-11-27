import ROOT


class trainer(ROOT.TMVA.Factory):

    def __init__( self, factory_name,outputFile ):
        ROOT.TMVA.Factory.__init__( self, factory_name, outputFile, "V:!Silent:Color:DrawProgressBar" )
        # --> Default parameters
        self._nEventsMin = 10   # Min number of events in a leaf mode (fraction (%) of the training samples) 
        self._NTrees     = 200 # Number of tres
        
    def SetVariablesFromFile(self, variables_file):
        for line in open(variables_file):
            if "#" in line: continue
            words = line.strip().split(',')
            self.AddVariable( words[0],words[1],"",words[2] )

    def BookBDT(self,nEventsMin=10,NTrees=100):
        self._nEventsMin   = nEventsMin  
        self._NTrees       = NTrees  
#!H:V:PruneBeforeBoost=False:SeparationType=GiniIndex:BoostType=AdaBoost:MaxDepth=8:NNodesMax=100000:NTrees=100:PruneMethod=NoPruning:nCuts=200:UseYesNoLeaf=False:AdaBoostBeta=0.2:DoBoostMonitor
        params  = ["PruneBeforeBoost=False"]
        params += ["SeparationType=GiniIndex"]
        params += ["BoostType=AdaBoost"]
        params += ["PruneMethod=NoPruning"]
        params += ["UseYesNoLeaf=False"]
        params += ["AdaBoostBeta=0.2"]
        params += ["DoBoostMonitor"]
        params += ["MaxDepth=8"]
        params += ["nCuts=200"]
        params += ["NNodesMax=100000"]
#         params += ["nEventsMin={0}".format(self._nEventsMin )]
        params += ["NTrees={0}".format(self._NTrees)]
        print params
#         method_name = "BDT_NTrees{0}_nEventsMin{1}".format(self._NTrees,self._nEventsMin)
        method_name = "BDT_AlekseyParams".format(self._NTrees,self._nEventsMin)

        params_string = "!H:V"
        for param in params: params_string+= ":"+param
            

        self.BookMethod( ROOT.TMVA.Types.kBDT,
                         method_name,
                         params_string )

        
