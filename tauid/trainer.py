# ---> python imports
import os
from multiprocessing import Process
# ---> local imports
from . import log; log=log[__name__]
# ---> ROOT imports
from ROOT import TMVA
# ---> rootpy imports
from rootpy.utils.lock import lock

class BDTScan(Process):
    """
    """
    def __init__(self,
                 output_name,
                 factory, ntrees, neventsmin):
        super(BDTScan, self).__init__()
        self.output_name = output_name
        self.factory = factory
        self.ntrees = ntrees
        self.neventsmin = neventsmin

    def run(self):
        with lock(self.output_name):
            self.factory.BookBDT(NTrees=self.ntrees, nEventsMin=self.neventsmin)
            self.factory.TrainAllMethods()
            self.factory.TestAllMethods()
            self.factory.EvaluateAllMethods()
        
class trainer(TMVA.Factory):

    def __init__(self, factory_name,outputFile):
        TMVA.Factory.__init__(self,
                              factory_name,
                              outputFile,
                              "V:!Silent:Color:DrawProgressBar")
        
    def SetVariablesFromList(self, variables_list):
        for var in variables_list:
            self.AddVariable(var['training'], var['root'], '', var['type'])

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
        params += ["nCuts={0}".format(nCuts)]
        params += ["NNodesMax={0}".format(NNodesMax)]
        params += ["nEventsMin={0}".format(nEventsMin)]
        params += ["NTrees={0}".format(NTrees)]
        log.info(params)

        method_name = "BDT_Train_Ntrees{0}_Nevents{1}".format(NTrees, nEventsMin)
        params_string = "!H:V"
        for param in params:
            params_string+= ":"+param
        self.BookMethod(TMVA.Types.kBDT,
                        method_name,
                        params_string)

        
