# ---> python imports
import os
from multiprocessing import Process
# ---> local imports
from . import log; log=log[__name__]
# ---> ROOT imports
from ROOT import TMVA
# ---> rootpy imports
from rootpy.utils.lock import lock
from rootpy.io import root_open

class BDTScan(Process):
    """
    """
    def __init__(self,
                 output_name,
                 factory_name,
                 variables_list,
                 sig_tree,
                 bkg_tree,
                 sig_cut,
                 bkg_cut,
                 ntrees,
                 neventsmin):
        super(BDTScan, self).__init__()
        self.output_name = output_name
        self.factory_name = factory_name
        self.variables_list = variables_list
        self.sig_tree = sig_tree
        self.bkg_tree = bkg_tree
        self.sig_cut = sig_cut
        self.bkg_cut = bkg_cut
        self.ntrees = ntrees
        self.neventsmin = neventsmin

    def run(self):
        with root_open(self.output_name, 'recreate') as output_file: 
            factory = trainer(self.factory_name, output_file)
            factory.SetVariablesFromList(self.variables_list)
            factory.SetInputTrees(self.sig_tree, self.bkg_tree)
            factory.PrepareTrainingAndTestTree(self.sig_cut, self.bkg_cut,
                                               "NormMode=EqualNumEvents:SplitMode=Block:!V")
            factory.BookBDT(NTrees=self.ntrees, nEventsMin=self.neventsmin)
            factory.TrainAllMethods()
            factory.TestAllMethods()
            factory.EvaluateAllMethods()
        

class trainer(TMVA.Factory):

    def __init__(self, factory_name,outputFile, verbose=''):
        TMVA.Factory.__init__(self,
                              factory_name,
                              outputFile,
                              verbose)
                              #"V:!Silent:Color:DrawProgressBar")
        
    def SetVariablesFromList(self, variables_list):
        for var in variables_list:
            self.AddVariable(var['training'], var['root'], '', var['type'])

    def BookBDT(self,
                nEventsMin=10,
                NTrees=100,
                #MaxDepth=8,
                nCuts=200,
                NNodesMax=100000):

        params  = ["PruneBeforeBoost=False"]
        params += ["SeparationType=GiniIndex"]
        params += ["BoostType=AdaBoost"]
        params += ["PruneMethod=NoPruning"]
        params += ["UseYesNoLeaf=False"]
        params += ["AdaBoostBeta=0.2"]
        params += ["DoBoostMonitor"]
        #         params += ["MaxDepth={0}".format(MaxDepth)]
        params += ["nCuts={0}".format(nCuts)]
        params += ["NNodesMax={0}".format(NNodesMax)]
        params += ["nEventsMin={0}".format(nEventsMin)]
        params += ["NTrees={0}".format(NTrees)]
        log.info(params)

        method_name = "BDT"
        params_string = "!H:V"
        for param in params:
            params_string+= ":"+param
        self.BookMethod(TMVA.Types.kBDT,
                        method_name,
                        params_string)

        
