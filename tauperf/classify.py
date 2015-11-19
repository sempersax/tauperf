"""
Although I'm well aware of the limitations
of TMVA and the existence of the sciki-learn package,
I dediced to stick to this for now. 
"""
# ---> python imports
import os
from multiprocessing import Process

from ROOT import TMVA
from rootpy.utils.lock import lock
from rootpy.io import root_open
from rootpy.tree import Cut

from . import log; log=log[__name__]
from samples.db import get_file
from . import UNMERGED_NTUPLE_PATH
from samples import Tau, Jet, JZ
from .variables import VARIABLES

class Classifier(TMVA.Factory):

    def __init__(self, 
                 category,
                 output_name,
                 factory_name,
                 prefix='hlt',
                 tree_name='tau',
                 training_mode='dev', # 'prod'
                 features=None,
                 split_cut=None,
                 verbose=''):

        self.output = root_open(output_name, 'recreate')
        TMVA.Factory.__init__(
            self, factory_name, self.output, verbose)
        self.factory_name = factory_name
        self.category = category
        self.prefix = prefix
        self.tree_name = tree_name
        self.training_mode = training_mode
        self.features = features
        if split_cut is None:
            self.split_cut = Cut()
        else:
            self.split_cut = Cut(split_cut)
            


    def set_variables(self, category, prefix):
        if self.features is None:
            self.features = category.features
        for varName in self.features:
            var = VARIABLES[varName]
            self.AddVariable(prefix + '_' + var['name'], var['root'], '', var['type'])

    def book(self,
             ntrees=100,
             node_size=5,
             depth=8):
        
        #         params  = ["PruneBeforeBoost=False"]
        params = ["SeparationType=GiniIndex"]
        params += ["BoostType=AdaBoost"]
        params += ["PruneMethod=CostComplexity"]
        params += ["PruneStrength=60"]
        params += ["PruningValFraction=0.5"]
        params += ["nCuts=500"]
        params += ["UseYesNoLeaf=False"]
        params += ["AdaBoostBeta=0.2"]
        params += ["DoBoostMonitor"]
        params += ["MaxDepth={0}".format(depth)]
        params += ["MinNodeSize={0}".format(node_size)]
        params += ["NTrees={0}".format(ntrees)]
        #         params += ["nCuts={0}".format(nCuts)]
        #         params += ["NNodesMax={0}".format(NNodesMax)]
        #         params += ["nEventsMin={0}".format(nEventsMin)]
        log.info(params)

        method_name = "BDT_{0}".format(self.factory_name)
        params_string = "!H:V"
        for param in params:
            params_string+= ":"+param
        log.info('booking ..')
        self.BookMethod(
            TMVA.Types.kBDT,
            method_name,
            params_string)

    def train(self, ana, **kwargs):

        self.set_variables(self.category, self.prefix)
        self.sig_cut = ana.tau.cuts(self.category, feat_cuts=True) & self.split_cut
        self.bkg_cut = ana.jet.cuts(self.category, feat_cuts=True) & self.split_cut

        params = ['NormMode=EqualNumEvents']
        params += ['SplitMode=Random']
        if self.training_mode == 'prod':
            params += ['nTest_Background=1']
            params += ['nTest_Signal=1']
            params += ['!V']
        params_string = ':'.join(params)
        log.info(self.sig_cut)
        log.info(self.bkg_cut)
        self.PrepareTrainingAndTestTree(self.sig_cut, self.bkg_cut, params_string)
        
        # Signal file
        log.info('prepare signal tree')
        sig_file = get_file(ana.tau.ntuple_path, ana.tau.student) 
        self.sig_tree = sig_file[ana.tau.tree_name]
        self.AddSignalTree(self.sig_tree)
        self.SetSignalWeightExpression(ana.tau.weight_field)

        # Bkg files
        log.info('prepare background tree')
        if isinstance(ana.jet, JZ):
            for sample, scale in zip(ana.jet.components, ana.jet.scales):
                rfile = get_file(sample.ntuple_path, sample.student)
                tree = rfile[sample.tree_name]
                self.AddBackgroundTree(tree, scale)
        else:
            bkg_file = get_file(ana.jet.ntuple_path, ana.jet.student)
            self.bkg_tree = bkg_file[ana.jet.tree_name]
            self.AddBackgroundTree(self.bkg_tree)

        self.SetBackgroundWeightExpression(ana.jet.weight_field)
        log.info('preparation is done, start booking')
        # Actual training
        self.book(**kwargs)
        # booking is done, start training
        self.TrainAllMethods()
        if self.training_mode == 'dev':
            self.output.cd()
            self.TestAllMethods()
            self.EvaluateAllMethods()
            self.output.Close()
