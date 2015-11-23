"""
Although I'm well aware of the limitations
of TMVA and the existence of the sciki-learn package,
I dediced to stick to this for now. 
"""
# ---> python imports
import os
from multiprocessing import Process

import ROOT
from rootpy.utils.lock import lock
from rootpy.io import root_open
from rootpy.tree import Cut

from . import log; log=log[__name__]
from samples.db import get_file
from . import UNMERGED_NTUPLE_PATH
from samples import Tau, Jet, JZ
from .variables import VARIABLES
from .parallel import FuncWorker, run_pool



# class Classifier(ROOT.TMVA.Factory):
class Classifier(object):

    def __init__(self, 
                 category,
                 output_name,
                 factory_name,
                 prefix='hlt',
                 tree_name='tau',
                 training_mode='dev', # 'prod'
                 features=None,
                 train_split='odd',
                 test_split='even',
                 verbose=''):

        self.output_name = output_name
        self.factory_name = factory_name
        self.category = category
        self.prefix = prefix
        self.tree_name = tree_name
        self.training_mode = training_mode
        self.features = features
        self.train_split = train_split
        self.test_split = test_split
        self.verbose = verbose


    def set_variables(self, factory, category, prefix):
        if self.features is None:
            self.features = category.features
        for varName in self.features:
            var = VARIABLES[varName]
            factory.AddVariable(prefix + '_' + var['name'], var['root'], '', var['type'])

    def book(self,
             factory,
             ntrees=100,
             node_size=5,
             depth=8):
        
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
        log.info(params)

        method_name = "BDT_{0}".format(self.factory_name)
        params_string = "!H:V"
        for param in params:
            params_string+= ":"+param
        log.info('booking ..')
        factory.BookMethod(
            ROOT.TMVA.Types.kBDT,
            method_name,
            params_string)

    def train(self, tau, jet, **kwargs):

        #         params  = ["PruneBeforeBoost=False"]
        output = root_open(self.output_name, 'recreate')
        factory = ROOT.TMVA.Factory(self.factory_name, output, self.verbose)

        self.set_variables(factory, self.category, self.prefix)
        self.sig_cut = tau.cuts(self.category, feat_cuts=True) 
        self.bkg_cut = jet.cuts(self.category, feat_cuts=True) 

        params = ['NormMode=EqualNumEvents']
        params += ['SplitMode=Block']
        params_string = ':'.join(params)

        log.info(self.sig_cut)
        log.info(self.bkg_cut)
        
        # Signal file
        log.info('prepare signal tree')
        sig_file = get_file(tau.ntuple_path, tau.student) 
        self.sig_tree_train = sig_file.Get(tau.tree_name + '_' + self.train_split)
        self.sig_tree_test = sig_file.Get(tau.tree_name + '_' + self.test_split)

        factory.AddSignalTree(self.sig_tree_train, 1., ROOT.TMVA.Types.kTraining)
        factory.AddSignalTree(self.sig_tree_test, 1., ROOT.TMVA.Types.kTesting)
        factory.SetSignalWeightExpression(tau.weight_field)

        # Bkg files
        log.info('prepare background tree')
        if isinstance(jet, JZ):
            for sample, scale in zip(jet.components, jet.scales):
                rfile = get_file(sample.ntuple_path, sample.student)
                tree = rfile[sample.tree_name]
                self.AddBackgroundTree(tree, scale)
        else:
            bkg_file = get_file(jet.ntuple_path, jet.student)
            self.bkg_tree = bkg_file[jet.tree_name]
            self.bkg_tree_train = bkg_file.Get(jet.tree_name + '_' + self.train_split)
            self.bkg_tree_test = bkg_file.Get(jet.tree_name + '_' + self.test_split)
            factory.AddBackgroundTree(self.bkg_tree_train, 1., ROOT.TMVA.Types.kTraining)
            factory.AddBackgroundTree(self.bkg_tree_test, 1., ROOT.TMVA.Types.kTesting)

        factory.SetBackgroundWeightExpression(jet.weight_field)
        log.info('preparation the trees')
        output.cd()
        factory.PrepareTrainingAndTestTree(
            self.sig_cut, self.bkg_cut, params_string)
        log.info('preparation is done, start booking')
        # Actual training
        self.book(factory, **kwargs)
        # booking is done, start training
        log.info(ROOT.gDirectory.GetPath())
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()
        log.info(ROOT.gDirectory.GetPath())
        output.Close()



class working_point(object):
    def __init__(self, cut, eff_s, eff_b, name='wp'):
        self.name = name
        self.cut = cut
        self.eff_s = eff_s
        self.eff_b = eff_b
