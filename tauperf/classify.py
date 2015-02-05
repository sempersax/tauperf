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
from .analysis import Analysis
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

    def bookBDT(self,
                ntrees=100,
                node_size=5,
                depth=8):

        #         params  = ["PruneBeforeBoost=False"]
        params = ["SeparationType=GiniIndex"]
        params += ["BoostType=AdaBoost"]
        params += ["PruneMethod=NoPruning"]
        params += ["UseYesNoLeaf=False"]
        params += ["AdaBoostBeta=0.2"]
        params += ["DoBoostMonitor"]
        params += ["MaxDepth={0}".format(depth)]
        params += ["MinNodeSize={0}%".format(node_size)]
        params += ["NTrees={0}".format(ntrees)]
        #         params += ["nCuts={0}".format(nCuts)]
        #         params += ["NNodesMax={0}".format(NNodesMax)]
        #         params += ["nEventsMin={0}".format(nEventsMin)]
        log.info(params)

        method_name = "BDT_{0}".format(self.factory_name)
        params_string = "!H:V"
        for param in params:
            params_string+= ":"+param
        self.BookMethod(
            TMVA.Types.kBDT,
            method_name,
            params_string)

    def train(self, **kwargs):
        self.set_variables(self.category, self.prefix)
        ana = Analysis(ntuple_path=os.path.join(UNMERGED_NTUPLE_PATH, 'merge_corr'))
        tau = ana.tau
        jet = ana.jet
        self.sig_cut = tau.cuts(self.category) & self.split_cut
        self.bkg_cut = jet.cuts(self.category) & self.split_cut

        params = ['NormMode=EqualNumEvents']
        params += ['SplitMode=Random']
        if self.training_mode == 'prod':
            params += ['nTest_Background=1']
            params += ['nTest_Signal=1']
            params += ['!V']
        params_string = ':'.join(params)

        self.PrepareTrainingAndTestTree(self.sig_cut, self.bkg_cut, params_string)

        # Signal file
        sig_file = get_file(tau.ntuple_path, tau.student) 
        self.sig_tree = sig_file[tau.tree_name]
        self.AddSignalTree(self.sig_tree)

        # Bkg files
        for sample, scale in zip(jet.components, jet.scales):
            rfile = get_file(sample.ntuple_path, sample.student)
            tree = rfile[sample.tree_name]
            self.AddBackgroundTree(tree, scale)
        self.SetBackgroundWeightExpression(jet.weight_field)

        # Actual training
        self.bookBDT(**kwargs)
        self.TrainAllMethods()
        if self.training_mode == 'dev':
            self.output.cd()
            self.TestAllMethods()
            self.EvaluateAllMethods()
