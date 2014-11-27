# ---> local imports
from .model import *
from rootpy import log
ignore_warning = log['/ROOT.TVector3.PseudoRapidity'].ignore('.*transvers momentum.*')

IncludeList = ['Truth', 'Calo', 'Track', 'CaloTrack', 'ClusterBased']


class EventInfoBlock(EventInfo):
    @classmethod
    def set(cls, event, tree):
        EventInfo.set(tree, event)

class TrueTauBlock(TrueTau.prefix('true_')):
    @classmethod
    def set(cls, event, tree, tau):
        outtau, intau = tree.tau_true, tau
        outtau.index = intau.index
        TrueTau.set(outtau, intau)

class RecoTauBlock(RecoTau.prefix('off_')):
    @classmethod
    def set(cls, event, tree, tau):
        outtau, intau = tree.tau, tau
        outtau.index = intau.index
        FourMomentum.set(outtau, intau)
        if 'Truth' in IncludeList:
            outtau.index_matched_true = intau.trueTauAssoc_index
        if 'Calo' in IncludeList:
            CaloTau.set(outtau, intau)
            # HACK HACK HACK (VARIABLE NOT DEFINED FOR EF TAUS)
            outtau.nEffStripCells = intau.cell_nEffStripCells
        if 'Track' in IncludeList:
            TrackTau.set(outtau, intau)
        if 'CaloTrack' in IncludeList:
            CaloTrackTau.set(outtau, intau)
        if 'ClusterBased' in IncludeList:
            ClusterBasedTau.set(outtau, intau)

class EFTauBlock(EFTau.prefix('EF_')):
    @classmethod
    def set(cls, event, tree, EFtau):
        outtau, intau = tree.tau_EF, EFtau
        outtau.index = intau.index
        FourMomentum.set(outtau, intau)
        if 'Calo' in IncludeList:
            CaloTau.set(outtau, intau)
        if 'Track' in IncludeList:
            TrackTau.set(outtau, intau)
        if 'CaloTrack' in IncludeList:
            CaloTrackTau.set(outtau, intau)

class L2TauBlock(L2Tau.prefix('L2_')):
    @classmethod
    def set(cls, event, tree, L2tau):
        outtau, intau = tree.tau_L2, L2tau
        outtau.index = intau.index
        FourMomentum.set(outtau, intau)

class L1TauBlock(L1Tau.prefix('L1_')):
    @classmethod
    def set(cls, event, tree, L1tau):
        outtau, intau = tree.tau_L1, L1tau
        outtau.index = intau.index
        FourMomentum.set(outtau, intau)

class L1_OfflineMatched_TauBlock(L1Tau.prefix('L1_OfflineMatched_')):
    @classmethod
    def set(cls, event, tree, L1tau):
        outtau, intau = tree.tau_L1_OfflineMatched, L1tau
        outtau.index = intau.index
        FourMomentum.set(outtau, intau)
