# ---> python imports
import math
# ---> rootpy imports
from rootpy import asrootpy
from rootpy.tree import TreeModel, FloatCol, IntCol
from rootpy.vector import LorentzVector
# ---> local imports
from rootpy import log
ignore_warning = log['/ROOT.TVector3.PseudoRapidity'].ignore('.*transvers momentum.*')

# __all__ = ['EventInfo',
#            'FourMomentum',
#            'TrueTau',
#            'CaloTau',
#            'TrackTau',
#            'CaloTrackTau',
#            'ClusterBasedTau',
#            'RecoTau',
#            'EFTau',
#            'L2Tau',
#            'L1Tau',]

class EventInfo(TreeModel):
    runnumber              = IntCol()
    evtnumber              = IntCol()
    lumiblock              = IntCol()
    npv                    = IntCol()
    mu                     = FloatCol()
    chain_EF_tau20_medium1 = IntCol()
    chain_EF_tauNoCut      = IntCol()
    chain_L2_tauNoCut      = IntCol()
    chain_L1_TAU8          = IntCol()
    chain_L1_TAU11I        = IntCol()
    chain_L2_tau18Ti_loose2_e18vh_medium1 = IntCol(-1111)

    @classmethod
    def set(cls, this, other):
        this.runnumber              = other.RunNumber
        this.evtnumber              = other.EventNumber
        this.lumiblock              = other.lbn
        this.npv                    = other.evt_calcVars_numGoodVertices
        this.mu                     = other.averageIntPerXing
        this.chain_EF_tau20_medium1 = other.EF_tau20_medium1 
        this.chain_EF_tauNoCut      = other.EF_tauNoCut      
        this.chain_L2_tauNoCut      = other.L2_tauNoCut      
        this.chain_L1_TAU8          = other.L1_TAU8          
        this.chain_L1_TAU11I        = other.L1_TAU11I        
        # this.chain_L2_tau18Ti_loose2_e18vh_medium1 = other.L2_tau18Ti_loose2_e18vh_medium1

class FourMomentum(TreeModel):
    pt  = FloatCol(default=-1111.)
    p   = FloatCol(default=-1111.)
    et  = FloatCol(default=-1111.)
    e   = FloatCol(default=-1111.)
    eta = FloatCol(default=-1111.)
    phi = FloatCol(default=-1111.)
    m   = FloatCol(default=-1111.)

    @classmethod
    def set(cls, this, other):
        if isinstance(other, LorentzVector):
            vect = other
        else:
            vect = other.fourvect
        this.pt = vect.Pt()
        this.p = vect.P()
        this.et = vect.Et()
        this.e = vect.E()
        this.m = vect.M()
        with ignore_warning:
            this.phi = vect.Phi()
            this.eta = vect.Eta()

class TrueTau(FourMomentum):
    nProng       = IntCol(default=-1111)
    nPi0         = IntCol(default=-1111)
    charge       = IntCol(default=-1111)
    vis_pt       = FloatCol(default=-1111.)
    vis_p        = FloatCol(default=-1111.)
    vis_et       = FloatCol(default=-1111.)
    vis_e        = FloatCol(default=-1111.)
    vis_eta      = FloatCol(default=-1111.)
    vis_phi      = FloatCol(default=-1111.)
    vis_m        = FloatCol(default=-1111.)

    @classmethod
    def set(cls,this,other):
        FourMomentum.set(this,other)
        this.nProng = other.nProng
        this.nPi0   = other.nPi0
        this.charge = other.charge
        this.vis_pt       = other.fourvect_vis.Pt()
        this.vis_p        = other.fourvect_vis.P()
        this.vis_et       = other.fourvect_vis.Et()
        this.vis_e        = other.fourvect_vis.E()
        this.vis_m        = other.fourvect_vis.M()
        with ignore_warning:
            this.vis_eta      = other.fourvect_vis.Eta()        
            this.vis_phi      = other.fourvect_vis.Phi()


class CaloTau(TreeModel):
    PSSFraction        = FloatCol(-1111.)
    nStrip             = IntCol(-1111)
    nEffStripCells     = FloatCol(-1111.)
    corrCentFrac       = FloatCol(-1111.)
    centFrac           = FloatCol(-1111.)
    isolFrac           = FloatCol(-1111.)
    EMRadius           = FloatCol(-1111.)
    HadRadius          = FloatCol(-1111.)
    EMEnergy           = FloatCol(-1111.)
    HadEnergy          = FloatCol(-1111.)
    CaloRadius         = FloatCol(-1111.)
    stripWidth2        = FloatCol(-1111.)
    numTopoClusters    = IntCol(-1111)
    numEffTopoClusters =  FloatCol(-1111.)
    topoInvMass        =  FloatCol(-1111.)
    effTopoInvMass     =  FloatCol(-1111.)
    topoMeanDeltaR     =  FloatCol(-1111.)
    effTopoMeanDeltaR =  FloatCol(-1111.)
    lead2ClusterEOverAllClusterE  = FloatCol(-1111.) 
    lead3ClusterEOverAllClusterE =  FloatCol(-1111.)
    EMFractionAtEMScale          =  FloatCol(-1111.)
    HADFractionAtEMScale         =  FloatCol(-1111.)

    @classmethod
    def set(cls,this,other):
        this.PSSFraction                  = other.calcVars_PSSFraction
        this.nStrip                       = other.seedCalo_nStrip
        this.corrCentFrac                 = other.calcVars_corrCentFrac
        this.centFrac                     = other.seedCalo_centFrac
        this.isolFrac                     = other.seedCalo_isolFrac
        this.EMRadius                     = other.seedCalo_EMRadius
        this.HadRadius                    = other.seedCalo_hadRadius
        this.EMEnergy                     = other.seedCalo_etEMAtEMScale
        this.HadEnergy                    = other.seedCalo_etHadAtEMScale
        this.CaloRadius                   = (this.EMRadius*this.EMEnergy+this.HadRadius*this.HadEnergy)
        this.stripWidth2                  = other.seedCalo_stripWidth2
        this.numTopoClusters              = other.numTopoClusters
        this.numEffTopoClusters           = other.numEffTopoClusters
        this.topoInvMass                  = other.topoInvMass
        this.effTopoInvMass               = other.effTopoInvMass
        this.topoMeanDeltaR               = other.topoMeanDeltaR
        this.effTopoMeanDeltaR            = other.effTopoMeanDeltaR
        this.lead2ClusterEOverAllClusterE = other.seedCalo_lead2ClusterEOverAllClusterE
        this.lead3ClusterEOverAllClusterE = other.seedCalo_lead3ClusterEOverAllClusterE
        this.EMFractionAtEMScale          = other.calcVars_EMFractionAtEMScale
        seedCalo_et = other.seedCalo_etEMAtEMScale+other.seedCalo_etHadAtEMScale
        this.HADFractionAtEMScale = 0 if seedCalo_et==0 else other.seedCalo_etHadAtEMScale/(seedCalo_et)

class TrackTau(TreeModel):
    numTrack         = IntCol(default = -1111)
    nWideTrk         = IntCol(default = -1111)
    nOtherTrk        = IntCol(default = -1111)
    ipSigLeadTrk     = FloatCol(-1111.)
    trFlightPathSig  = FloatCol(-1111.)
    massTrkSys       = FloatCol(-1111.)
    dRmax            = FloatCol(-1111.)

    @classmethod
    def set(cls,this,other):
        this.numTrack        = other.numTrack
        this.nWideTrk        = other.seedCalo_nWideTrk
        this.nOtherTrk       = other.otherTrk_n
        this.ipSigLeadTrk    = other.ipSigLeadTrk
        this.trFlightPathSig = other.trFlightPathSig        
        this.massTrkSys      = other.massTrkSys
        this.dRmax           = other.seedCalo_dRmax
    
class CaloTrackTau(TreeModel):
    EMPOverTrkSysP     = FloatCol(-1111.)
    ChPiEMEOverCaloEME = FloatCol(-1111.) 
    EtOverLeadTrackPt  = FloatCol(-1111.)
    corrFTrk           = FloatCol(-1111.)
    trkAvgDist         = FloatCol(-1111.)

    @classmethod
    def set(cls,this,other):
        this.EMPOverTrkSysP     = other.calcVars_EMPOverTrkSysP
        this.ChPiEMEOverCaloEME = other.calcVars_ChPiEMEOverCaloEME
        this.corrFTrk           = other.calcVars_corrFTrk
        this.EtOverLeadTrackPt  = 1./this.corrFTrk
        this.trkAvgDist         = other.seedCalo_trkAvgDist

class ClusterBasedTau(TreeModel):
    pi0BDTPrimary    = FloatCol(-1111.)
    pi0BDTSecondary  = FloatCol(-1111.)
    pi0_ptratio      = FloatCol(-1111.)
    pi0_vistau_m     = FloatCol(-1111.)
    pi0_n            = IntCol(-1111)
    clbased_pt = FloatCol(-1111.)

    @classmethod
    def set(cls,this,other):
        this.pi0BDTPrimary    = other.calcVars_pi0BDTPrimaryScore
        this.pi0BDTSecondary  = other.calcVars_pi0BDTSecondaryScore
        this.pi0_ptratio      = other.pi0_vistau_pt/other.pt
        this.pi0_vistau_m     = other.pi0_vistau_m
        this.pi0_n            = other.pi0_n
        this.clbased_pt = other.fourvect_clbased.Pt()

class RecoTau(FourMomentum+CaloTau+TrackTau+CaloTrackTau+ClusterBasedTau):
    index              = IntCol(default=-1)
    BDTloose           = FloatCol(default=-1111.)
    BDTmedium          = FloatCol(default=-1111.)
    BDTtight           = FloatCol(default=-1111.)
    index_matched_true = IntCol(default=-1)
    index_matched_EF   = IntCol(default=-1)
    index_matched_L1   = IntCol(default=-1)

class EFTau(FourMomentum+CaloTau+TrackTau+CaloTrackTau):
    index            = IntCol(default=-1)
    index_matched_L2 = IntCol(default=-1)
    
class L2Tau(FourMomentum):
    index = IntCol(default=-1)
    index_matched_L1 = IntCol(default=-1)

class L1Tau(FourMomentum):
    index = IntCol(default=-1)

class Track(FourMomentum):
    d0       = FloatCol(default=-1111.)
    z0       = FloatCol(default=-1111.)
    nBLHits  = IntCol(default=-1111)
    nPixHits = IntCol(default=-1111)
    nSCTHits = IntCol(default=-1111)
    nTRTHits = IntCol(default=-1111)
    nHits    = IntCol(default=-1111)
