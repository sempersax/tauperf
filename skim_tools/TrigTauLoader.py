#TauLoader class
import ROOT
import math
from helpers.andrew_variables import andrew_variables

class EFTauLoader:
    """A class to dump tau-related info at Event Filter level"""

    #-----------------------------------------------------------
    def __init__(self, tree, ef_index, include):
        """Constructor"""

        self._tree           = tree
        self._trigIndex      = -1
        self.author          = -1
        self.hasEFmatched    = False

        if ef_index > -1:
            self._trigIndex      = ef_index
            self.hasEFmatched    = True
            self.author          = self._tree.trig_EF_tau_author[self._trigIndex]  
            
            # --> trigger chain
            self.tau20_medium1 = self._tree.trig_EF_tau_EF_tau20_medium1[self._trigIndex]
            self.tauNoCut      = self._tree.trig_EF_tau_EF_tauNoCut[self._trigIndex]

            if 'basic' in include:
                self.EF4Vector                = self.getEventFilter4Vector()
                self.nTracks, self.nClusters, self.nEffClusters    = self.getEventFilterDecayN()
                self.nWideTrk        = self._tree.trig_EF_tau_seedCalo_nWideTrk    [self._trigIndex]
                self.numTrack        = self._tree.trig_EF_tau_numTrack             [self._trigIndex]
                self.notherTrk       = self._tree.trig_EF_tau_otherTrk_n           [self._trigIndex]                 
                
            if 'EventFilterObjects' in include:
                # self.clusters                   = self.getClusters()
                self.tracks,self.tracks_d0,self.tracks_z0 = self.getTracks()
                
            if 'wideTracks' in include:
                self.wideTracks,self.wideTracks_d0,self.wideTracks_z0 = self.getWideTracks()
                self.wideTracks_nBLHits,self.wideTracks_nPixHits,self.wideTracks_nSCTHits,self.wideTracks_nTRTHits,self.wideTracks_nHits = self.getWideTracksHits()
                    
            if 'otherTracks' in include:
                self.otherTracks,self.otherTracks_d0,self.otherTracks_z0 = self.getOtherTracks()
                self.otherTracks_nBLHits,self.otherTracks_nPixHits,self.otherTracks_nSCTHits,self.otherTracks_nTRTHits,self.otherTracks_nHits = self.getOtherTracksHits()

            if 'EDMVariables' in include:
                self.nStrip             = self._tree.trig_EF_tau_seedCalo_nStrip              [self._trigIndex]
                self.etOverPt           = self._tree.trig_EF_tau_etOverPtLeadTrk              [self._trigIndex]
                self.mEflow             = self._tree.trig_EF_tau_mEflow                       [self._trigIndex]
                # self.nEffStripCells     = self._tree.trig_EF_tau_cell_nEffStripCells          [self._trigIndex]
                self.effMass            = self._tree.trig_EF_tau_effTopoInvMass               [self._trigIndex]
                self.EMRadius           = self._tree.trig_EF_tau_seedCalo_EMRadius            [self._trigIndex]
                self.TrkAvgDist         = self._tree.trig_EF_tau_seedCalo_trkAvgDist          [self._trigIndex]               
                self.ChPiEMEOverCaloEME = self._tree.trig_EF_tau_calcVars_ChPiEMEOverCaloEME  [self._trigIndex]
                self.PSSFraction        = self._tree.trig_EF_tau_calcVars_PSSFraction         [self._trigIndex]
                self.EMPOverTrkSysP     = self._tree.trig_EF_tau_calcVars_EMPOverTrkSysP      [self._trigIndex]
                self.pi0BDTPrimary      = self._tree.trig_EF_tau_calcVars_pi0BDTPrimaryScore  [self._trigIndex]
                self.pi0BDTSecondary    = self._tree.trig_EF_tau_calcVars_pi0BDTSecondaryScore[self._trigIndex]

            if 'TauID' in include:
                self.corrCentFrac    = self._tree.trig_EF_tau_calcVars_corrCentFrac[self._trigIndex]
                self.centFrac        = self._tree.trig_EF_tau_seedCalo_centFrac    [self._trigIndex]
                self.corrFTrk        = self._tree.trig_EF_tau_calcVars_corrFTrk    [self._trigIndex]
                self.trkAvgDist      = self._tree.trig_EF_tau_seedCalo_trkAvgDist  [self._trigIndex]
                self.ipSigLeadTrk    = self._tree.trig_EF_tau_ipSigLeadTrk         [self._trigIndex]
                # self.pi0_ptratio     = self._tree.trig_EF_tau_pi0_ptratio          [self._trigIndex]
                self.pi0_vistau_m    = self._tree.trig_EF_tau_pi0_vistau_m         [self._trigIndex]
                self.pi0_n           = self._tree.trig_EF_tau_pi0_n                [self._trigIndex]
                self.trFlightPathSig = self._tree.trig_EF_tau_trFlightPathSig      [self._trigIndex]
                self.massTrkSys      = self._tree.trig_EF_tau_massTrkSys           [self._trigIndex]
                self.topoMeanDeltaR  = self._tree.trig_EF_tau_topoMeanDeltaR       [self._trigIndex]
                self.CaloRadius      = self._tree.trig_EF_tau_calcVars_calRadius   [self._trigIndex]
                self.HADRadius       = self._tree.trig_EF_tau_seedCalo_hadRadius   [self._trigIndex]
                self.IsoFrac         = self._tree.trig_EF_tau_seedCalo_isolFrac    [self._trigIndex]
                self.EMFrac          = self._tree.trig_EF_tau_calcVars_EMFractionAtEMScale [self._trigIndex]
                self.stripWidth      = self._tree.trig_EF_tau_seedCalo_stripWidth2 [self._trigIndex]
                self.dRmax           = self._tree.trig_EF_tau_seedCalo_dRmax       [self._trigIndex]

            if 'CaloDetails' in include:
                self.PSEnergy    = self._tree.trig_EF_tau_calcVars_PSEnergy   [self._trigIndex]
                self.S1Energy    = self._tree.trig_EF_tau_calcVars_S1Energy   [self._trigIndex]
                self.S2Energy    = self._tree.trig_EF_tau_calcVars_S2Energy   [self._trigIndex]
                self.S3Energy    = self._tree.trig_EF_tau_calcVars_S3Energy   [self._trigIndex]
                self.HADEnergy   = self._tree.trig_EF_tau_calcVars_HADEnergy [self._trigIndex]
                self.CaloEnergy  = self._tree.trig_EF_tau_calcVars_CaloEnergy [self._trigIndex]

                
        return None

    #-----------------------------------------------------------
    def getEventFilter4Vector(self):
        """Get the TLorentzVector for the EF tau"""
        pt  = self._tree.trig_EF_tau_pt [self._trigIndex]
        eta = self._tree.trig_EF_tau_eta[self._trigIndex]
        phi = self._tree.trig_EF_tau_phi[self._trigIndex]
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, 0)
        return vector

        

    #-----------------------------------------------------------
    def getTracks(self):
        """Returns the standard tracks container"""
        tracks = []
        d0=[]
        z0=[]
        n = self._tree.trig_EF_tau_track_n[self._trigIndex]

        for i in range(0, n):
            pt  = self._tree.trig_EF_tau_track_pt [self._trigIndex][i]
            eta = self._tree.trig_EF_tau_track_eta[self._trigIndex][i]
            phi = self._tree.trig_EF_tau_track_phi[self._trigIndex][i]
            m   = 140
            d0.append(self._tree.trig_EF_tau_track_d0[self._trigIndex][i])
            z0.append(self._tree.trig_EF_tau_track_z0[self._trigIndex][i])
            vector = ROOT.TLorentzVector()
            vector.SetPtEtaPhiM(pt, eta, phi, m)
            tracks.append(vector)
        return tracks,d0,z0


    #-----------------------------------------------------------
    def getWideTracks(self):
        """Returns the seedCalo tracks"""
        tracks = []
        d0     = []
        z0     = []
        n = self._tree.trig_EF_tau_seedCalo_wideTrk_n[self._trigIndex]
        for i in range(0, n):
            pt  = self._tree.trig_EF_tau_seedCalo_wideTrk_pt [self._trigIndex][i]
            eta = self._tree.trig_EF_tau_seedCalo_wideTrk_eta[self._trigIndex][i]
            phi = self._tree.trig_EF_tau_seedCalo_wideTrk_phi[self._trigIndex][i]
            m   = 0
            vector = ROOT.TLorentzVector()
            vector.SetPtEtaPhiM(pt, eta, phi, m)
            tracks.append(vector)
            d0.append(self._tree.trig_EF_tau_seedCalo_wideTrk_d0[self._trigIndex][i])
            z0.append(self._tree.trig_EF_tau_seedCalo_wideTrk_z0[self._trigIndex][i])
        return tracks,d0,z0


    #-----------------------------------------------------------
    def getOtherTracks(self):
        """Returns the otherTrk tracks"""
        tracks = []
        d0     = []
        z0     = []
        n = self._tree.trig_EF_tau_otherTrk_n[self._trigIndex]

        for i in range(0, n):
            pt  = self._tree.trig_EF_tau_otherTrk_pt [self._trigIndex][i]
            eta = self._tree.trig_EF_tau_otherTrk_eta[self._trigIndex][i]
            phi = self._tree.trig_EF_tau_otherTrk_phi[self._trigIndex][i]
            m   = 0
            vector = ROOT.TLorentzVector()
            vector.SetPtEtaPhiM(pt, eta, phi, m)
            tracks.append(vector)
            d0.append(self._tree.trig_EF_tau_otherTrk_d0[self._trigIndex][i])
            z0.append(self._tree.trig_EF_tau_otherTrk_z0[self._trigIndex][i])
        return tracks,d0,z0


    #-----------------------------------------------------------
    def getWideTracksHits(self):
        """Returns the seedCalo tracks hits"""
        nBLHits  = []
        nPixHits = []
        nSCTHits = []
        nTRTHits = []
        nHits    = []
        n = self._tree.trig_EF_tau_seedCalo_wideTrk_n[self._trigIndex]
        for i in range(0, n):
            nBLHits  .append( self._tree.trig_EF_tau_seedCalo_wideTrk_nBLHits[self._trigIndex][i]  )
            nPixHits .append( self._tree.trig_EF_tau_seedCalo_wideTrk_nPixHits[self._trigIndex][i] )
            nSCTHits .append( self._tree.trig_EF_tau_seedCalo_wideTrk_nSCTHits[self._trigIndex][i] )
            nTRTHits .append( self._tree.trig_EF_tau_seedCalo_wideTrk_nTRTHits[self._trigIndex][i] )
            nHits    .append( self._tree.trig_EF_tau_seedCalo_wideTrk_nHits[self._trigIndex][i] )
        return nBLHits,nPixHits,nSCTHits,nTRTHits,nHits


    #-----------------------------------------------------------
    def getOtherTracksHits(self):
        """Returns the otherTrk tracks hits"""
        nBLHits  = []
        nPixHits = []
        nSCTHits = []
        nTRTHits = []
        nHits    = []
        n = self._tree.trig_EF_tau_otherTrk_n[self._trigIndex]
        for i in range(0, n):
            nBLHits  .append( self._tree.trig_EF_tau_otherTrk_nBLHits[self._trigIndex][i]  )
            nPixHits .append( self._tree.trig_EF_tau_otherTrk_nPixHits[self._trigIndex][i] )
            nSCTHits .append( self._tree.trig_EF_tau_otherTrk_nSCTHits[self._trigIndex][i] )
            nTRTHits .append( self._tree.trig_EF_tau_otherTrk_nTRTHits[self._trigIndex][i] )
            nHits    .append( self._tree.trig_EF_tau_otherTrk_nHits[self._trigIndex][i] )
        return nBLHits,nPixHits,nSCTHits,nTRTHits,nHits


    #-----------------------------------------------------------
    def getEventFilterDecayN(self):
        """get numbers of tracks and clusters"""
        # --> Tracks are not taken at TJVA.. Different from offline ! 
        # --> Nclusters and nEffClusters are not filled yet.
        nTracks      = self._tree.trig_EF_tau_track_n                         [self._trigIndex]
        nClusters    = self._tree.trig_EF_tau_numTopoClusters                 [self._trigIndex]
        nEffClusters = int(math.ceil(self._tree.trig_EF_tau_numEffTopoClusters[self._trigIndex]))
        
        return nTracks, nClusters, nEffClusters

    # -----------------------------------------------------------
    def getL2TauIndex(self):
        """Perform a matching between the EF and L2 containers """
        ef_roi = self._tree.trig_EF_tau_RoIWord[self._trigIndex]
        index = -1
        for i in range(0,self._tree.trig_L2_tau_n):
            l2_roi = self._tree.trig_L2_tau_RoIWord[i]
            if l2_roi == ef_roi:
                index = i
                break
        return index







###########################################################################################
#### ----------------   L2TAULOADER         ------------------                        #####
###########################################################################################
class L2TauLoader:
    """A class to dump tau-related info at Event Filter level"""

    #-----------------------------------------------------------
    def __init__(self, tree, l2_index, include):
        """Constructor"""

        self._tree           = tree
        self._trigIndex      = -1
        self.hasL2matched    = False

        if l2_index > -1:
            self._trigIndex      = l2_index
            self.hasL2matched    = True

            # --> trigger chain
            self.tau20_medium                 = self._tree.trig_L2_tau_L2_tau20_medium[self._trigIndex]
            self.tau20_medium1                = self._tree.trig_L2_tau_L2_tau20_medium1[self._trigIndex]
            self.tauNoCut                     = self._tree.trig_L2_tau_L2_tauNoCut[self._trigIndex]
            if 'trigger_14TeV' in include:
                self.tau18Ti_loose2_e18vh_medium1 = self._tree.trig_L2_tau_L2_tau18Ti_loose2_e18vh_medium1[self._trigIndex]
            if 'basic' in include:
                self.L2_4Vector  = self.getL2_4Vector()

            if 'TauID' in include:
                self.CaloRadius      = self._tree.trig_L2_tau_cluster_CaloRadius [self._trigIndex]
                self.HADRadius       = self._tree.trig_L2_tau_cluster_HADRadius  [self._trigIndex]
                self.IsoFrac         = self._tree.trig_L2_tau_cluster_IsoFrac    [self._trigIndex]
                self.EMFrac          = self._tree.trig_L2_tau_cluster_EMFrac     [self._trigIndex]
                self.stripWidth      = self._tree.trig_L2_tau_cluster_stripWidth [self._trigIndex]
                and_var = andrew_variables( self._tree.trig_L2_tau_cluster_HADenergy[self._trigIndex],
                                            self._tree.trig_L2_tau_cluster_EMenergy[self._trigIndex],
                                            self._tree.trig_L2_tau_cluster_numTotCells[self._trigIndex])
                self.HADtoEMEnergy  = and_var.HADtoEMEnergy
                self.EnergyTonCells = and_var.EnergyTonCells

                
        return None

        
    #-----------------------------------------------------------
    def getL2_4Vector(self):
        """Get the TLorentzVector for the EF reco. tau"""
        pt  = self._tree.trig_L2_tau_pt [self._trigIndex]
        eta = self._tree.trig_L2_tau_eta[self._trigIndex]
        phi = self._tree.trig_L2_tau_phi[self._trigIndex]
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, 0)
        return vector

    # -----------------------------------------------------------
    def getL1TauIndex(self):
        """Perform a matching between the L2 and L1 containers """
        l2_roi = self._tree.trig_L2_tau_RoIWord[self._trigIndex]
        index = -1
        for i in range(0,self._tree.trig_L1_emtau_n):
            l1_roi = self._tree.trig_L1_emtau_RoIWord[i]
            if l1_roi == l2_roi:
                index = i
                break
        return index
        








###########################################################################################
#### ----------------   L1TAULOADER         ------------------                        #####
###########################################################################################
    
class L1TauLoader:
    """A class to dump tau-related info at level 1"""

    #-----------------------------------------------------------
    def __init__(self, tree, l1_index, include):
        """Constructor"""

        self._tree           = tree
        self._trigIndex      = -1
        self.author          = -1
        self.hasL1matched    = False

        if l1_index > -1:
            self._trigIndex      = l1_index
            self.hasL1matched    = True
            if 'basic' in include:
                self.L1_4Vector  = self.get4Vector()
        return None


    #-----------------------------------------------------------
    def get4Vector(self):
        """Get the TLorentzVector for the EF reco. tau"""
        pt  = self._tree.trig_L1_emtau_tauClus [self._trigIndex]
        eta = self._tree.trig_L1_emtau_eta[self._trigIndex]
        phi = self._tree.trig_L1_emtau_phi[self._trigIndex]
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, 0)
        return vector
