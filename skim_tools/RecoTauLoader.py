#RecoTauLoader class
from rootpy import stl
import ROOT
from ROOT import TLorentzVector
import math
from substructure_tools.ClusterBasedTau import ClusterBasedTau

class RecoTauLoader:
    """A class to dump tau-related info for reconstructed taus."""

    #-----------------------------------------------------------
    def __init__(self, tree, reco_index, include):
        """Constructor"""
        self._tree           = tree

        if reco_index > -1:
            self._recoIndex  = reco_index
            self.hasReco     = True
            self.author      = self._tree.tau_author[self._recoIndex]  
          
            self.EFIndex,self.DeltaR_EF_off = self.getEventFilterTauIndex()
            self.L1Index,self.DeltaR_L1_off = self.getL1TauIndex()
                
            if 'truth' in include:
                self.truthIndex = self.getTruthIndex()
                if self.truthIndex>-1: self.hasTruth = True
                else: self.hasTruth = False

                self.truthIndex_alt,self.truth_dR = self.getTruthIndex_dR()
                if self.truthIndex_alt>-1: self.hasTruth_dR = True
                else: self.hasTruth_dR = False


            if 'basic' in include:
                self.reco4Vector = self.getReco4Vector()
                self.nTracks, self.nClusters, self.nEffClusters    = self.getRecoDecayN()
                self.numTrack        = self._tree.tau_numTrack             [self._recoIndex]
                self.nWideTrk        = self._tree.tau_seedCalo_nWideTrk    [self._recoIndex]
                self.notherTrk       = self._tree.tau_otherTrk_n           [self._recoIndex]

            if 'EDMVariables' in include:
                self.nStrip             = self._tree.tau_seedCalo_nStrip               [self._recoIndex]
                self.etOverPt           = self._tree.tau_etOverPtLeadTrk               [self._recoIndex]
                self.mEflow             = self._tree.tau_mEflow                        [self._recoIndex]
                self.nEffStripCells     = self._tree.tau_cell_nEffStripCells           [self._recoIndex]
                self.effMass            = self._tree.tau_effTopoInvMass                [self._recoIndex]
                self.EMRadius           = self._tree.tau_seedCalo_EMRadius             [self._recoIndex]
                self.TrkAvgDist         = self._tree.tau_seedCalo_trkAvgDist           [self._recoIndex]               
                self.ChPiEMEOverCaloEME = self._tree.tau_calcVars_ChPiEMEOverCaloEME   [self._recoIndex]
                self.PSSFraction        = self._tree.tau_calcVars_PSSFraction          [self._recoIndex]
                self.EMPOverTrkSysP     = self._tree.tau_calcVars_EMPOverTrkSysP       [self._recoIndex]
                self.pi0BDTPrimary      = self._tree.tau_calcVars_pi0BDTPrimaryScore   [self._recoIndex]
                self.pi0BDTSecondary    = self._tree.tau_calcVars_pi0BDTSecondaryScore [self._recoIndex]

            if 'TauID' in include:
                self.BDTJetScore     = self._tree.tau_BDTJetScore          [self._recoIndex]
                self.BDTloose        = self._tree.tau_JetBDTSigLoose       [self._recoIndex]
                self.BDTmedium       = self._tree.tau_JetBDTSigMedium      [self._recoIndex]
                self.BDTtight        = self._tree.tau_JetBDTSigTight       [self._recoIndex]                
                self.BDTEleScore     = self._tree.tau_BDTEleScore          [self._recoIndex] 
                self.corrCentFrac    = self._tree.tau_calcVars_corrCentFrac[self._recoIndex]
                self.centFrac        = self._tree.tau_seedCalo_centFrac    [self._recoIndex]
                self.isolFrac        = self._tree.tau_seedCalo_isolFrac    [self._recoIndex]
                self.corrFTrk        = self._tree.tau_calcVars_corrFTrk    [self._recoIndex]
                self.trkAvgDist      = self._tree.tau_seedCalo_trkAvgDist  [self._recoIndex]
                self.ipSigLeadTrk    = self._tree.tau_ipSigLeadTrk         [self._recoIndex]
                self.pi0_ptratio     = self._tree.tau_pi0_vistau_pt[self._recoIndex]
                self.pi0_ptratio    *= 1./self._tree.tau_pt[self._recoIndex]
                self.pi0_vistau_m    = self._tree.tau_pi0_vistau_m         [self._recoIndex]
                self.pi0_n           = self._tree.tau_pi0_n                [self._recoIndex]
                self.trFlightPathSig = self._tree.tau_trFlightPathSig      [self._recoIndex]
                self.massTrkSys      = self._tree.tau_massTrkSys           [self._recoIndex]
                self.dRmax           = self._tree.tau_seedCalo_dRmax       [self._recoIndex]
                self.EMRadius        = self._tree.tau_seedCalo_EMRadius    [self._recoIndex]
                self.HadRadius       = self._tree.tau_seedCalo_hadRadius   [self._recoIndex]
                self.EMEnergy        = self._tree.tau_seedCalo_etEMAtEMScale [self._recoIndex]
                self.HadEnergy       = self._tree.tau_seedCalo_etHadAtEMScale [self._recoIndex]
                self.CaloRadius      = (self.EMRadius*self.EMEnergy+self.HadRadius*self.HadEnergy)
                self.stripWidth2     = self._tree.tau_seedCalo_stripWidth2 [self._recoIndex]
                if (self.EMEnergy+self.HadEnergy) !=0:
                    self.CaloRadius     *=  1./(self.EMEnergy+self.HadEnergy)
                else:
                    self.CaloRadius = -99999
                self.numTopoClusters    = self._tree.tau_numTopoClusters[self._recoIndex]  
                self.numEffTopoClusters = self._tree.tau_numEffTopoClusters[self._recoIndex]  
                self.topoInvMass        = self._tree.tau_topoInvMass[self._recoIndex]  
                self.effTopoInvMass     = self._tree.tau_effTopoInvMass[self._recoIndex]  
                self.topoMeanDeltaR     = self._tree.tau_topoMeanDeltaR[self._recoIndex]  
                self.effTopoMeanDeltaR  = self._tree.tau_effTopoMeanDeltaR[self._recoIndex]  

                
	    if 'cellObjects' in include:
		self.cell4Vector, self.nCells, self.cellsamplingID     = self.getCell4Vector()
		self.strip4Vector, self.nStrips, self.stripsamplingID     = self.getStrip4Vector()
	
            if 'recoObjects' in include:
                self.clusters = self.getClusters()
                self.tracks,self.tracks_d0,self.tracks_z0 = self.getTracks()

            if 'wideTracks' in include:
                self.wideTracks,self.wideTracks_d0,self.wideTracks_z0 = self.getWideTracks()
                self.wideTracks_nBLHits,self.wideTracks_nPixHits,self.wideTracks_nSCTHits,self.wideTracks_nTRTHits,self.wideTracks_nHits = self.getWideTracksHits()

            if 'otherTracks' in include:
                self.otherTracks,self.otherTracks_d0,self.otherTracks_z0 = self.getOtherTracks()
                self.otherTracks_nBLHits,self.otherTracks_nPixHits,self.otherTracks_nSCTHits,self.otherTracks_nTRTHits,self.otherTracks_nHits = self.getOtherTracksHits()

            if 'Pi0Finder' in include and 'recoObjects' in include:
                if self.numTrack>0:
                    cb_tau = ClusterBasedTau(self._tree,self._recoIndex)
                    self.clbased_pT = cb_tau.Pt()
                else:
                    self.clbased_pT = -9999
#                 self.EMFClusters = self.getEMFClusters()
#                 clusters = stl.vector( 'TLorentzVector' )()
#                 PSSF  = stl.vector( 'float' )()
#                 EM2FS = stl.vector( 'float' )()
#                 EM3FS = stl.vector( 'float' )()
#                 for clus in self.EMFClusters:
#                     clusters.push_back( clus[0] )
#                     PSSF .push_back( clus[1] )
#                     EM2FS.push_back( clus[2] )
#                     EM3FS.push_back( clus[3] )
#                 self.pi0_vistau_m_alt = -9999
#                 self.pi0_ptratio_alt  = -9999
#                 if clusters.size()>0:
#                     pi0F = ROOT.Pi0Finder(self.tracks,self.clusters,PSSF,EM2FS,EM3FS)
#                     self.pi0_vistau_m_alt  = pi0F.visTauTLV().M()
#                     self.pi0_ptratio_alt = pi0F.visTauTLV().Pt()
#                     self.pi0_ptratio_alt *= 1./self._tree.tau_pt[self._recoIndex]
                                
            if 'CaloDetails' in include:
                self.PSEnergy    = self._tree.tau_calcVars_PSEnergy   [self._recoIndex]
                self.S1Energy    = self._tree.tau_calcVars_S1Energy   [self._recoIndex]
                self.S2Energy    = self._tree.tau_calcVars_S2Energy   [self._recoIndex]
                self.S3Energy    = self._tree.tau_calcVars_S3Energy   [self._recoIndex]
                self.HADEnergy   = self._tree.tau_calcVars_HADEnergy  [self._recoIndex]
                self.CaloEnergy  = self._tree.tau_calcVars_CaloEnergy [self._recoIndex]
        
        return None


    # -----------------------------------------------------------
    def getEventFilterTauIndex(self):
        """Perform a matching between the reco and EF containers """
        reco_eta=self._tree.tau_eta[self._recoIndex]
        reco_phi=self._tree.tau_phi[self._recoIndex]
        DeltaR = 9999.
        Index = -1
        n = self._tree.trig_EF_tau_n
        for i in range(0,n):
            EF_eta=self._tree.trig_EF_tau_eta[i]
            EF_phi=self._tree.trig_EF_tau_phi[i]
            Deta = EF_eta-reco_eta
            Dphi = ROOT.TVector2.Phi_mpi_pi(EF_phi-reco_phi)#--> compute DeltaPhi in -pi,+pi range
            NewDeltaR=ROOT.TMath.Sqrt( pow(Deta,2)+ pow(Dphi,2) )
            if NewDeltaR<DeltaR:
                DeltaR=NewDeltaR
                Index=i
            else:
                continue
        if DeltaR<0.4:
            return Index,DeltaR
        else:
            return -1,9999.


    # -----------------------------------------------------------
    def getL1TauIndex(self):
        """ Perform a matching between the reco and L1_emtau containers """
        reco_eta=self._tree.tau_eta[self._recoIndex]
        reco_phi=self._tree.tau_phi[self._recoIndex]
        DeltaR = 9999.
        Index = -1

        n = self._tree.trig_L1_emtau_n
        for i in range(0,n):
            L1_eta=self._tree.trig_L1_emtau_eta[i]
            L1_phi=self._tree.trig_L1_emtau_phi[i]
            L1_tauclus = self._tree.trig_L1_emtau_tauClus[i]
            Deta = L1_eta-reco_eta
            Dphi = ROOT.TVector2.Phi_mpi_pi(L1_phi-reco_phi)#--> compute DeltaPhi in -pi,+pi range
            NewDeltaR=ROOT.TMath.Sqrt( pow(Deta,2)+ pow(Dphi,2) )
            if NewDeltaR<DeltaR:
                DeltaR=NewDeltaR
                Index=i
            else:
                continue
        if DeltaR<0.4:
            return Index,DeltaR
        else:
            return -1,9999.

    #-----------------------------------------------------------
    def getRecoIndex(self):
        return self._recoIndex
    #-----------------------------------------------------------
    def getTruthIndex(self):
        return self._tree.tau_trueTauAssoc_index[self._recoIndex]

    #-----------------------------------------------------------
    def getTruthIndex_dR(self):
        """Perform a dR matching between the reco and trueTau containers """
        reco_eta=self._tree.tau_eta[self._recoIndex]
        reco_phi=self._tree.tau_phi[self._recoIndex]
        n_trueTaus = self._tree.trueTau_n
        DeltaR = 9999.
        Index = -1
        for true_ind in xrange(0,n_trueTaus):
            true_eta=self._tree.trueTau_eta[true_ind]
            true_phi=self._tree.trueTau_phi[true_ind]
            Deta = true_eta-reco_eta
            Dphi = ROOT.TVector2.Phi_mpi_pi(true_phi-reco_phi)#--> compute DeltaPhi in -pi,+pi range
            NewDeltaR=ROOT.TMath.Sqrt( pow(Deta,2)+ pow(Dphi,2) )
            if NewDeltaR<DeltaR:
                DeltaR=NewDeltaR
                Index=true_ind
            else:
                continue
        if DeltaR<0.2:
            return Index,DeltaR
        else:
            return -1,9999.




    #-----------------------------------------------------------
    def getReco4Vector(self):
        """Get the TLorentzVector for the reco. tau"""
        pt  = self._tree.tau_pt[self._recoIndex]
        eta = self._tree.tau_eta[self._recoIndex]
        phi = self._tree.tau_phi[self._recoIndex]

        vect = ROOT.TLorentzVector()
        vect.SetPtEtaPhiM(pt, eta, phi, 0)

        return vect

    #-----------------------------------------------------------
    def getCell4Vector(self):
	"""Returns the 4-vectors of cells containing energy"""
	cells=[]
	samplingIDs=[]
        n = self._tree.tau_cell_n[self._recoIndex]
	for i in range(0, n):
		if not(self._tree.tau_cell_E[self._recoIndex][i]==0.):
			E=self._tree.tau_cell_E[self._recoIndex][i]
			eta=self._tree.tau_cell_eta_atTJVA[self._recoIndex][i]
			phi=self._tree.tau_cell_phi_atTJVA[self._recoIndex][i]
			samplingID=self._tree.tau_cell_samplingID[self._recoIndex][i]
			pt=E/math.cosh(eta)

	            	vect = ROOT.TLorentzVector()
	            	vect.SetPtEtaPhiM(pt, eta, phi, 0)
			cells.append(vect)
			samplingIDs.append(samplingID)
	return cells, n, samplingIDs

    #-----------------------------------------------------------
    def getStrip4Vector(self):	
        """Returns the 4-vectors of cells in the strip layer containing energy"""
	strips=[]
	samplingIDs=[]
        n = self._tree.tau_cell_n[self._recoIndex]
	for i in range(0, n):
		if (self._tree.tau_cell_samplingID[self._recoIndex][i] == 5 and abs(self._tree.tau_cell_eta_atTJVA[self._recoIndex][i]) > 1.475) or self._tree.tau_cell_samplingID[self._recoIndex][i] == 1:
                    E=self._tree.tau_cell_E[self._recoIndex][i]
                    eta=self._tree.tau_cell_eta_atTJVA[self._recoIndex][i]
                    phi=self._tree.tau_cell_phi_atTJVA[self._recoIndex][i]
                    pt=E/math.cosh(eta)
                    vect = ROOT.TLorentzVector()
                    vect.SetPtEtaPhiM(pt, eta, phi, 0)
                    strips.append(vect)
                    if self._tree.tau_cell_samplingID[self._recoIndex][i] == 5:
                        samplingIDs.append(5)
                    else:
                        samplingIDs.append(1)
	return strips, len(strips), samplingIDs


    #-----------------------------------------------------------
    def getClusters(self, eff=False):
        """Returns the clusters (all or effective)"""
        clusters = stl.vector( 'TLorentzVector' )()

        n = self._tree.tau_cluster_n[self._recoIndex]
        if eff: n = int(math.ceil(self._tree.tau_numEffTopoClusters[self._recoIndex]))
        for i in range(0, n):
            E   = self._tree.tau_cluster_E[self._recoIndex][i]
            eta = self._tree.tau_cluster_eta_atTJVA[self._recoIndex][i]
            phi = self._tree.tau_cluster_phi_atTJVA[self._recoIndex][i]
            pt  = E/math.cosh(eta)

            vect = TLorentzVector()
            vect.SetPtEtaPhiM(pt, eta, phi, 0)
            clusters.push_back(vect)

        return clusters
        

    #-----------------------------------------------------------
    def getTracks(self):
        """Returns the seedCalo tracks"""
        tracks = stl.vector( 'TLorentzVector' )()
        d0     = []
        z0     = []
        n = self._tree.tau_track_n[self._recoIndex]

        for i in range(0, n):
#             pt  = self._tree.tau_track_atTJVA_pt[self._recoIndex][i]
#             eta = self._tree.tau_track_atTJVA_eta[self._recoIndex][i]
#             phi = self._tree.tau_track_atTJVA_phi[self._recoIndex][i]
            pt  = self._tree.tau_track_atTJVA_pt[self._recoIndex][i]
            eta = self._tree.tau_track_atTJVA_eta[self._recoIndex][i]
            phi = self._tree.tau_track_atTJVA_phi[self._recoIndex][i]
            m   = 140
            d0.append(self._tree.tau_track_atTJVA_d0[self._recoIndex][i])
            z0.append(self._tree.tau_track_atTJVA_z0[self._recoIndex][i])
            if pt > 0.0 and eta < 5.0:
                vec = TLorentzVector()
                vec.SetPtEtaPhiM(pt, eta, phi, 0.)
                tracks.push_back(vec)
        
        return tracks,d0,z0


    #-----------------------------------------------------------
    def getWideTracks(self):
        """Returns the seedCalo tracks"""
        tracks = []
        d0     = []
        z0     = []
        n = self._tree.tau_seedCalo_wideTrk_n[self._recoIndex]

        for i in range(0, n):
            pt  = self._tree.tau_seedCalo_wideTrk_pt[self._recoIndex][i]
            eta = self._tree.tau_seedCalo_wideTrk_eta[self._recoIndex][i]
            phi = self._tree.tau_seedCalo_wideTrk_phi[self._recoIndex][i]
            m   = 0

            vect = ROOT.TLorentzVector()
            vect.SetPtEtaPhiM(pt, eta, phi, m)
            tracks.append(vect)

            d0.append( self._tree.tau_seedCalo_wideTrk_d0[self._recoIndex][i] )
            z0.append( self._tree.tau_seedCalo_wideTrk_z0[self._recoIndex][i] )

        return tracks,d0,z0


    #-----------------------------------------------------------
    def getOtherTracks(self):
        """Returns the seedCalo tracks"""
        tracks = []
        d0     = []
        z0     = []
        n = self._tree.tau_otherTrk_n[self._recoIndex]

        for i in range(0, n):
            pt  = self._tree.tau_otherTrk_pt[self._recoIndex][i]
            eta = self._tree.tau_otherTrk_eta[self._recoIndex][i]
            phi = self._tree.tau_otherTrk_phi[self._recoIndex][i]
            m   = 0

            vect = ROOT.TLorentzVector()
            vect.SetPtEtaPhiM(pt, eta, phi, m)
            tracks.append(vect)
            d0.append( self._tree.tau_otherTrk_d0[self._recoIndex][i] )
            z0.append( self._tree.tau_otherTrk_z0[self._recoIndex][i] )

        return tracks,d0,z0


    #-----------------------------------------------------------
    def getWideTracksHits(self):
        """Returns the seedCalo tracks hits"""
        nBLHits  = []
        nPixHits = []
        nSCTHits = []
        nTRTHits = []
        nHits    = []
        n = self._tree.tau_seedCalo_wideTrk_n[self._recoIndex]
        for i in range(0, n):
            nBLHits  .append( self._tree.tau_seedCalo_wideTrk_nBLHits [self._recoIndex][i]  )
            nPixHits .append( self._tree.tau_seedCalo_wideTrk_nPixHits[self._recoIndex][i] )
            nSCTHits .append( self._tree.tau_seedCalo_wideTrk_nSCTHits[self._recoIndex][i] )
            nTRTHits .append( self._tree.tau_seedCalo_wideTrk_nTRTHits[self._recoIndex][i] )
            nHits    .append( self._tree.tau_seedCalo_wideTrk_nHits   [self._recoIndex][i] )
        return nBLHits,nPixHits,nSCTHits,nTRTHits,nHits


    #-----------------------------------------------------------
    def getOtherTracksHits(self):
        """Returns the otherTrk tracks hits"""
        nBLHits  = []
        nPixHits = []
        nSCTHits = []
        nTRTHits = []
        nHits    = []
        n = self._tree.tau_otherTrk_n[self._recoIndex]
        for i in range(0, n):
            nBLHits  .append( self._tree.tau_otherTrk_nBLHits [self._recoIndex][i]  )
            nPixHits .append( self._tree.tau_otherTrk_nPixHits[self._recoIndex][i] )
            nSCTHits .append( self._tree.tau_otherTrk_nSCTHits[self._recoIndex][i] )
            nTRTHits .append( self._tree.tau_otherTrk_nTRTHits[self._recoIndex][i] )
            nHits    .append( self._tree.tau_otherTrk_nHits   [self._recoIndex][i] )
        return nBLHits,nPixHits,nSCTHits,nTRTHits,nHits

    #-----------------------------------------------------------
    def getTruthDecayN(self):
        """Get number of charged and neutral pions"""

        nProngs = self._tree.trueTau_nProng[self._truthIndex]
        nPi0s   = self._tree.trueTau_nPi0[self._truthIndex]

        return nProngs, nPi0s


    #-----------------------------------------------------------
    def getRecoDecayN(self):
        """get numbers of tracks and clusters"""

        nTracks   = self._tree.tau_track_atTJVA_n[self._recoIndex]
        nClusters = self._tree.tau_cluster_n[self._recoIndex]
        nEffClusters = int(math.ceil(self._tree.tau_numEffTopoClusters[self._recoIndex]))
        
        return nTracks, nClusters, nEffClusters



    #-----------------------------------------------------------
    def getEMFClusters(self, eff=False):
        """Returns the clusters (all or effective)"""
        clusters = []
        n = self._tree.tau_cluster_n[self._recoIndex]
        if eff: n = int(math.ceil(self._tree.tau_numEffTopoClusters[self._recoIndex]))

        for i in range(0, n):
            PSSF = self._tree.tau_cluster_PreSamplerStripF[self._recoIndex][i]
            EM2F  = self._tree.tau_cluster_EMLayer2F[self._recoIndex][i]
            EM3F  = self._tree.tau_cluster_EMLayer3F[self._recoIndex][i]

            if PSSF < 0.: PSSF = 0.
            if PSSF > 1.: PSSF = 1.

            if EM2F < 0.: EM2F = 0.
            if EM2F > 1.:EM2F = 1.

            if EM3F < 0.: EM3F = 0.
            if EM3F > 1.: EM3F = 1.

            E   = self._tree.tau_cluster_E[self._recoIndex][i]
            eta = self._tree.tau_cluster_eta_atTJVA[self._recoIndex][i]
            phi = self._tree.tau_cluster_phi_atTJVA[self._recoIndex][i]
            pt  = E/math.cosh(eta)
            vect = ROOT.TLorentzVector()
            vect.SetPtEtaPhiM(pt, eta, phi, 0)

            if pt > 0.0 and eta < 5.0:
                clusters.append([vect, PSSF, EM2F, EM3F])

        return clusters
