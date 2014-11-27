#TauLoader class
from ROOT import *
import math

gROOT.LoadMacro("load_stl_float.h+")
gROOT.LoadMacro("load_stl_int.h+")
print 'libraries loaded'

class TauLoader:
    """A class to dump tau-related info for reconstructed taus."""

    #-----------------------------------------------------------
    def __init__(self, tree, index, include,isData):
        """Constructor"""

        self._tree           = tree
        self._recoIndex      = -1
        self._isData         = isData
        self.hasReco         = False
        self.author          = -1
        self.reco4Vector     = 0
	self.cell4Vector     = []
	self.strip4Vector    = []
        self.truthVis4Vector = 0
        self.truth4Vector    = 0
        self.invis4Vector    = 0
        self.clusters        = []
        self.nClusters       = 0
        self.nEffClusters    = 0
        self.EMFClusters     = []
        self.tracks          = []
        self.tracks_d0       = []
        self.tracks_z0       = []
        self.numTrack        = 0
        self.nTracks         = 0
        self.wideTracks      = []
        self.notherTrk       = 0
        self.otherTracks     = []
        self.nProngs         = 0
        self.nPi0s           = 0
        self.truthPiCh       = []
        self.truthPi0s       = []

        # Variables 
        self.nStrip             = 0
        self.etOverPt           = 0
        self.mEflow             = 0
        self.nEffStripCells     = 0
        self.effMass            = 0
        self.ChPiEMEOverCaloEME = 0
        self.PSSFraction        = 0
        self.EMPOverTrkSysP     = 0
        self.EMRadius           = 0
        self.trkAvgDist         = 0
        self.BDTJetScore        = 0
        self.BDTEleScore        = 0
        self.pi0BDTPrimary      = 0
        self.pi0BDTSecondary    = 0
        self.cellsamplingID     = []
	self.nCells   		= 0
	self.nStrips		= 0
	self.stripsampplingID	= []

        self.corrCentFrac      = []
        self.centFrac          = []
        self.corrFTrk          = []
        self.trkAvgDist        = []
        self.ipSigLeadTrk      = []
        self.nWideTrk          = []
        # self.pi0_ptratio       = []
        self.pi0_vistau_m      = []
        self.pi0_n             = []
        self.trFlightPathSig   = []
        self.massTrkSys        = []

        # --> Perform a truth matching if running on signal MC
        recoIndex = -1
        if self._isData == False:
            recoIndex = tree.trueTau_tauAssoc_index[index]
        else: recoIndex=index

        if recoIndex > -1:
            self._recoIndex  = recoIndex
            self.hasReco     = True
            self.author      = self._tree.tau_author[self._recoIndex]  
          
            if 'basic' in include:
                self.reco4Vector = self.getReco4Vector()
                self.nTracks, self.nClusters, self.nEffClusters    = self.getRecoDecayN()

	    if 'cellObjects' in include:
		self.cell4Vector, self.nCells, self.cellsamplingID     = self.getCell4Vector()
		self.strip4Vector, self.nStrips, self.stripsamplingID     = self.getStrip4Vector()
	
            if 'recoObjects' in include:
                self.clusters                   = self.getClusters()
                self.tracks,self.tracks_d0,self.tracks_z0 = self.getTracks()

            if 'wideTracks' in include:
                self.wideTracks,self.wideTracks_d0,self.wideTracks_z0 = self.getWideTracks()
                self.wideTracks_nBLHits,self.wideTracks_nPixHits,self.wideTracks_nSCTHits,self.wideTracks_nTRTHits,self.wideTracks_nHits = self.getWideTracksHits()

            if 'otherTracks' in include:
                self.otherTracks,self.otherTracks_d0,self.otherTracks_z0 = self.getOtherTracks()
                self.otherTracks_nBLHits,self.otherTracks_nPixHits,self.otherTracks_nSCTHits,self.otherTracks_nTRTHits,self.otherTracks_nHits = self.getOtherTracksHits()

            if 'truth' in include:
                self.nProngs, self.nPi0s        = self.getTruthDecayN()
                self.truthVis4Vector            = self.getTruthVis4Vector()
                self.truth4Vector               = self.getTruth4Vector()
                self.invis4Vector               = self.truth4Vector - self.truthVis4Vector
                self.truthPiCh, self.truthPi0s  = self.getTruthDecay()

            if 'EMFClusters' in include:
                self.EMFClusters                   = self.getEMFClusters()

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
                self.corrFTrk        = self._tree.tau_calcVars_corrFTrk    [self._recoIndex]
                self.trkAvgDist      = self._tree.tau_seedCalo_trkAvgDist  [self._recoIndex]
                self.ipSigLeadTrk    = self._tree.tau_ipSigLeadTrk         [self._recoIndex]
                # self.pi0_ptratio     = self._tree.tau_pi0_ptratio[self._recoIndex]
                self.pi0_vistau_m    = self._tree.tau_pi0_vistau_m         [self._recoIndex]
                self.pi0_n           = self._tree.tau_pi0_n                [self._recoIndex]
                self.trFlightPathSig = self._tree.tau_trFlightPathSig      [self._recoIndex]
                self.massTrkSys      = self._tree.tau_massTrkSys           [self._recoIndex]
                self.numTrack        = self._tree.tau_numTrack             [self._recoIndex]
                self.nWideTrk        = self._tree.tau_seedCalo_nWideTrk    [self._recoIndex]
                self.notherTrk       = self._tree.tau_otherTrk_n           [self._recoIndex]

            if 'CaloDetails' in include:
                self.PSEnergy   = self._tree.tau_calcVars_PSEnergy   [self._recoIndex]
                self.S1Energy   = self._tree.tau_calcVars_S1Energy   [self._recoIndex]
                self.S2Energy   = self._tree.tau_calcVars_S2Energy   [self._recoIndex]
                self.S3Energy   = self._tree.tau_calcVars_S3Energy   [self._recoIndex]
                self.HADEnergy  = self._tree.tau_calcVars_HADEnergy  [self._recoIndex]
                self.CaloEnergy = self._tree.tau_calcVars_CaloEnergy [self._recoIndex]
        
        return None


    #-----------------------------------------------------------
    def getRecoIndex(self):
        return self._recoIndex

    #-----------------------------------------------------------
    def getReco4Vector(self):
        """Get the TLorentzVector for the reco. tau"""
        pt  = self._tree.tau_pt[self._recoIndex]
        eta = self._tree.tau_eta[self._recoIndex]
        phi = self._tree.tau_phi[self._recoIndex]

        vector = TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, 0)

        return vector


    #-----------------------------------------------------------
    def getTruth4Vector(self):
        """Get the TLorentzVector for the truth tau (including invisible)"""
	pt  = self._tree.trueTau_pt[self._truthIndex]        
	eta = self._tree.trueTau_eta[self._truthIndex]
        phi = self._tree.trueTau_phi[self._truthIndex]
        m   = self._tree.trueTau_m[self._truthIndex]

        vector = TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, m)

        return vector


    #-----------------------------------------------------------
    def getTruthVis4Vector(self):
        """Get the TLorentzVector for the truth tau (including invisible)"""
        pt  = self._tree.trueTau_vis_Et[self._truthIndex]
        eta = self._tree.trueTau_vis_eta[self._truthIndex]
        phi = self._tree.trueTau_vis_phi[self._truthIndex]
        m   = self._tree.trueTau_vis_m[self._truthIndex]

        vector = TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, m)

        return vector


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

	            	vector = TLorentzVector()
	            	vector.SetPtEtaPhiM(pt, eta, phi, 0)
			cells.append(vector)
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
                    vector = TLorentzVector()
                    vector.SetPtEtaPhiM(pt, eta, phi, 0)
                    strips.append(vector)
                    if self._tree.tau_cell_samplingID[self._recoIndex][i] == 5:
                        samplingIDs.append(5)
                    else:
                        samplingIDs.append(1)
	return strips, len(strips), samplingIDs


    #-----------------------------------------------------------
    def getClusters(self, eff=False):
        """Returns the clusters (all or effective)"""
        clusters = []
        n = self._tree.tau_cluster_n[self._recoIndex]
        if eff: n = int(math.ceil(self._tree.tau_numEffTopoClusters[self._recoIndex]))

        for i in range(0, n):
            E   = self._tree.tau_cluster_E[self._recoIndex][i]
            eta = self._tree.tau_cluster_eta_atTJVA[self._recoIndex][i]
            phi = self._tree.tau_cluster_phi_atTJVA[self._recoIndex][i]
            pt  = E/math.cosh(eta)

            vector = TLorentzVector()
            vector.SetPtEtaPhiM(pt, eta, phi, 0)
            clusters.append(vector)

        return clusters
        

    #-----------------------------------------------------------
    def getTracks(self):
        """Returns the seedCalo tracks"""
        tracks = []
        d0     = []
        z0     = []
        n = self._tree.tau_track_n[self._recoIndex]

        for i in range(0, n):
            pt  = self._tree.tau_track_atTJVA_pt[self._recoIndex][i]
            eta = self._tree.tau_track_atTJVA_eta[self._recoIndex][i]
            phi = self._tree.tau_track_atTJVA_phi[self._recoIndex][i]
            m   = 140
            d0.append(self._tree.tau_track_atTJVA_d0[self._recoIndex][i])
            z0.append(self._tree.tau_track_atTJVA_z0[self._recoIndex][i])
            vector = TLorentzVector()
            vector.SetPtEtaPhiM(pt, eta, phi, m)

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

            vector = TLorentzVector()
            vector.SetPtEtaPhiM(pt, eta, phi, m)
            tracks.append(vector)

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

            vector = TLorentzVector()
            vector.SetPtEtaPhiM(pt, eta, phi, m)
            tracks.append(vector)
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
    def getTruthDecay(self):
        """Get 4-vectors for the true charged pions"""
        productsIndex = self._tree.trueTau_truthAssoc_index[self._truthIndex]

        ChargedPions = []
        NeutralPions = []

        for i in range(0, len(productsIndex)):
            k = productsIndex[i]
            PDGID  = self._tree.mc_pdgId[k]
            status = self._tree.mc_status[k]

            if abs(PDGID) == 15 and status == 2:
                indices = self.getDaughters(k)

                # Loop over tau final daughters to find charged pions and pi0s
                for j in indices:
                    pdgId  = self._tree.mc_pdgId[j]

                    if abs(pdgId) == 211 or abs(pdgId) == 321:
                        pt  = self._tree.mc_pt[j]
                        eta = self._tree.mc_eta[j]
                        phi = self._tree.mc_phi[j]
                        m   = self._tree.mc_m[j]

                        PiCh = TLorentzVector()
                        PiCh.SetPtEtaPhiM(pt, eta, phi, m)
                        ChargedPions.append(PiCh)

                    if abs(pdgId) == 111:
                        photons = self.getPhotons(j)
                        photonVectors = []
                        for p in photons:
                            pt  = self._tree.mc_pt[p]
                            eta = self._tree.mc_eta[p]
                            phi = self._tree.mc_phi[p]
                            m   = self._tree.mc_m[p]

                            ph = TLorentzVector()
                            ph.SetPtEtaPhiM(pt, eta, phi, m)
                            photonVectors.append(ph)

                        NeutralPions.append(photonVectors)
        
        return ChargedPions, NeutralPions


    #-----------------------------------------------------------
    def getDaughters(self, parentIndex):
        """To be used recursively until all particles are in final state"""
        #Get daughters:
        daughters = self._tree.mc_child_index[parentIndex]

        daughtersData = []

        for i in range(0, len(daughters)):
            k = daughters[i]
            status = self._tree.mc_status[k]
            pdgId  = self._tree.mc_pdgId[k]

            if status == 1:
                daughtersData.append(k)

            if status == 2:
                if pdgId == 111:
                    daughtersData.append(k)
                else:
                    newDaughtersData = self.getDaughters(k)
                    daughtersData.extend(newDaughtersData)

        return daughtersData
            

    #-----------------------------------------------------------
    def getPhotons(self, parentIndex):
        """Get the photon daughters of a pi0"""
        daughters = self._tree.mc_child_index[parentIndex]
        return list(daughters)

    #-----------------------------------------------------------
    def getEMFClusters(self, eff=False):
        """Returns the clusters (all or effective)"""
        clusters = []
        n = self._tree.tau_cluster_n[self._recoIndex]
        if eff: n = int(math.ceil(self._tree.tau_numEffTopoClusters[self._recoIndex]))

        for i in range(0, n):
            E   = self._tree.tau_cluster_E[self._recoIndex][i]
            eta = self._tree.tau_cluster_eta_atTJVA[self._recoIndex][i]
            phi = self._tree.tau_cluster_phi_atTJVA[self._recoIndex][i]
            pt  = E/math.cosh(eta)
            EMFPS = self._tree.tau_cluster_PreSamplerStripF[self._recoIndex][i]
            EMF2  = self._tree.tau_cluster_EMLayer2F[self._recoIndex][i]
            EMF3  = self._tree.tau_cluster_EMLayer3F[self._recoIndex][i]

            vector = TLorentzVector()
            vector.SetPtEtaPhiM(pt, eta, phi, 0)
            clusters.append([vector, EMFPS, EMF2, EMF3])

        return clusters
