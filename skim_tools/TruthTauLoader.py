#--> TruthTauLoader class
import ROOT

class TruthTauLoader:
    """A class to dump tau-related truth info, and do truth-matching."""

    #-----------------------------------------------------------
    def __init__(self, tree, truth_index,include):
        """Constructor"""

        self._tree           = tree
        self._truthIndex     = truth_index
        self._recoIndex      = -1
        self.hasReco         = False
        self.author          = -1

        if 'basic' in include:
            self.truthVis4Vector            = self.getTruthVis4Vector()
            self.truth4Vector               = self.getTruth4Vector()
            self.nProngs, self.nPi0s        = self.getTruthDecayN()
            self.invis4Vector               = self.truth4Vector - self.truthVis4Vector
        if 'decays' in include:
            self.truthPiCh, self.truthPi0s  = self.getTruthDecay()

        # ---> Truth-Reco matching 
        self._recoIndex = tree.trueTau_tauAssoc_index[self._truthIndex]
        if self._recoIndex > -1:
            self.hasReco = True
            self.author  = self._tree.tau_author[self._recoIndex]  
          
        return None

    #-----------------------------------------------------------
    def getRecoIndex(self):
        return self._recoIndex

    #-----------------------------------------------------------
    def getTruth4Vector(self):
        """Get the TLorentzVector for the truth tau (including invisible)"""
	pt  = self._tree.trueTau_pt[self._truthIndex]        
	eta = self._tree.trueTau_eta[self._truthIndex]
        phi = self._tree.trueTau_phi[self._truthIndex]
        m   = self._tree.trueTau_m[self._truthIndex]
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, m)
        return vector

    #-----------------------------------------------------------
    def getTruthVis4Vector(self):
        """Get the TLorentzVector for the truth tau (including invisible)"""
        pt  = self._tree.trueTau_vis_Et[self._truthIndex]
        eta = self._tree.trueTau_vis_eta[self._truthIndex]
        phi = self._tree.trueTau_vis_phi[self._truthIndex]
        m   = self._tree.trueTau_vis_m[self._truthIndex]
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, m)
        return vector


    #-----------------------------------------------------------
    def getTruthDecayN(self):
        """Get number of charged and neutral pions"""

        nProngs = self._tree.trueTau_nProng[self._truthIndex]
        nPi0s   = self._tree.trueTau_nPi0[self._truthIndex]

        return nProngs, nPi0s


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

                        PiCh = ROOT.TLorentzVector()
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

                            ph = ROOT.TLorentzVector()
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

