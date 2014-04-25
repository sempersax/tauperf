from rootpy import stl
import ROOT


class ClusterBasedTau(ROOT.TLorentzVector):
    """ Cluster-based tau calculation inspired from
    atlasusr/limbach/Standalone/SubstructureComparison/trunk/HelperClasses/TauGetter_ClusterBased.h
    """
    def __init__(self, tree,reco_index):
        ROOT.TLorentzVector.__init__(self)
        self._tree = tree
        self._recoIndex = reco_index

        tau_numTrack = self._tree.tau_numTrack[self._recoIndex]
        tau_nPi0s = self._tree.tau_pi0_n[self._recoIndex]
        if tau_nPi0s == 0:
            if self._tree.tau_track_n[self._recoIndex]>0:
                sumTrk = ROOT.TLorentzVector()
                for trk_ind in xrange(0,self._tree.tau_track_n[self._recoIndex]):
                    curTrk = ROOT.TLorentzVector()
                    curTrk.SetPtEtaPhiM( self._tree.tau_track_atTJVA_pt[self._recoIndex][trk_ind],
                                         self._tree.tau_track_atTJVA_eta[self._recoIndex][trk_ind],
                                         self._tree.tau_track_atTJVA_phi[self._recoIndex][trk_ind],
                                         139.8 )
                    sumTrk += curTrk
                self.SetPtEtaPhiM( sumTrk.Pt(),sumTrk.Eta(),sumTrk.Phi(),sumTrk.M() )
            else:
                self.SetPtEtaPhiM ( self._tree.tau_pt[self._recoIndex],
                                    self._tree.tau_eta[self._recoIndex],
                                    self._tree.tau_phi[self._recoIndex],
                                    self._tree.tau_m[self._recoIndex] )
        elif tau_nPi0s == 1 or tau_nPi0s==2:
            if self._tree.tau_pi0_vistau_pt[self._recoIndex]==0:
                self.SetPtEtaPhiM ( self._tree.tau_pt[self._recoIndex],
                                    self._tree.tau_eta[self._recoIndex],
                                    self._tree.tau_phi[self._recoIndex],
                                    self._tree.tau_m[self._recoIndex] )
            else:
                self.SetPtEtaPhiM ( self._tree.tau_pi0_vistau_pt[self._recoIndex],
                                    self._tree.tau_pi0_vistau_eta[self._recoIndex],
                                    self._tree.tau_pi0_vistau_phi[self._recoIndex],
                                    self._tree.tau_pi0_vistau_m[self._recoIndex] )

        else:
            self.SetPtEtaPhiM ( self._tree.tau_vistau_pt[self._recoIndex],
                                self._tree.tau_vistau_eta[self._recoIndex],
                                self._tree.tau_vistau_phi[self._recoIndex],
                                self._tree.tau_vistau_m[self._recoIndex] )
