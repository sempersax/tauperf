import ROOT
#-----------------------------------------------------------
class Category:
    """A class to categorize EF taus"""
    def __init__(self,tauCell):
        self._tree = tauCell
        self.prong_cat = self.getProngCat()
        self.pi0_cat   = self.getPi0Cat()
        self.prongpi0_cat = self.getProngPi0Cat()
        self.eta_cat   = self.getEtaCat()
        self.mu_cat    = self.getMuCat()
        self.categories = self.getCategories()
        self.ID_cat = self.getIDCat()
        return None

    def getCategories(self):
        return self.prong_cat+self.prongpi0_cat+self.eta_cat+self.mu_cat
    def getProngCat(self):
        if self._tree.EF_nTracks==1: return ["1p"]
        if self._tree.EF_nTracks==2: return ["2p","mp"]
        if self._tree.EF_nTracks==3: return ["3p","mp"]
        else: return ["mp"]
    def getPi0Cat(self):
        if self._tree.off_pi0BDTPrimary>0.47: return ["0n"]
        else: return ["Xn"]
    def getProngPi0Cat(self):
        if "1p" in self.prong_cat:
            if "0n" in self.pi0_cat: return ["1p_0n"]
            else: return ["1p_Xn"]
        elif "2p" in self.prong_cat:
            if "0n" in self.pi0_cat: return ["2p_0n","mp_0n"]
            else: return ["2p_Xn","2p_Xn"]
        elif "3p" in self.prong_cat:
            if "0n" in self.pi0_cat: return ["3p_0n","mp_0n"]
            else: return ["3p_Xn","mp_Xn"]
        else:
            if "0n" in self.pi0_cat: return ["mp_0n"]
            else: return ["mp_Xn"]

            

    def getEtaCat(self):
        if abs(self._tree.EF_eta)<1.37: return ["central"]
        else: return ["endcap"]
    def getMuCat(self):
        if self._tree.mu<20: return ["low_mu"]
        elif self._tree.mu<40: return ["medium_mu"]
        elif self._tree.mu<60: return ["high_mu"]
        else : return ["crazy_mu"]
    def getIDCat(self):
        if self._tree.EF_nTracks==1:
            if self._tree.off_pi0BDTPrimary>0.47: return ["all","1p","1p_0n"]
            else:                             return ["all","1p","1p_Xn"]        
        else:
            if self._tree.off_pi0BDTPrimary>0.47: return ["all","mp","mp_0n"]
            else:                             return ["all","mp","mp_Xn"]        
            
