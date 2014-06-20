class TrueCategories(object):
    def __init_(self, truetau):
        self.truetau = truetau
    
    @property
    def category(self):
        return self.prongpi0cat

    @property
    def pi0cat(self):
        if self.truetau.nPi0==0:
            return ['0n']
        elif self.truetau.nPi0==1:
            return ['1n', 'Xn']
        else:
            return ['2n', 'Xn']

    @property
    def prongcat(self):
        if self.truetau.nProng==1:
            return ['1p']
        else:
            return ['3p']
    @property
    def prongpi0cat(self):
        cats = []
        for pi0cat in self.pi0cat:
            cats.append('{0}_{1}'.format(self.prongcat[0], pi0cat))
        return cats

class TauCategories(object):

    def __init__(self, tau):
        self.tau = tau 

    @property
    def category(self):
        return self.prongcat+self.etacat

    @property
    def prongcat(self):
        if self.tau.numTrack==0:
            return ['0p']
        elif self.tau.numTrack==1:
            return ["1p"]
        elif self.tau.numTrack==2:
            return ["2p","mp"]
        elif self.tau.numTrack==3:
            return ["3p","mp"]
        else:
            return ["mp"]

    @property
    def pi0cat(self):
        if self.tau.pi0BDTPrimary>0.47:
            return ["0n"]
        else:
            return ["Xn"]

    @property
    def prongpi0cat(self):
        if "1p" in self.prong_cat:
            if "0n" in self.pi0_cat:
                return ["1p_0n"]
            else:
                return ["1p_Xn"]
        elif "2p" in self.prong_cat:
            if "0n" in self.pi0_cat:
                return ["2p_0n","mp_0n"]
            else:
                return ["2p_Xn","2p_Xn"]
        elif "3p" in self.prong_cat:
            if "0n" in self.pi0_cat:
                return ["3p_0n","mp_0n"]
            else:
                return ["3p_Xn","mp_Xn"]
        else:
            if "0n" in self.pi0_cat:
                return ["mp_0n"]
            else:
                return ["mp_Xn"]

    @property
    def etacat(self):
        if abs(self.tau.eta)<1.37:
            return ["central"]
        else:
            return ["endcap"]

    @property
    def idcat(self):
        if self.tau.numTrack==1:
            if self.tau.pi0BDTPrimary>0.47:
                return ["all","1p","1p_0n"]
            else:
                return ["all","1p","1p_Xn"]
        else:
            if self.tau.pi0BDTPrimary>0.47:
                return ["all","mp","mp_0n"]
            else:
                return ["all","mp","mp_Xn"]
