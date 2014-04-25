"""
Event filters 
"""
from rootpy.tree.filtering import EventFilter

class Offline_Truth_matching(EventFilter):
    def passes(self, event):
        for tau in event.taus:
            tau.index_matched_truth = tau.trueTauAssoc_index
        return True

class Offline_EF_matching(EventFilter):
    def passes(self, event):
        for tau in event.taus:
            dR_off_EF = 1e10
            for eftau in event.taus_EF:
                if eftau.matches(tau, 0.4):
                    dr_tmp = eftau.dr(tau)
                    if dr_tmp < dR_off_EF:
                        dR_off_EF = dr_tmp
                        tau.index_matched_EF = eftau.index
                        tau.dR_matched_EF = dR_off_EF
        return True

class Offline_L1_matching(EventFilter):
    def passes(self, event):
        for tau in event.taus:
            dR_off_L1 = 1e10
            for l1tau in event.taus_L1:
                if l1tau.matches(tau, 0.4):
                    dr_tmp = l1tau.dr(tau)
                    if dr_tmp < dR_off_L1:
                        dR_off_L1 = dr_tmp
                        tau.index_matched_L1 = l1tau.index
                        tau.dR_matched_L1 = dR_off_L1
        return True

class EF_L2L1_matching(EventFilter):
    def passes(self, event):
        for eftau in event.taus_EF:
            for l2tau in event.taus_L2:
                if l2tau.RoIWord_matches(eftau):
                    eftau.index_matched_L2 = l2tau.index
                    break
            for l1tau in event.taus_L1:
                if l1tau.RoIWord_matches(eftau):
                    eftau.index_matched_L1 = l1tau.index
                    break
        return True

class L2_L1_matching(EventFilter):
    def passes(self, event):
        for l2tau in event.taus_L2:
            for l1tau in event.taus_L1:
                if l1tau.RoIWord_matches(l2tau):
                    l2tau.index_matched_L1 = l1tau.index
                    break
        return True


class GRLFilter(EventFilter):

    def __init__(self, grl, **kwargs):
        super(GRLFilter, self).__init__(**kwargs)
        if isinstance(grl, GRL):
            self.grl = grl
        else:
            self.grl = GRL(grl)
            
    def passes(self, event):
        if not self.grl:
            return False
        return (event.RunNumber, event.lbn) in self.grl
