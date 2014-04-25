from .mixins import Tau, EF_Tau, L2_Tau, L1_Tau, TrueTau

def define_objects(tree):
    tree.define_collection(name='taus', prefix='tau_', size='tau_n', mix=Tau)
    tree.define_collection(name='taus_EF', prefix='trig_EF_tau_', size='trig_EF_tau_n', mix=EF_Tau)
    tree.define_collection(name='taus_L2', prefix='trig_L2_tau_', size='trig_L2_tau_n', mix=L2_Tau)
    tree.define_collection(name='taus_L1', prefix='trig_L1_emtau_', size='trig_L1_emtau_n', mix=L1_Tau)
    tree.define_collection(name='truetaus', prefix='trueTau_', size='trueTau_n', mix=TrueTau)

