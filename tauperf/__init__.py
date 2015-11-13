import logging
import os
import rootpy

DEFAULT_STUDENT = 'Ztautau'
DEFAULT_TREE = 'tau'

if 'lxplus' in os.getenv('HOSTNAME'):
    UNMERGED_NTUPLE_PATH = '/afs/cern.ch/user/q/qbuat/CrackPotTauID/data/'
    NTUPLE_PATH = UNMERGED_NTUPLE_PATH
else:
    UNMERGED_NTUPLE_PATH = os.path.join(
        # os.getenv('DATA_AREA'), 'crackpotauid_ntuples', 'v3')
        os.getenv('DATA_AREA'), 'tauid_ntuples', 'v2')
    NTUPLE_PATH = UNMERGED_NTUPLE_PATH
    # NTUPLE_PATH = os.path.join(UNMERGED_NTUPLE_PATH, 'merge_weighted')



import ROOT

log = logging.getLogger('tauperf')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)
rootpy.log.setLevel(logging.INFO)

ROOT.gROOT.SetBatch(True)

ATLAS_LABEL = os.getenv('ATLAS_LABEL', 'Internal').strip()


