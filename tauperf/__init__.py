import logging
import os
import rootpy

DEFAULT_STUDENT = 'Ztautau'
DEFAULT_TREE = 'tau'
NTUPLE_PATH = os.path.join(
    os.getenv('DATA_AREA'), 'tauperf_skims_xaods', 'v7')



import ROOT

log = logging.getLogger('tauperf')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)
rootpy.log.setLevel(logging.INFO)

ROOT.gROOT.SetBatch(True)

ATLAS_LABEL = os.getenv('ATLAS_LABEL', 'Internal').strip()


