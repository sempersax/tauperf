DEFAULT_STUDENT = 'Ztautau'
DEFAULT_TREE = 'tau'
NTUPLE_PATH = '/Users/quentin/Desktop/DATA_CLUSTER/'

import logging
import os
import rootpy

import ROOT

log = logging.getLogger('tauperf')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)
rootpy.log.setLevel(logging.INFO)

ROOT.gROOT.SetBatch(True)

ATLAS_LABEL = os.getenv('ATLAS_LABEL', 'Internal').strip()


