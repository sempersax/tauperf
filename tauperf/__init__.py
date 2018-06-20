import logging
import os

DEFAULT_STUDENT = 'Ztautau'
DEFAULT_TREE = 'tau'

H5_FILE = 'output_100files.h5'

if os.getenv('HOSTNAME') is not None and 'lxplus' in os.getenv('HOSTNAME'):
    UNMERGED_NTUPLE_PATH = '/afs/cern.ch/user/q/qbuat/CrackPotTauID/data/'
    NTUPLE_PATH = UNMERGED_NTUPLE_PATH
else:
    UNMERGED_NTUPLE_PATH = os.path.join(
        os.getenv('DATA_AREA'), 'tauid_ntuples', 'v5')
    NTUPLE_PATH = UNMERGED_NTUPLE_PATH




#import rootpy
# log = logging.getLogger('tauperf')
# from rootpy import log; log = log["/tauperf"]
# if not os.environ.get("DEBUG", False):
#     log.setLevel(logging.INFO)
# rootpy.log.setLevel(logging.INFO)

# import ROOT
# ROOT.gROOT.SetBatch(True)

ATLAS_LABEL = os.getenv('ATLAS_LABEL', 'Internal').strip()


import sys

# Print iterations progress
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """

    bar_letter = os.environ.get('USER').strip()[0].capitalize()
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = bar_letter * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
