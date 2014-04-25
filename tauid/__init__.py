import os
import logging
from variables import *

log = logging.getLogger('tauid')
if not os.environ.get('DEBUG', False):
    log.setLevel(logging.INFO)



VARIABLES = {
    'presel_1': [centfrac,
                 pssfraction,
                 nstrip,
                 emradius,
                 hadradius,
                 emfraction,
                 hadenergy,
                 stripwidth,
                 lead2clustereoverallclusterE,
                 lead3clustereoverallclusterE,
                 numtopoclusters,
                 topoinvmass,
                 topomeandeltar,
                 ],
    'presel_2': [centfrac,
                 pssfraction,
                 nstrip,
                 emradius,
                 hadradius,
                 emfraction,
                 hadenergy,
                 stripwidth,
                 lead2clustereoverallclusterE,
                 lead3clustereoverallclusterE,
                 numefftopoclusters,
                 efftopoinvmass,
                 efftopomeandeltar,
                 ]
    'presel_3var': [centfrac,
                    pssfraction,
                    nstrip,
                    ]
    'presel_5var': [centfrac,
                    pssfraction,
                    nstrip,
                    emradius,
                    hadradius,
                    ]
    
}
