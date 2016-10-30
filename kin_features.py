#!/usr/bin/env python

import os
import numpy as np
from h5py import File

from tauperf import log; log = log['/kin-features']
from tauperf.imaging import process_taus



h5_filename = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v6', 'output_210files.h5')
log.info('open h5 file {0}'.format(h5_filename))

h5file = File(h5_filename, mode='r')
rec_1p0n = h5file.get('rec_1p0n')
rec_1p1n = h5file.get('rec_1p1n')


log.info('process 1p0n: {0}'.format(len(rec_1p0n)))
log.info('')
tau_1p0n_images, kin_1p0n = process_taus(rec_1p0n)
log.info('')

# log.info('process 1p1n: {0}'.format(len(rec_1p1n)))
# log.info('')
# tau_1p1n_images, kin_1p1n = process_taus(rec_1p1n)
# log.info('')
