import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
from h5py import File

from tauperf import print_progress
from tauperf import log; log = log[os.path.basename(__file__)]
from tauperf.imaging import tau_image, process_taus


def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi


data_dir = 'data_test'



h5_filename = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v5', 'output_100files.h5')
log.info('open h5 file {0}'.format(h5_filename))

h5file = File(h5_filename, mode='r')
rec_1p1n = h5file.get('rec_1p1n')
rec_1p0n = h5file.get('rec_1p0n')


log.info('process 1p1n: {0}'.format(len(rec_1p1n)))
log.info('')
tau_1p1n_images = process_taus(rec_1p1n)
log.info('')


np.save(os.path.join(
        data_dir, 'images_1p1n_dr0.2.npy'), tau_1p1n_images)

log.info('process 1p0n: {0}'.format(len(rec_1p0n)))
log.info('')
tau_1p0n_images = process_taus(rec_1p0n, suffix='1p0n')
log.info('')

np.save(os.path.join(
        data_dir, 'images_1p0n_dr0.2.npy'), tau_1p0n_images)

