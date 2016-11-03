import os
import numpy as np
from h5py import File

from tauperf import log; log = log['/select-img']
from tauperf.imaging.processing import process_taus

h5_filename = os.path.join(
    os.getenv('DATA_AREA'), 'tauid_ntuples', 'v6', 'output_210files.h5')

h5file = File(h5_filename, mode='r')
rec_1p2n = h5file.get('rec_1p2n')
process_taus(rec_1p2n, nentries=10, do_plot=True, suffix='1p2n')

# rec_1p0n = h5file.get('rec_1p0n')
# process_taus(rec_1p0n, nentries=10, do_plot=True, suffix='1p0n')



# print 'process 1p1n:', len(rec_1p1n)

# s1
# npixels = 192
# n_eta = 24
# r_eta = 0.2
# n_phi = 2
# r_phi = 0.401
# layer = 1

# s2
# npixels = 256
# n_eta = 8
# r_eta = 0.201
# n_phi = 8
# r_phi = 0.401
# layer = 2

# s3
# npixels = 128
# n_eta = 4
# r_eta = 0.201
# n_phi = 8
# r_phi = 0.201
# layer = 3


# bad_img = 0

# for ix in xrange(len(rec_1p1n)):
#     rec = rec_1p1n[ix]
#     indices = np.where(rec['off_cells_samp'] == layer)
    
#     eta_r = rec['off_cells_deta'].take(indices[0])
#     phi_r = rec['off_cells_dphi'].take(indices[0])

#     eta = rec['off_cells_deta_digit'].take(indices[0])
#     phi = rec['off_cells_dphi_digit'].take(indices[0])
#     ene = rec['off_cells_e_norm'].take(indices[0])

#     indices_ =  (np.abs(eta) < n_eta) * (np.abs(phi) < n_phi) *(np.abs(eta_r) < r_eta) * (np.abs(phi_r) < r_phi)
        

#     eta_ = eta_r[indices_]
#     phi_ = phi_r[indices_]
#     ene_ = ene[indices_]
    
#     arr = np.array([eta_, phi_, ene_])
#     rec_new = np.core.records.fromarrays(
#         arr, names='x, y, z', formats='f8, f8, f8')
#     rec_new.sort(order=('x', 'y'))

#     if len(ene_) != npixels:
#         bad_img += 1
#     print ix, len(ene_), len(ene_) < npixels
    
#     if ix < 100 or len(ene_) != npixels:
#         plt.figure()
#         plt.scatter(
#             rec_new['x'], rec_new['y'], c=rec_new['z'], s=40,
#             marker='s', label= 'Number of cells = {0}'.format(len(eta_)))
#         plt.xlim(-0.4, 0.4)
#         plt.ylim(-0.4, 0.4)
#         plt.legend(loc='upper right', fontsize='small', numpoints=1)
#         plt.savefig('plots/grid_1p1n_S{0}_{1}.pdf'.format(layer, ix))
#         plt.clf()
#         plt.close()


# print bad_img, '/', len(rec_1p1n)
