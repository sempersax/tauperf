import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
from h5py import File

from tauperf import print_progress
from tauperf import log; log = log[os.path.basename(__file__)]
from tauperf.imaging import tau_image


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
rec_1p0n = h5file.get('rec_1p1n')


print 'process 1p1n:', len(rec_1p1n)
tau_1p1n_images = [] 
for ir in xrange(len(rec_1p1n)):
    print_progress(ir, len(rec_1p1n), prefix='Progress')
    try: 
        rec = rec_1p1n[ir]
    except:
        rec = None

    indices = np.where(rec['off_cells_samp'] == 2)
    if len(indices) == 0:
        continue
    eta_ = rec['off_cells_deta'].take(indices[0])
    phi_ = rec['off_cells_dphi'].take(indices[0])
    ene_ = rec['off_cells_e_norm'].take(indices[0])
    image_tuple = tau_image(eta_, phi_, ene_)

    if image_tuple is not None:
        image, eta, phi, ene = image_tuple
        tau_1p1n_images.append(image)
        if ir < 100:
            plt.figure()
            plt.scatter(eta, phi, c=ene, marker='s', label= 'Number of cells = {0}'.format(len(eta)))
            plt.xlim(-0.4, 0.4)
            plt.ylim(-0.4, 0.4)
            plt.legend(loc='upper right', fontsize='small', numpoints=1)
            plt.savefig('plots/grid_1p1n_%s.pdf' % ir)
            plt.clf()
            plt.close()


np.save(os.path.join(
        data_dir, 'images_1p1n_dr0.2.npy'), tau_1p1n_images)

for ir, image in enumerate(tau_1p1n_images):
    # only make 100 displays
    if ir > 100:
        break

    if image is None:
        continue
    r = rec_1p1n[ir]
    plt.imshow(image,
        extent=[-0.2, 0.2, -0.2, 0.2])
    plt.plot(
        r['true_charged_eta'] - r['true_eta'], 
        dphi(r['true_charged_phi'], r['true_phi']), 'ro', 
        label='charge pi, pT = %1.2f GeV' % (r['true_charged_pt'] / 1000.))
    plt.plot(
        r['true_neutral_eta'] - r['true_eta'], 
        dphi(r['true_neutral_phi'], r['true_phi']), 'g^', 
        label='neutral pi, pT = %1.2f GeV' % (r['true_neutral_pt'] / 1000.))
    plt.title('1p1n heatmap %s' % ir)
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.legend(loc='upper right', fontsize='small', numpoints=1)
    print 'save fig', ir
    plt.savefig('plots/heatmap_1p1n_%s.pdf' % ir)
    plt.clf()  


print 'process 1p0n', len(rec_1p0n)
tau_1p0n_images = []
for ir in xrange(len(rec_1p0n)):
    print_progress(ir, len(rec_1p0n))

    try:
        rec = rec_1p0n[ir]
    except:
        rec = None

    if rec is None:
        continue;

    indices = np.where(rec['off_cells_samp'] == 2)
    if len(indices) == 0:
        continue
    eta_ = rec['off_cells_deta'].take(indices[0])
    phi_ = rec['off_cells_dphi'].take(indices[0])
    ene_ = rec['off_cells_e_norm'].take(indices[0])
    image_tuple = tau_image(eta_, phi_, ene_)

    if image_tuple is not None:
        image, eta, phi, ene = image_tuple
        tau_1p0n_images.append(image)
        if ir < 100:
            plt.figure()
            plt.scatter(eta, phi, c=ene, marker='s', label= 'Number of cells = {0}'.format(len(eta)))
            plt.legend(loc='upper right', fontsize='small', numpoints=1)
            plt.savefig('plots/grid_1p0n_%s.pdf' % ir)
            plt.clf()  
            plt.close()

np.save(os.path.join(
        data_dir, 'images_1p0n_dr0.2.npy'), tau_1p0n_images)

for ir, image in enumerate(tau_1p0n_images):
    # only make 100 displays
    if ir > 100:
        break

    if image is None:
        continue
    r = rec_1p0n[ir]

    plt.imshow(image,
        extent=[-0.2, 0.2, -0.2, 0.2])
    plt.plot(
        r['true_charged_eta'] - r['true_eta'], 
        dphi(r['true_charged_phi'], r['true_phi']), 'ro', 
        label='charge pi, pT = %1.2f GeV' % (r['true_charged_pt'] / 1000.))
    plt.title('1p0n heatmap %s' % ir)
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.legend(loc='upper right', fontsize='small', numpoints=1)
    print 'save fig', ir
    plt.savefig('plots/heatmap_1p0n_%s.pdf' % ir)
    plt.clf()  

