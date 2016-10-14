import os
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
import skimage.transform as sk

from .. import print_progress
from . import log; log = log[__name__]

def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi

def make_xy_scatter_matrix(x, y, z, scat_pow=2, mean_pow=1):

    cell_values = z
    cell_x = x
    cell_y = y

    etot = np.sum((cell_values>0) * np.power(cell_values, mean_pow))
    if etot == 0:
        log.error('Found a jet with no energy.  DYING!')
        raise ValueError

    x_1  = np.sum((cell_values>0) * np.power(cell_values, mean_pow) * cell_x) / etot
    y_1  = np.sum((cell_values>0) * np.power(cell_values, mean_pow) * cell_y) / etot
    x_2  = np.sum((cell_values>0) * np.power(cell_values, scat_pow) * np.square(cell_x -x_1))
    y_2  = np.sum((cell_values>0) * np.power(cell_values, scat_pow) * np.square(cell_y -y_1))
    xy   = np.sum((cell_values>0) * np.power(cell_values, scat_pow) * (cell_x - x_1) * (cell_y -y_1))

    ScatM = np.array([[x_2, xy], [xy, y_2]])
    MeanV = np.array([x_1, y_1])

    return ScatM, MeanV

def get_principle_axis(mat):

    if mat.shape != (2,2):
        print "ERROR: getPrincipleAxes(theMat), theMat size is not 2x2. DYING!"
        sys.exit(1)

    las, lav = np.linalg.eigh(mat)
    return -1 * lav[::-1], las[::-1]

def interpolate_rbf(x, y, z, function='linear', rotate_pc=True):
    """
    """
#     print 'interpolate ..'
    xi, yi = np.meshgrid(
        np.linspace(x.min(), x.max(), 16),
        np.linspace(y.min(), y.max(), 16))
    from scipy import interpolate
    rbf = interpolate.Rbf(x, y, z, function=function)
    im = rbf(xi, yi)
    if rotate_pc:
        scat_mat, cent = make_xy_scatter_matrix(x, y, z)
        #         scat_mat, cent = make_xy_scatter_matrix(xi, yi, im)
        paxes, pvars = get_principle_axis(scat_mat)
        angle = np.arctan2(paxes[0, 0], paxes[0, 1])
        im = sk.rotate(
            im, np.rad2deg(angle), order=3)
    return im


def tau_image(eta, phi, ene, rotate_pc=True, cal_layer=2):
    """
    return a pixelized image of a tau candidate.
    """
    if cal_layer != 2:
        log.error('layer {0} is not implemented yet'.format(cal_layer))
        raise ValueError

    # the following only applies to the second layer of the ECAL
    # in the barrel
    
    square_ = (np.abs(eta) < 0.2) * (np.abs(phi) < 0.2)
    eta_ = eta[square_]
    phi_ = phi[square_]
    ene_ = ene[square_]

    arr = np.array([eta_, phi_, ene_])
    rec_new = np.core.records.fromarrays(
        arr, names='x, y, z', formats='f8, f8, f8')

    # sort first by x and then by y
    rec_new.sort(order=('x', 'y'))

    # disgard image with wrong pixelization (need to fix!)
    if len(rec_new) != 256:
        return None

    if len(ene) == 0:
        log.warning('pathologic case with 0 cells --> need to figure out why')
        return None

    image = rec_new['z'].reshape((16, 16))
    if rotate_pc:
        scat_mat, cent = make_xy_scatter_matrix(
            rec_new['x'], rec_new['y'], rec_new['z'])
        paxes, pvars = get_principle_axis(scat_mat)
        angle = np.arctan2(paxes[0, 0], paxes[0, 1])
        image = sk.rotate(
            image, np.rad2deg(angle), order=3)

    # return image and selected cells eta, phi, ene
    return image, eta_, phi_, ene_


def process_taus(records, cal_layer=2, do_plot=True, suffix='1p1n'):
    log.info('')
    images = []
    for ir in xrange(len(records)):
        print_progress(ir, len(records), prefix='Progress')

        # protect against pathologic arrays
        try:
            rec = records[ir]
        except:
            rec = None
        
        if rec is None:
            continue

        indices = np.where(rec['off_cells_samp'] == cal_layer)
        if len(indices) == 0:
            continue
        eta_ = rec['off_cells_deta'].take(indices[0])
        phi_ = rec['off_cells_dphi'].take(indices[0])
        ene_ = rec['off_cells_e_norm'].take(indices[0])
        image_tuple = tau_image(eta_, phi_, ene_, cal_layer=cal_layer)

        if image_tuple is not None:
            image, eta, phi, ene = image_tuple
            images.append(image)
            if do_plot and ir < 100:
                # scatter for the selected pixels
                plt.figure()
                plt.scatter(
                    eta, phi, c=ene, marker='s', 
                    label= 'Number of cells = {0}'.format(len(eta)))
                plt.xlim(-0.4, 0.4)
                plt.ylim(-0.4, 0.4)
                plt.xlabel('eta')
                plt.ylabel('phi')
                plt.legend(loc='upper right', fontsize='small', numpoints=1)
                plt.savefig('plots/imaging/selected_grid_%s_%s.pdf' % (suffix, ir))
                plt.clf()
                plt.close()
                # scatter for all the pixels
                plt.figure()
                plt.scatter(
                    eta_, phi_, c=ene_, marker='s', 
                    label= 'Number of cells = {0}'.format(len(eta_)))
                plt.xlim(-0.4, 0.4)
                plt.ylim(-0.4, 0.4)
                plt.xlabel('eta')
                plt.ylabel('phi')
                plt.legend(loc='upper right', fontsize='small', numpoints=1)
                plt.savefig('plots/imaging/grid_%s_%s.pdf' % (suffix, ir))
                plt.clf()
                plt.close()
                # heatmap
                plt.imshow(image, extent=[-0.2, 0.2, -0.2, 0.2])
                plt.plot(
                    rec['true_charged_eta'] - rec['true_eta'], 
                    dphi(rec['true_charged_phi'], rec['true_phi']), 'ro', 
                    label='charge pi, pT = %1.2f GeV' % (rec['true_charged_pt'] / 1000.))
                plt.plot(
                    rec['true_neutral_eta'] - rec['true_eta'], 
                    dphi(rec['true_neutral_phi'], rec['true_phi']), 'g^', 
                    label='neutral pi, pT = %1.2f GeV' % (rec['true_neutral_pt'] / 1000.))
                plt.title('%s heatmap %s' % (suffix, ir))
                plt.xlabel('eta')
                plt.ylabel('phi')
                plt.legend(loc='upper right', fontsize='small', numpoints=1)
                plt.savefig('plots/imaging/heatmap_1p1n_%s.pdf' % ir)
                plt.clf()  
                plt.close()

    # return the images to be stored
    print
    return images
