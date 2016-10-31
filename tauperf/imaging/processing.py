import os
import uuid
import numpy as np
from numpy.lib import recfunctions
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
import skimage.transform as sk
from sklearn import model_selection

from .. import print_progress
from . import log; log = log[__name__]
from .plotting import plot_image

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


def tau_image(
    irec, rec, 
    rotate_pc=False, 
    cal_layer=2, 
    do_plot=False,
    suffix='1p1n'):
    """
    return a pixelized image of a tau candidate.
    """


    # retrieve eta, phi and energy arrays in a given layers
    indices = np.where((rec['off_cells_samp'] == cal_layer) * (rec['off_ntracks'] == 1))
#     indices = np.where(rec['off_cells_samp'] == cal_layer)
    if len(indices) == 0:
        return None
    eta_r = rec['off_cells_deta'].take(indices[0])
    phi_r = rec['off_cells_dphi'].take(indices[0])
    eta = rec['off_cells_deta_digit'].take(indices[0])
    phi = rec['off_cells_dphi_digit'].take(indices[0])
    ene = rec['off_cells_e_norm'].take(indices[0])

    kin_arr = np.array([
            rec['off_pt'], rec['off_eta'], rec['averageintpercrossing']])
    kin_rec = np.core.records.fromarrays(
        kin_arr, names='pt, eta, mu', formats='f8, f8, f8')
    

    # define the square used to collect cells for the image
    if cal_layer == 2:
        n_eta = 8
        n_phi = 8
        r_eta = 0.201
        r_phi = 0.201
    elif cal_layer == 1:
        n_eta = 64
        n_phi = 2
        r_eta = 0.401
        r_phi = 0.401
    elif cal_layer == 3:
        n_eta = 4
        n_phi = 8
        r_eta = 0.201
        r_phi = 0.201
    else:
        log.error('layer {0} is not implemented yet'.format(cal_layer))
        raise ValueError
        

    # collect cells in a square (or rectangle)
    square_ = (np.abs(eta) < n_eta) * (np.abs(phi) < n_phi) *(np.abs(eta_r) < r_eta) * (np.abs(phi_r) < r_phi)
    eta_r_ = eta_r[square_]
    phi_r_ = phi_r[square_]
    eta_ = eta[square_]
    phi_ = phi[square_]
    ene_ = ene[square_]

    if do_plot is True:
        plot_image(
            rec, eta_r, phi_r, ene, irec, cal_layer, suffix)
        plot_image(
            rec, eta_r_, phi_r_, ene_, irec, cal_layer, 'selected_' + suffix)


    # create the raw image
    arr = np.array([eta_, phi_, ene_])
    rec_new = np.core.records.fromarrays(
        arr, names='x, y, z', formats='f8, f8, f8')
    # order the pixels by sorting first by x and then by y
    rec_new.sort(order=('x', 'y'))

    if len(ene) == 0:
#         log.warning('pathologic case with 0 cells --> need to figure out why')
        return None

    # disgard image with wrong pixelization (need to fix!)
    if cal_layer == 1 and len(rec_new) != 512:
        return None

    if cal_layer == 2 and len(rec_new) != 256:
        return None

    if cal_layer == 3 and len(rec_new) != 128:
        return None

    # reshaping
    if cal_layer == 1:
        image = rec_new['z'].reshape((4, 128))

    # reshaping
    elif cal_layer == 2:
        image = rec_new['z'].reshape((16, 16))
    elif cal_layer == 3:
        image = rec_new['z'].reshape((16, 8))
    else:
        log.error('layer {0} is not implemented yet'.format(cal_layer))
        raise ValueError

    # rotating
    if rotate_pc:
        scat_mat, cent = make_xy_scatter_matrix(
            rec_new['x'], rec_new['y'], rec_new['z'])
        paxes, pvars = get_principle_axis(scat_mat)
        angle = np.arctan2(paxes[0, 0], paxes[0, 1])
        image = sk.rotate(
            image, np.rad2deg(angle), order=3)

    # return image and selected cells eta, phi, ene
    return image, kin_rec, eta_, phi_, ene_


def process_taus(records, nentries=None, cal_layer=None, do_plot=False, suffix='1p1n'):
    log.info('')
    images = []
    for ir in xrange(len(records)):
        if nentries is None:
            print_progress(ir, len(records), prefix='Progress')
        else:
            print_progress(ir, nentries, prefix='Progress')
            
        # kill the loop after number of specified entries
        if nentries is not None and ir == nentries:
            break

        # protect against pathologic arrays
        try:
            rec = records[ir]
        except:
            rec = None
        
        if rec is None:
            continue

        if cal_layer is None:
            image_tuple_s1 = tau_image(ir, rec, cal_layer=1, do_plot=do_plot, suffix=suffix)
            image_tuple_s2 = tau_image(ir, rec, cal_layer=2, do_plot=do_plot, suffix=suffix)
            image_tuple_s3 = tau_image(ir, rec, cal_layer=3, do_plot=do_plot, suffix=suffix)

            if image_tuple_s1 is None:
                continue
            if image_tuple_s2 is None:
                continue
            if image_tuple_s3 is None:
                continue
            
            s1 = image_tuple_s1[0]
            s2 = image_tuple_s2[0]
            s3 = image_tuple_s3[0]
            pt = rec['off_pt']
            eta = rec['off_eta']
            mu = rec['averageintpercrossing']


            image = np.array([(s1, s2, s3, pt, eta, mu)],
                              dtype=[
                    ('s1', 'f8', s1.shape), 
                    ('s2', 'f8', s2.shape), 
                    ('s3', 'f8', s3.shape), 
                    ('pt', 'f8'), 
                    ('eta', 'f8'), 
                    ('mu', 'f8')])
            images.append(image)

        else:
            image_tuple = tau_image(ir, rec, cal_layer=cal_layer, rotate_pc=False)
            if image_tuple is not None:
                image_cal, kin_rec, eta, phi, ene = image_tuple

                
                image = np.array([(image_cal, pt, eta, mu)],
                                 dtype=[
                        ('s{0}'.format(cal_layer), 'f8', image_cal.shape), 
                        ('pt', 'f8'), 
                        ('eta', 'f8'), 
                        ('mu', 'f8')])
                images.append(image)

                if do_plot and ir < 100:
                    # scatter for the selected pixels
                    plt.figure()
                    plt.scatter(
                        eta, phi, c=ene, marker='s', s=40,
                        label= 'Number of cells = {0}'.format(len(eta)))
                    plt.xlim(-0.4, 0.4)
                    plt.ylim(-0.4, 0.4)
                    plt.plot(
                    rec['true_charged_eta'] - rec['true_eta'], 
                    dphi(rec['true_charged_phi'], rec['true_phi']), 'ro', 
                    label='charge pi, pT = %1.2f GeV' % (rec['true_charged_pt'] / 1000.))
                    plt.plot(
                        rec['true_neutral_eta'] - rec['true_eta'], 
                        dphi(rec['true_neutral_phi'], rec['true_phi']), 'g^', 
                        label='neutral pi, pT = %1.2f GeV' % (rec['true_neutral_pt'] / 1000.))
                    plt.xlabel('eta')
                    plt.ylabel('phi')
                    plt.legend(loc='upper right', fontsize='small', numpoints=1)
                    plt.savefig('plots/imaging/selected_grid_%s_%s.pdf' % (suffix, ir))
                    plt.clf()
                    plt.close()
                    # heatmap
                    plt.imshow(image, interpolation='nearest')
                    plt.title('%s heatmap %s' % (suffix, ir))
                    plt.xlabel('eta')
                    plt.ylabel('phi')
                    plt.legend(loc='upper right', fontsize='small', numpoints=1)
                    plt.savefig('plots/imaging/heatmap_%s_%s.pdf' % (suffix, ir))
                    plt.clf()  
                    plt.close()

    # return the images to be stored
    print
    images = np.asarray(images)
    return images


