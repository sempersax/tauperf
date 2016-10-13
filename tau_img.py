import os
from copy import deepcopy
import numpy as np
from root_numpy import tree2array
from rootpy.io import root_open
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
import skimage.transform as sk
from tauperf.parallel import Worker, run_pool
from tauperf import print_progress

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

def tau_image(rec, cal_layer=2, rotate_pc=True):
    """
    """
    indices = np.where(rec['off_cells_samp'] == cal_layer)
    if len(indices) == 0:
        return None
    eta_ = rec['off_cells_deta'].take(indices[0])
    phi_ = rec['off_cells_dphi'].take(indices[0])
    ene_ = rec['off_cells_e_norm'].take(indices[0])

    square_ = (np.abs(eta_) < 0.2) * (np.abs(phi_) < 0.2)
    eta = eta_[square_]
    phi = phi_[square_]
    ene = ene_[square_]

    if len(ene) == 0:
        return None
    image = interpolate_rbf(eta, phi, ene, rotate_pc=rotate_pc)
    return image, eta, phi, ene

def make_xy_scatter_matrix(x, y, z, scat_pow=2, mean_pow=1):

    cell_values = z
    cell_x = x
    cell_y = y

    etot = np.sum((cell_values>0) * np.power(cell_values, mean_pow))
    if etot == 0:
        print 'Found a jet with no energy.  DYING!'
        sys.exit(1)

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



def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi


data_dir = 'data_test'
rfile = root_open(os.path.join(
        os.getenv('DATA_AREA'), 
        'tauid_ntuples', 'output_6files.root'))


tree = rfile['tau']
rec_1p1n = tree2array(
    tree, selection='true_nprongs==1 && true_npi0s == 1 && abs(off_eta) < 1.1').view(np.recarray)


print 'process 1p1n:', len(rec_1p1n)
tau_1p1n_images = [] 
for ir in xrange(len(rec_1p1n)):
    print_progress(ir, len(rec_1p1n), prefix='Progress')
    image_tuple = tau_image(rec_1p1n[ir])
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

rec_1p0n = tree2array(
    tree, selection='true_nprongs==1 && true_npi0s == 0 && abs(off_eta) < 1.1').view(np.recarray)

print 'process 1p0n', len(rec_1p0n)
tau_1p0n_images = []
for ir in xrange(len(rec_1p0n)):
    print_progress(ir, len(rec_1p0n))
    image_tuple = tau_image(rec_1p0n[ir])
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

