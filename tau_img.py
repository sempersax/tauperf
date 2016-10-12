import os
import numpy as np
from root_numpy import tree2array
from rootpy.io import root_open
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import skimage.transform as sk


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

def interpolate_rbf(x, y, z, function='linear', rotate_pc=True):

    xi, yi = np.meshgrid(
        np.linspace(x.min(), x.max(), 100),
        np.linspace(y.min(), y.max(), 100))

    rbf = interpolate.Rbf(x, y, z, function=function)
    im = rbf(xi, yi)

    if rotate_pc:
        scat_mat, cent = make_xy_scatter_matrix(x, y, z)
        paxes, pvars = get_principle_axis(scat_mat)
        angle = np.arctan2(paxes[0, 0], paxes[0, 1])
        im = sk.rotate(
            im, np.rad2deg(angle), order=3)
    return im


def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi


def tau_image(rec, cal_layer=2, rotate_pc=True):
    """
    """
    indices = np.where(rec['off_cells_samp'] == cal_layer)
    if len(indices) == 0:
        return None
    eta = rec['off_cells_deta'].take(indices[0])
    phi = rec['off_cells_dphi'].take(indices[0])
    ene = rec['off_cells_e_norm'].take(indices[0])
    if len(ene) == 0:
        return None
    im = interpolate_rbf(eta, phi, ene, rotate_pc=rotate_pc)
    return im

rfile = root_open(os.path.join(
        os.getenv('DATA_AREA'), 
        'tauid_ntuples', 'output.root'))


tree = rfile['tau']
rec_1p1n = tree2array(tree, selection='true_nprongs==1 && true_npi0s == 1 && abs(off_eta) < 1.3').view(np.recarray)


print 'process 1p1n:', len(rec_1p1n)
tau_1pn_images = [
    tau_image(rec_1p1n[ir]) for ir in xrange(len(rec_1p1n))
] 

for ir, image in enumerate(tau_1pn_images):
    if image is None:
        continue
    r = rec_1p1n[ir]
    plt.imshow(image,
        extent=[-0.4, 0.4, -0.4, 0.4])
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

rec_1p0n = tree2array(tree, selection='true_nprongs==1 && true_npi0s == 0 && abs(off_eta) < 1.3').view(np.recarray)

print 'process 1p0n'
tau_1p0n_images = [
    tau_image(rec_1p0n[ir]) for ir in xrange(len(rec_1p0n))
] 

for ir, image in enumerate(tau_1p0n_images):
    if image is None:
        continue
    r = rec_1p0n[ir]

    plt.imshow(image,
        extent=[-0.4, 0.4, -0.4, 0.4])
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

