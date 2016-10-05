import os
import numpy as np
from root_numpy import tree2array
from rootpy.io import root_open
import matplotlib.pyplot as plt
from scipy import interpolate


def interpolate_rbf(x, y, z, function='linear'):
    xi, yi = np.meshgrid(
        np.linspace(x.min(), x.max(), 100),
        np.linspace(y.min(), y.max(), 100))
    rbf = interpolate.Rbf(x, y, z, function=function)
    return rbf(xi, yi)


rfile = root_open(os.path.join(
        os.getenv('DATA_AREA'), 
        'tauid_ntuples', 'output.root'))


tree = rfile['tau']

rec_1p1n = tree2array(tree, selection='true_nprongs==1 && true_npi0s == 1 && abs(off_eta) < 1.3').view(np.recarray)

for ir in xrange(len(rec_1p1n)):
    r = rec_1p1n[ir]
    indices = np.where(r['off_cells_samp']==2)

    eta = r['off_cells_deta'].take(indices[0])
    phi = r['off_cells_dphi'].take(indices[0])
    ene = r['off_cells_e'].take(indices[0])
    print ir, r['off_pt'], r['off_eta'], r['off_phi']
    im = plt.imshow(
        interpolate_rbf(eta, phi, ene), 
        extent=[eta.min(), eta.max(), phi.min(), phi.max()])
    plt.title('1p1n heatmap %s' % ir)
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.savefig('plots/heatmap_1p1n_%s.pdf' % ir)

rec_1p0n = tree2array(tree, selection='true_nprongs==1 && true_npi0s == 0 && abs(off_eta) < 1.3').view(np.recarray)

for ir in xrange(len(rec_1p0n)):
    r = rec_1p0n[ir]
    indices = np.where(r['off_cells_samp']==2)

    eta = r['off_cells_deta'].take(indices[0])
    phi = r['off_cells_dphi'].take(indices[0])
    ene = r['off_cells_e'].take(indices[0])
    print ir, r['off_pt'], r['off_eta'], r['off_phi']
    im = plt.imshow(
        interpolate_rbf(eta, phi, ene), 
        extent=[eta.min(), eta.max(), phi.min(), phi.max()])
    plt.title('1p0n heatmap %s' % ir)
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.savefig('plots/heatmap_1p0n_%s.pdf' % ir)


