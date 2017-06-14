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
from .plotting import plot_image, plot_heatmap
from ROOT import TLorentzVector

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


def tau_topo_image(irec, rec, cal_layer=2, width=32, height=32):
    """
    """
    indices = np.where(rec['off_cells_samp'] == cal_layer)
    #     if len(indices[0]) == 0:
    #         log.warning('event {0}: No cell selected in layer {1} --> need to figure out why'.format(irec, cal_layer))

    ene_ = rec['off_cells_e_norm'].take(indices[0])
    eta_ = rec['off_cells_deta_digit'].take(indices[0])
    phi_ = rec['off_cells_dphi_digit'].take(indices[0])

    image = [[0 for j in range(width)] for i in range(height)]
    for eta, phi, ene in zip(eta_, phi_, ene_):
        eta_ind = int(eta + math.floor(width / 2))
        phi_ind = int(phi + math.floor(height / 2))
        if eta_ind < width  and eta_ind > 0 and phi_ind < height and phi_ind > 0:
            image[phi_ind][eta_ind] = ene
    image = np.asarray(image)
    return image

def tau_calo_image(
    irec, rec, 
    rotate_pc=False, 
    cal_layer=2, 
    do_plot=False,
    suffix='1p1n'):
    """
    return a pixelized image of a tau candidate...
    """

    # retrieve eta, phi and energy arrays in a given layers
#     indices = np.where((rec['off_cells_samp'] == cal_layer) * (rec['off_ntracks'] == 1))
    indices = np.where(rec['off_cells_samp'] == cal_layer)

    if len(indices[0]) == 0:
        log.warning('No cell selected in layer --> need to figure out why'.format(cal_layer))
        return None

    eta_r = rec['off_cells_deta'].take(indices[0])
    phi_r = rec['off_cells_dphi'].take(indices[0])
    eta = rec['off_cells_deta_digit'].take(indices[0])
    phi = rec['off_cells_dphi_digit'].take(indices[0])
    ene = rec['off_cells_e_norm'].take(indices[0])
    ene_raw = rec['off_cells_e'].take(indices[0])

    # define the square used to collect cells for the image
    if cal_layer == 2:
        n_eta = 15
        n_phi = 15
        r_eta = 0.201
        r_phi = 0.201
        n_pixels = 225
    elif cal_layer == 1:
        n_eta = 120
        n_phi = 4
        r_eta = 0.401
        r_phi = 0.401
        n_pixels = 480
    elif cal_layer == 3:
        n_eta = 8
        n_phi = 15
        r_eta = 0.201
        r_phi = 0.201
        n_pixels = 120
    else:
        log.error('layer {0} is not implemented yet'.format(cal_layer))
        raise ValueError
        

    # collect cells in a square (or rectangle)
    square_ = (np.abs(eta) < n_eta) * (np.abs(phi) < n_phi) # *(np.abs(eta_r) < r_eta) * (np.abs(phi_r) < r_phi)

    eta_r_ = eta_r#[square_]
    phi_r_ = phi_r#[square_]
    eta_ = eta#[square_]
    phi_ = phi#[square_]
    ene_ = ene#[square_]

    # create the raw image
    arr = np.array([eta_r_, phi_r_, ene_])
    rec_new = np.core.records.fromarrays(
        arr, names='x, y, z', formats='f8, f8, f8')

    # order the pixels by sorting first by x and then by y
    rec_new = np.sort(rec_new, order=['x', 'y'])

    if do_plot is True:
        plot_image(
            rec, eta_r, phi_r, ene, irec, cal_layer, suffix)

    if len(ene) == 0:
        log.warning('pathologic case with 0 cells --> need to figure out why')
        return None

    # disgard image with wrong pixelization (need to fix!)
    #     n_pixels = 2 * n_eta * 2 * n_phi
    if len(rec_new) != n_pixels:
        log.debug('wrong array lenght: {0} (expect {1})'.format(len(rec_new), n_pixels))
        log.debug('image {3}: tau pt, eta, phi = {0}, {1}, {2}'.format(rec['off_pt'], rec['off_eta'], rec['off_phi'], irec))
        return None

    # reshaping
#     image = rec_new['z'].reshape((2 * n_eta, 2 * n_phi))
    image = rec_new['z'].reshape((n_eta, n_phi))

    if do_plot is True:
        
        # get real eta and phi of the central cell (to compute the distance of the lead trk from)
        eta_cells = rec['off_cells_eta'].take(indices[0])
        phi_cells = rec['off_cells_phi'].take(indices[0])
        eta_cells = eta_cells[square_]
        phi_cells = phi_cells[square_]
        dr_cells = eta_r_ * eta_r_ + phi_r_ * phi_r_
        index = np.argmin(dr_cells)
        
        pos_central_cell = {
            'eta': eta_cells[index], 
            'phi': phi_cells[index]
            }

        plot_image(
            rec, eta_r_, phi_r_, ene_, irec, cal_layer, 
            'selected_real_pix_' + suffix)

        plot_heatmap(
            image.T, rec, pos_central_cell, irec, cal_layer, 
            'selected_' + suffix, fixed_scale=False)

        plot_heatmap(
            image.T, rec, pos_central_cell, irec, cal_layer, 
            'selected_fixed_scale_' + suffix, fixed_scale=True)

    # rotating
    if rotate_pc:

        scat_mat, cent = make_xy_scatter_matrix(
            rec_new['x'], rec_new['y'], rec_new['z'])
        paxes, pvars = get_principle_axis(scat_mat)
        angle = np.arctan2(paxes[0, 0], paxes[0, 1])
        image = sk.rotate(
            image, np.rad2deg(angle), order=3)

    # return image
    return image

def dphi_corr(phi1, phi2):
    dphi = phi1 - phi2
    if dphi > math.pi:
        dphi = dphi - 2* math.pi
    elif dphi < -math.pi:
        dphi = dphi + 2 * math.pi
    else:
        pass
    return dphi

def tau_tracks(rec, n_eta=30, n_phi=30):
    """
    """
    indices = np.where(rec['off_tracks_pt'] > 0)
    pt = rec['off_tracks_pt'].take(indices[0])
    eta = rec['off_tracks_eta'].take(indices[0])
    phi = rec['off_tracks_phi'].take(indices[0])

    sum_vec = TLorentzVector()
    for (p, e, f) in zip(pt, eta, phi):
        v = TLorentzVector()
        v.SetPtEtaPhiM(p, e, f, 0)
        sum_vec += v
    
    tau_eta = rec['off_eta']
    tau_phi = rec['off_phi']

    deta = rec['off_tracks_eta'].take(indices[0]) - tau_eta
    deta_gran = deta / (0.4 / float(n_eta - 1))
    deta_ind = [math.floor(i) + (n_eta - 1)/2 for i in deta_gran]
    deta_ind = np.array(deta_ind, dtype=np.int)

    dphi = [dphi_corr(phi1, tau_phi) for phi1 in rec['off_tracks_phi'].take(indices[0])]
    dphi = np.array(dphi)
    dphi_gran = dphi / (0.4 / float(n_phi - 1)) 
    dphi_ind = [math.floor(i) + (n_phi - 1)/2 for i in dphi_gran]
    dphi_ind = np.array(dphi_ind, dtype=np.int)

    tracks = [[0 for i in range(n_eta)] for j in range(n_phi)]
    for i, j, e in zip(deta_ind, dphi_ind, pt):
        tracks[i][j] = e / sum_vec.Pt()
    tracks = np.asarray(tracks)

    return tracks

def tau_tracks_simple(rec):
    """
    Laser was here.
    """
    maxtracks = 10
    imp  = []
    deta = []
    dphi = []
    d0   = []
    z0   = []

    indices = np.where(rec['off_tracks_deta'] > -1000)
    rpt  = rec['off_tracks_pt']            .take(indices[0])
    reta = rec['off_tracks_eta']           .take(indices[0])
    rphi = rec['off_tracks_phi']           .take(indices[0])
    rd0  = rec['off_tracks_d0']            .take(indices[0])
    rz0  = rec['off_tracks_z0sinThetaTJVA'].take(indices[0])
    rdeta = rec['off_tracks_deta']           .take(indices[0])
    rdphi = rec['off_tracks_dphi']           .take(indices[0])

    rp = rpt * np.cosh(reta)
#     sum_vec = TLorentzVector()
#     for (p, e, f) in zip(rpt, reta, rphi):
#         v = TLorentzVector()
#         v.SetPtEtaPhiM(p, e, f, 0)
#         sum_vec += v

    tau_eta = rec['off_eta']
    tau_phi = rec['off_phi']
    tau_ene = rec['off_pt'] * np.cosh(rec['off_eta'])

    for (p, e, f, d, z) in zip(rp, rdeta, rdphi, rd0, rz0):
        imp.append(p / tau_ene)
        deta.append(e)
        dphi.append(f)
        d0.append(d)
        z0.append(z)

    imp  += [0] * (maxtracks - len(imp))
    deta += [0] * (maxtracks - len(deta))
    dphi += [0] * (maxtracks - len(dphi))
    d0   += [0] * (maxtracks - len(d0)  )
    z0   += [0] * (maxtracks - len(z0)  )

    #tracks = zip(pt, deta, dphi, d0, z0)
    tracks = zip(imp, deta, dphi, d0)
    tracks = np.asarray(tracks)

    return tracks

def process_taus(
    records, 
    nentries=None, 
    cal_layer=None, 
    do_tracks=False,
    do_plot=False, 
    suffix='1p1n', 
    show_progress=True):
    '''
    process the records one by one and compute the image for each layer
    ----------
    returns a record array of the images + basic kinematics of the tau
    '''

    log.info('')

    # make a list (convert to array later)
    images = []

    for ir in xrange(len(records)):

        # fancy printout
        if show_progress: 
            if nentries is None:
                print_progress(ir, len(records), prefix='Progress')
            else:
                print_progress(ir, nentries, prefix='Progress')
            
        # kill the loop after number of specified entries
        if nentries is not None and ir == nentries:
            break

        # protect against pathologic arrays
        #         try:
        #             rec = records[ir]
        #         except:
        #             rec = None
        
        #         if rec is None:
        #             print ir
        #             log.warning('record array is broken')
        #             continue

        rec = records[ir]

        if cal_layer is None:

            # get the image for each layer
            s1 = tau_topo_image(ir, rec, cal_layer=1, width=120, height=4)
            s2 = tau_topo_image(ir, rec, cal_layer=2, width=16, height=16)
            s3 = tau_topo_image(ir, rec, cal_layer=3, width=8, height=16)
            s4 = tau_topo_image(ir, rec, cal_layer=12, width=8, height=8)
            s5 = tau_topo_image(ir, rec, cal_layer=13, width=8, height=8)

#             s1 = tau_calo_image(ir, rec, cal_layer=1, do_plot=do_plot, suffix=suffix)
#             s2 = tau_calo_image(ir, rec, cal_layer=2, do_plot=do_plot, suffix=suffix)
#             s3 = tau_calo_image(ir, rec, cal_layer=3, do_plot=do_plot, suffix=suffix)

            if s1 is None:
                continue

            if s2 is None:
                continue

            if s3 is None:
                continue
            
            if s4 is None:
                continue
            
            if s5 is None:
                continue

            pt = rec['off_pt']
            eta = rec['off_eta']
            phi = rec['off_phi']
            ntracks= rec['off_ntracks']
            empovertrksysp = rec['off_EMPOverTrkSysP']
            chpiemeovercaloeme = rec['off_ChPiEMEOverCaloEME']
            masstrksys = rec['off_massTrkSys']
            mu = rec['averageintpercrossing']

            if do_tracks:
                #tracks = tau_tracks(rec, n_eta=30, n_phi=30)
                tracks = tau_tracks_simple(rec)

                image = np.array([(
                            s1, s2, s3, s4, s5, tracks, pt, eta, phi, 
                            ntracks, empovertrksysp, chpiemeovercaloeme, masstrksys, mu)],
                                 dtype=[
                        ('s1', 'f8', s1.shape), 
                        ('s2', 'f8', s2.shape), 
                        ('s3', 'f8', s3.shape), 
                        ('s4', 'f8', s4.shape), 
                        ('s5', 'f8', s5.shape), 
                        ('tracks', 'f8', tracks.shape), 
                        ('pt', 'f8'), 
                        ('eta', 'f8'), 
                        ('phi', 'f8'), 
                        ('ntracks', 'f8'), 
                        ('empovertrksysp', 'f8'), 
                        ('chpiemeovercaloeme', 'f8'), 
                        ('masstrksys', 'f8'), 
                        ('mu', 'f8')])
            else:
                image = np.array([(
                            s1, s2, s3, pt, eta, phi, 
                            ntracks, empovertrksysp, chpiemeovercaloeme, masstrksys, mu)],
                                 dtype=[
                        ('s1', 'f8', s1.shape), 
                        ('s2', 'f8', s2.shape), 
                        ('s3', 'f8', s3.shape), 
                        ('s4', 'f8', s4.shape), 
                        ('s5', 'f8', s5.shape), 
                        ('pt', 'f8'), 
                        ('eta', 'f8'), 
                        ('phi', 'f8'), 
                        ('ntracks', 'f8'), 
                        ('empovertrksysp', 'f8'), 
                        ('chpiemeovercaloeme', 'f8'), 
                        ('masstrksys', 'f8'), 
                        ('mu', 'f8')])


            images.append(image)

        else:

            image_layer = tau_calo_image(ir, rec, cal_layer=cal_layer, do_plot=do_plot, suffix=suffix)

            if image_layer is None:
                continue

            pt = rec['off_pt']
            eta = rec['off_eta']
            mu = rec['averageintpercrossing']
                
            image = np.array([(
                        image_layer, pt, eta, mu)],
                             dtype=[
                    ('s{0}'.format(cal_layer), 'f8', image_layer.shape), 
                    ('pt', 'f8'), 
                    ('eta', 'f8'), 
                    ('mu', 'f8')])
            images.append(image)

    # return the images to be stored
    print
    images = np.asarray(images)
    return images


