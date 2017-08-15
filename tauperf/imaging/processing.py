import os
import numpy as np
import math

from .. import print_progress
from . import log; log = log[__name__]


def tau_topo_image(irec, rec, cal_layer=2, width=32, height=32):
    """
    """
    indices = np.where(rec['off_cells_samp'] == cal_layer)

    ene_ = rec['off_cells_e_norm'].take(indices[0])
    eta_ = rec['off_cells_deta_digit'].take(indices[0])
    phi_ = rec['off_cells_dphi_digit'].take(indices[0])

    # build a rectangle of 0 to start
    image = [[0 for j in range(width)] for i in range(height)]

    # loop over the recorded cells:
    # locate their position in the rectangle and replace the 0 by the energy value
    for eta, phi, ene in zip(eta_, phi_, ene_):
        eta_ind = int(eta + math.floor(width / 2))
        phi_ind = int(phi + math.floor(height / 2))
        if eta_ind < width  and eta_ind > 0 and phi_ind < height and phi_ind > 0:
            image[phi_ind][eta_ind] = ene
    image = np.asarray(image)
    return image

def tau_tracks_simple(rec):
    """
    Laser was here.
    """
    maxtracks = 15

    imp   = []
    deta  = []
    dphi  = []
    classes  = []

    indices = np.where(rec['off_tracks_deta'] > -1000)

    rp     = rec['off_tracks_p']            .take(indices[0])
    rdeta  = rec['off_tracks_deta']           .take(indices[0])
    rdphi  = rec['off_tracks_dphi']           .take(indices[0])
    rclass = rec['off_tracks_class']          .take(indices[0])

    tau_ene = rec['off_pt'] * np.cosh(rec['off_eta'])

    for (p, e, f, c) in zip(rp, rdeta, rdphi, rclass):
        imp.append(p / tau_ene)
        deta.append(e)
        dphi.append(f)
        classes.append(c)

    imp   += [0] * (maxtracks - len(imp))
    deta  += [0] * (maxtracks - len(deta))
    dphi  += [0] * (maxtracks - len(dphi))
    classes += [0] * (maxtracks - len(classes))

    tracks = zip(imp, deta, dphi, classes)
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
        try:
            rec = records[ir]
        except:
            raise ValueError('record array is broken')

        if cal_layer is None:

            # get the image for each layer
            s1 = tau_topo_image(ir, rec, cal_layer=1, width=120, height=4)
            s2 = tau_topo_image(ir, rec, cal_layer=2, width=32, height=32)
            s3 = tau_topo_image(ir, rec, cal_layer=3, width=16, height=32)
            s4 = tau_topo_image(ir, rec, cal_layer=12, width=16, height=16)
            s5 = tau_topo_image(ir, rec, cal_layer=13, width=16, height=16)

            if any(s is None for s in [s1, s2, s3, s4, s5]):
                continue


            pt = rec['off_pt']
            eta = rec['off_eta']
            phi = rec['off_phi']
            mu = rec['averageintpercrossing']
            pantau = rec['off_decaymode']
            truthmode = rec['true_decaymode']

            if do_tracks:
                tracks = tau_tracks_simple(rec)

                image = np.array([(
                            s1, s2, s3, s4, s5, tracks, 
                            pt, eta, phi, mu,
                            pantau, truthmode)],
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
                        ('mu', 'f8'),
                        ('pantau', 'f8'), 
                        ('truthmode', 'f8')])
            else:
                image = np.array([(
                            s1, s2, s3, pt, eta, phi, mu)],
                                 dtype=[
                        ('s1', 'f8', s1.shape), 
                        ('s2', 'f8', s2.shape), 
                        ('s3', 'f8', s3.shape), 
                        ('s4', 'f8', s4.shape), 
                        ('s5', 'f8', s5.shape), 
                        ('pt', 'f8'), 
                        ('eta', 'f8'), 
                        ('phi', 'f8'), 
                        ('mu', 'f8')])


            images.append(image)

        elif cal_layer == 2:

            image_layer = tau_topo_image(ir, rec, cal_layer=cal_layer, width=32, height=32)

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
        else:
            raise ValueError('Can not process for layer {0} alone'.format(cal_layer))

    # return the images to be stored
    print
    images = np.asarray(images)
    return images


