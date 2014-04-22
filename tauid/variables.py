def get_label(variable):
    label = variable['root']
    if 'units' in variable.keys:
        label += ' [{0}]'.format(variable['units'])
    return label

centfrac = {
    'name': 'centFrac',
    'root': 'f_{core}',
    'training': 'off_centFrac',
    'type': 'F',
    'bins': 20,
    'range': (0, 1)
    }
pssfraction = {
    'name': 'pssfraction',
    'root': 'E_{PSS}/E_{cluster}',
    'training': 'off_PSSFraction',
    'type': 'F',
    'bins': 20,
    'range': (0, 1)
    }
nstrip = {
    'name': 'nStrip',
    'root': 'Number of cells in the strips layer',
    'training': 'off_nStrip',
    'type': 'I',
    'bins': 20,
    'range': (0, 20)
    }
emradius = {
    'name': 'EMRadius',
    'root': 'Radius in the ECAL',
    'training': 'off_EMRadius',
    'type': 'F',
    'bins': 20,
    'range': (0, 0.4)
    }
hadradius = {
    'name': 'HadRadius',
    'root': 'Radius in the HCAL',
    'training': 'off_HadRadius',
    'type': 'F',
    'bins': 20,
    'range': (0, 0.4)
    }
emfraction = {
    'name': 'EMFraction',
    'root': 'f_{EM} (EMSCALE)',
    'training': 'off_EMFractionAtEMSCale',
    'type': 'F',
    'bins': 20,
    'range': (0, 1)
    }
hadfraction = {
    'name': 'HadFraction',
    'root': 'f_{HAD}',
    'training': 'off_HadFraction',
    'type': 'F',
    'bins': 20,
    'range': (0, 1)
    }
stripwidth = {
    'name': 'stripWidth2',
    'root': 'Energy weighed width in the S1 of theECAL',
    'training': 'off_stripWidth2',
    'type': 'F',
    'bins': 20,
    'range': (0, 20)
    }
lead2clustereoverallclusterE = {
    'name': 'lead2ClusterEOverAllClusterE',
    'root': 'E_(2nd cluster)/E_{tot}',
    'training': 'off_lead2ClusterEOverAllClusterE',
    'type': 'F',
    'bins': 20,
    'range': (0, 1)
    }
lead3clustereoverallclusterE = {
    'name': 'lead3ClusterEOverAllClusterE',
    'root': 'E_(3nd cluster)/E_{tot}',
    'training': 'off_lead3ClusterEOverAllClusterE',
    'type': 'F',
    'bins': 20,
    'range': (0, 1)
    }
numtopoclusters = {
    'name': 'numTopoClusters',
    'root': 'Number of topo clusters'
    'taining': 'off_numTopoClusters',
    'type': 'I',
    'bins': 20,
    'range': (0, 10)
    }
topoinvmass = {
    'name': 'TopoInvMass',
    'root': 'Invariant Mass of The Topoclusters'
    'training': 'off_TopoInvMass',
    'type': 'F',
    'bins': 20,
    'range': (0, 40000),
    'units': 'MeV'
    }
topomeandeltar = {
    'name': 'TopoMeanDeltaR',
    'root': 'Mean Radius of The Topoclusters'
    'training': 'off_TopoMeanDeltaR',
    'type': 'F',
    'bins': 20,
    'range': (0, 0.4),
    }
numefftopoclusters = {
    'name': 'numEffTopoClusters',
    'root': 'Effective Number of Topoclusters'
    'taining': 'off_numEffTopoClusters',
    'type': 'F',
    'bins': 20,
    'range': (0, 10)
    }
efftopoinvmass = {
    'name': 'effTopoInvMass',
    'root': 'Invariant Mass of The Effective Topoclusters'
    'training': 'off_effTopoInvMass',
    'type': 'F',
    'bins': 20,
    'range': (0, 40000),
    'units': 'MeV'
    }
efftopomeandeltar = {
    'name': 'effTopoMeanDeltaR',
    'root': 'Mean Radius of The Effective Topoclusters'
    'training': 'off_effTopoMeanDeltaR',
    'type': 'F',
    'bins': 20,
    'range': (0, 0.4),
    }

