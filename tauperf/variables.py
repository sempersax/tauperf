def get_label(variable):
    label = variable['root']
    if 'units' in variable.keys():
        label += ' [{0}]'.format(variable['units'])
    return label


VARIABLES = {
    'pt': {
        'name': 'pt',
        'root': 'p_{T}',
        'type': 'f',
        'units': 'GeV',
        'scale': 0.001,
        'prefix': ('off', 'hlt', 'true'),
        'bins': 30,
        'range': (10, 100)
        },
    
    'eta': {
        'name': 'eta',
        'root': '#eta',
        'type': 'f',
        'prefix': ('off', 'hlt', 'l1', 'true'),
        'bins': 20,
        'range': (-2.5, 2.5)
        },

    'npv': {
        'name': 'npv',
        'root': 'Number of Primary Vertices',
        'type': 'i',
        'bins': 20,
        'range': (0, 80)
    },
    
    'averageintpercrossing': {
        'name': 'averageintpercrossing',
        'root': 'Average Interactions Per Bunch Crossing',
        'type': 'i',
        'bins': 100,
        'range': (0, 100)
        },

    'ntracks': {
        'name': 'ntracks',
        'root': 'Number of Tracks',
        'type': 'i',
        'prefix': ('off', 'hlt'),
        'bins': 6,
        'range': (0, 6)
        },
    
    'centFrac': {
        'name': 'centFrac',
        'root': 'f_{core}',
        'training': 'off_centFrac',
        'training_and': 'centFrac',
        'training_old': 'centFrac',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },
    
    'isolFrac': {
        'name': 'isolFrac',
        'root': 'f_{core}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },
    
    'PSSFraction': {
        'name': 'PSSFraction',
        'root': 'E_{PSS}/E_{cluster}',
        'training': 'off_PSSFraction',
        'training_and': 'PSSFraction',
        'training_old': 'PSSFraction',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },

    'nStrip': {
        'name': 'nStrip',
        'root': 'Number of cells in the strips layer',
        'training': 'off_nStrip',
        'training_old': 'nStrip',
        'type': 'i',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 20)
        },
    
    'EMRadius': {
        'name': 'EMRadius',
        'root': 'Radius in the ECAL',
        'training': 'off_EMRadius',
        'training_old': 'EMRadius',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.4)
        },
    
    'hadRadius': {
        'name': 'hadRadius',
        'root': 'Radius in the HCAL',
        'training': 'off_HadRadius',
        'training_old': 'HadRadius',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.4)
        },
    
#     'emfraction': {
#         'name': 'EMFractionAtEMScale',
#         'root': 'f_{EM} (EMSCALE)',
#         'training': 'off_EMFractionAtEMScale',
#         'training_old': 'EMFractionAtEMScale',
#         'type': 'f',
#         'bins': 20,
#         'range': (0, 1)
#         },

#     'hadfraction': {
#         'name': 'HADFractionAtEMScale',
#         'root': 'f_{HAD} (EMSCALE)',
#         'training': 'off_HADFractionAtEMScale',
#         'type': 'f',
#         'bins': 20,
#         'range': (0, 1),
#         },
    
#     'hadenergy': {
#         'name': 'HadEnergy',
#         'root': 'E_{HAD}',
#         'training': 'off_HadEnergy',
#         'training_old': 'HadEnergy',
#         'type': 'f',
#         'bins': 20,
#         'range': (0, 40000),
#         'units': 'MeV'
#         },
    
#     'stripwidth': {
#         'name': 'stripWidth2',
#         'root': 'Energy weighed width in the S1 of theECAL',
#         'training': 'off_stripWidth2',
#         'training_old': 'stripWidth2',
#         'type': 'f',
#         'bins': 20,
#         'range': (-0.2, 0.2)
#         },
    
    'lead2clustereoverallclusterE': {
        'name': 'lead2ClusterEOverAllClusterE',
        'root': 'E_(2nd cluster)/E_{tot}',
        'training': 'off_lead2ClusterEOverAllClusterE',
        'training_old': 'lead2ClusterEOverAllClusterE',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },

    'lead3clustereoverallclusterE': {
        'name': 'lead3ClusterEOverAllClusterE',
        'root': 'E_(3nd cluster)/E_{tot}',
        'training': 'off_lead3ClusterEOverAllClusterE',
        'training_old': 'lead3ClusterEOverAllClusterE',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },

    'numtopoclusters': {
        'name': 'numTopoClusters',
        'root': 'Number of topo clusters',
        'training': 'off_numTopoClusters',
        'type': 'i',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 10)
        },
    
    'topoinvmass': {
        'name': 'topoInvMass',
        'root': 'Invariant Mass of The Topoclusters',
        'training': 'off_topoInvMass',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'scale': 0.001,
        'bins': 20,
        'range': (0, 40),
        'units': 'GeV'
        },
    
    'topomeandeltar': {
        'name': 'topoMeanDeltaR',
        'root': 'Mean Radius of The Topoclusters',
        'training': 'off_topoMeanDeltaR',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.4),
        },
    
    'numefftopoclusters': {
        'name': 'numEffTopoClusters',
        'root': 'Effective Number of Topoclusters',
        'training': 'off_numEffTopoClusters',
        'training_old': 'numEffTopoClusters',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 10)
        },

    'efftopoinvmass': {
        'name': 'effTopoInvMass',
        'root': 'Invariant Mass of The Effective Topoclusters',
        'training': 'off_effTopoInvMass',
        'training_old': 'effTopoInvMass',
        'type': 'f',
        'prefix': ('hlt', 'off'),
        'bins': 20,
        'scale': 0.001,
        'range': (0, 40),
        'units': 'MeV'
    },

#     'efftopomeandeltar': {
#         'name': 'effTopoMeanDeltaR',
#         'root': 'Mean Radius of The Effective Topoclusters',
#         'training': 'off_effTopoMeanDeltaR',
#         'training_old': 'effTopoMeanDeltaR',
#         'type': 'f',
#         'prefix': ('off', 'hlt'),
#         'bins': 20,
#         'range': (0, 0.4),
#         },
    
    'trkAvgDist': {
        'name': 'trkAvgDist',
        'root': 'R_{track}',
        'training': 'off_trkAvgDist',
        'training_and': 'trkAvgDist',
        'prefix': ('off', 'hlt'),
        'type': 'f',
        'bins': 20,
        'range': (0, 0.4),
        },
   
    'nwidetracks': {
        'name': 'nwidetracks',
        'root': 'N_{track}^{iso}',
        'training': 'off_nWideTrk',
        'training_and': 'nWideTrk',
        'type': 'i',
        'prefix': ('off', 'hlt'),
        'bins' : 20,
        'range': (0, 20),
        },
    
    'ChPiEMEOverCaloEME': {
        'name': 'ChPiEMEOverCaloEME',
        'root': 'E_{#pi^{#pm}}/E_{calo}',
        'training': 'off_ChPiEMEOverCaloEME',
        'training_and': 'ChPiEMEOverCaloEME',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1),
        },
    
    'etOverPtLeadTrk': {
        'name': 'etOverPtLeadTrk',
        'root': '1./f_{track}',
        'training': 'off_EtOverLeadTrackPt',
        'training_and': 'etOverPtLeadTrk',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 3),
        },
    
#     'empovertrksysp': {
#         'name' : 'EMPOverTrkSysP',
#         'root': 'p_{EM}/p_{tracks}',
#         'training': 'off_EMPOverTrkSysP',
#         'training_and': 'EMPOverTrkSysP',
#         'type': 'f',
#         'bins': 20,
#         'range': (0, 3),
#         },

    'ipSigLeadTrk': {
        'name' : 'ipSigLeadTrk',
        'root': 'S_{lead track}',
        'training': 'off_ipSigLeadTrk',
        'training_and': 'ipSigLeadTrk',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 5),
        },
    
    'dRmax': {
        'name' : 'dRmax',
        'root': '#DeltaR_{max}',
        'training': 'off_dRmax',
        'training_and': 'dRmax',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.2),
        },
    
    'trFlightPathSig': {
        'name' : 'trFlightPathSig',
        'root': 'S_{T}^{flight}',
        'training': 'off_trFlightPathSig',
        'training_and': 'trFlightPathSig',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (-10, 30),
        },
    
    'massTrkSys': {
        'name' : 'massTrkSys',
        'root': 'm_{track}',
        'training': 'off_massTrkSys',
        'training_and': 'massTrkSys',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'scale': 0.001,
        'units': 'GeV',
        'range': (0, 20),
        },
}
