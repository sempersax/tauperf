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
        'range': (10, 2500)
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
        'bins': 80,
        'range': (0, 40)
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
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },
    
    'centFrac_pileup_corrected': {
        'name': 'centFrac_pileup_corrected',
        'root': 'f_{core} (PU corrected)',
        'type': 'f',
        'prefix': ('hlt'),
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
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },

    'nStrip': {
        'name': 'nStrip',
        'root': 'Number of cells in the strips layer',
        'type': 'i',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 20)
        },
    
    'EMRadius': {
        'name': 'EMRadius',
        'root': 'Radius in the ECAL',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.4)
        },
    
    'hadRadius': {
        'name': 'hadRadius',
        'root': 'Radius in the HCAL',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.4)
        },
    
#     'emfraction': {
#         'name': 'EMFractionAtEMScale',
#         'root': 'f_{EM} (EMSCALE)',
#         'type': 'f',
#         'bins': 20,
#         'range': (0, 1)
#         },

#     'hadfraction': {
#         'name': 'HADFractionAtEMScale',
#         'root': 'f_{HAD} (EMSCALE)',
#         'type': 'f',
#         'bins': 20,
#         'range': (0, 1),
#         },
    
#     'hadenergy': {
#         'name': 'HadEnergy',
#         'root': 'E_{HAD}',
#         'type': 'f',
#         'bins': 20,
#         'range': (0, 40000),
#         'units': 'MeV'
#         },
    
#     'stripwidth': {
#         'name': 'stripWidth2',
#         'root': 'Energy weighed width in the S1 of theECAL',
#         'type': 'f',
#         'bins': 20,
#         'range': (-0.2, 0.2)
#         },
    
    'lead2clustereoverallclusterE': {
        'name': 'lead2ClusterEOverAllClusterE',
        'root': 'E_(2nd cluster)/E_{tot}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },

    'lead3clustereoverallclusterE': {
        'name': 'lead3ClusterEOverAllClusterE',
        'root': 'E_(3nd cluster)/E_{tot}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 1)
        },

    'numtopoclusters': {
        'name': 'numTopoClusters',
        'root': 'Number of topo clusters',
        'type': 'i',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 10)
        },
    
    'topoinvmass': {
        'name': 'topoInvMass',
        'root': 'Invariant Mass of The Topoclusters',
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
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.4),
        },
    
    'numefftopoclusters': {
        'name': 'numEffTopoClusters',
        'root': 'Effective Number of Topoclusters',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 10)
        },

    'efftopoinvmass': {
        'name': 'effTopoInvMass',
        'root': 'Invariant Mass of The Effective Topoclusters',
        'type': 'f',
        'prefix': ('hlt', 'off'),
        'bins': 20,
        'scale': 0.001,
        'range': (0, 40),
        'units': 'GeV'
    },

#     'efftopomeandeltar': {
#         'name': 'effTopoMeanDeltaR',
#         'root': 'Mean Radius of The Effective Topoclusters',
#         'type': 'f',
#         'prefix': ('off', 'hlt'),
#         'bins': 20,
#         'range': (0, 0.4),
#         },
    
    'trkAvgDist': {
        'name': 'trkAvgDist',
        'root': 'R_{track}',
        'prefix': ('off', 'hlt'),
        'type': 'f',
        'bins': 20,
        'range': (0, 0.4),
        },

    'InnerTrkAvgDist': {
        'name': 'InnerTrkAvgDist',
        'root': 'Inner R_{track}',
        'prefix': ('off', 'hlt'),
        'type': 'f',
        'bins': 20,
        'range': (0, 0.2),
        },
   
    'InnerTrkAvgDist_pileup_corrected': {
        'name': 'InnerTrkAvgDist_pileup_corrected',
        'root': 'Inner R_{track} (PU corrected)',
        'prefix': ('hlt'),
        'type': 'f',
        'bins': 20,
        'range': (0, 0.2),
        },

    'nwidetracks': {
        'name': 'nwidetracks',
        'root': 'N_{track}^{iso}',
        'type': 'i',
        'prefix': ('off', 'hlt'),
        'bins' : 20,
        'range': (0, 20),
        },
    
    'SumPtTrkFrac': {
        'name': 'SumPtTrkFrac',
        'root': '1 - p_{T}^{trk in 0.2} / p_{T}^{trk in 0.4}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins' : 20,
        'range': (0, 1),
        },

    'SumPtTrkFrac_pileup_corrected': {
        'name': 'SumPtTrkFrac_pileup_corrected',
        'root': '1 - p_{T}^{trk in 0.2} / p_{T}^{trk in 0.4}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins' : 20,
        'range': (0, 1),
        },

    'ChPiEMEOverCaloEME': {
        'name': 'ChPiEMEOverCaloEME',
        'root': 'E_{#pi^{#pm}}/E_{calo}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (-1, 1),
        },
    
    'ChPiEMEOverCaloEME_pileup_corrected': {
        'name': 'ChPiEMEOverCaloEME_pileup_corrected',
        'root': 'E_{#pi^{#pm}}/E_{calo}',
        'type': 'f',
        'prefix': ('hlt'),
        'bins': 20,
        'range': (0, 1),
        },

    'etOverPtLeadTrk': {
        'name': 'etOverPtLeadTrk',
        'root': '1./f_{track}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 3),
        },
    
    'etOverPtLeadTrk_pileup_corrected': {
        'name': 'etOverPtLeadTrk_pileup_corrected',
        'root': '1./f_{track}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 3),
        },

    'EMPOverTrkSysP': {
        'name' : 'EMPOverTrkSysP',
        'root': 'p_{EM}/p_{tracks}',
        'prefix': ('off', 'hlt'),
        'type': 'f',
        'bins': 20,
        'range': (0, 3),
        },

    'EMPOverTrkSysP_pileup_corrected': {
        'name' : 'EMPOverTrkSysP_pileup_corrected',
        'root': 'p_{EM}/p_{tracks}',
        'prefix': ('hlt'),
        'type': 'f',
        'bins': 20,
        'range': (0, 3),
        },

    'ipSigLeadTrk': {
        'name' : 'ipSigLeadTrk',
        'root': 'S_{lead track} (PV)',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (-5, 5),
        },
    
    'ipSigLeadTrk_BS': {
        'name' : 'ipSigLeadTrk_BS',
        'root': 'S_{lead track} (Beamspot)',
        'type': 'f',
        'prefix': ('hlt'),
        'bins': 20,
        'range': (-5, 5),
        },

    'AbsipSigLeadTrk': {
        'name' : 'AbsipSigLeadTrk',
        'root': '|S_{lead track} (PV)|',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 5),
        },
    
    'AbsipSigLeadTrk_BS': {
        'name' : 'AbsipSigLeadTrk_BS',
        'root': '|S_{lead track} (Beamspot)|',
        'type': 'f',
        'prefix': ('hlt'),
        'bins': 20,
        'range': (0, 5),
        },

    'AbsipSigLeadTrk_BS_pileup_corrected': {
        'name' : 'AbsipSigLeadTrk_BS_pileup_corrected',
        'root': '|S_{lead track} (Beamspot)|',
        'type': 'f',
        'prefix': ('hlt'),
        'bins': 20,
        'range': (0, 5),
        },

    'dRmax': {
        'name' : 'dRmax',
        'root': '#DeltaR_{max}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.2),
        },
    
    'trFlightPathSig': {
        'name' : 'trFlightPathSig',
        'root': 'S_{T}^{flight}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (-10, 30),
        },
    
    'massTrkSys': {
        'name' : 'massTrkSys',
        'root': 'm_{track}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'scale': 0.001,
        'units': 'GeV',
        'range': (0, 20),
        },

    'approx_ptRatio': {
        'name' : 'approx_ptRatio',
        'root': 'Approximated p_{T} ratio',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 2),
        },

    'approx_ptRatio_pileup_corrected': {
        'name' : 'approx_ptRatio_pileup_corrected',
        'root': 'Approximated p_{T} ratio',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 2),
        },

    'approx_vistau_m': {
        'name' : 'approx_vistau_m',
        'root': 'Approximated m_{#tau}^{vis}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'scale': 0.001,
        'units': 'GeV',
        'range': (0, 2),
        },

    'approx_vistau_m_pileup_corrected': {
        'name' : 'approx_vistau_m_pileup_corrected',
        'root': 'Approximated m_{#tau}^{vis}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'scale': 0.001,
        'units': 'GeV',
        'range': (0, 2),
        },

}
