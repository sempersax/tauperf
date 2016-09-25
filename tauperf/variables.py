def get_label(variable):
    label = variable['root']
    if 'units' in variable.keys():
        label += ' [{0}]'.format(variable['units'])
    return label


VARIABLES = {

    'met': {
        'name': 'met',
        'root': 'missing-E_{T}',
        'type': 'f',
        'units': 'GeV',
        'scale': 0.001,
        'bins': 24,
        'range': (-20, 120)
        },

    'met_significance': {
        'name': 'met_significance',
        'root': 'missing-E_{T} Significance',
        'type': 'f',
        'units': '#sqrt{GeV}',
        'scale': 0.1,
        'bins': 24,
        'range': (-20, 120)
        },

    'pt': {
        'name': 'pt',
        'root': 'p_{T}',
        'type': 'f',
        'units': 'GeV',
        'scale': 0.001,
        'prefix': ('off', 'hlt', 'true'),
        'bins': 50,
        'range': (0, 200)
        },

#     'presel_pt': {
#         'name': 'presel_pt',
#         'root': 'Presel. p_{T}',
#         'type': 'f',
#         'units': 'GeV',
#         'scale': 0.001,
#         'prefix': ('hlt'),
#         'bins': 50,
#         'range': (0, 500)
#         },
    
    'eta': {
        'name': 'eta',
        'root': '#eta',
        'mpl': 'pseudorapidity',
        'type': 'f',
        'prefix': ('off', 'hlt', 'l1', 'true'),
        'bins': 20,
        'range': (-2.5, 2.5)
        },

    'phi': {
        'name': 'phi',
        'root': '#phi',
        'mpl': r'$\phi(\tau)$',
        'type': 'f',
        'prefix': ('off', 'hlt', 'l1', 'true'),
        'bins': 20,
        'range': (-3.15, 3.15)
        },

    'good_npv': {
        'name': 'good_npv',
        'root': 'Number of Primary Vertices',
        'mpl': 'Number of Primary Vertices',
        'type': 'i',
        'bins': 20,
        'range': (0, 40)
    },
    
    'averageintpercrossing': {
        'name': 'averageintpercrossing',
        'root': 'Average Interactions Per Bunch Crossing',
        'mpl': 'Average Interactions Per Bunch Crossing',
        'type': 'i',
        'bins': 40,
        'range': (0, 40)
        },

    'ntracks': {
        'name': 'ntracks',
        'root': 'Number of Tracks',
        'mpl': 'Number of Tracks',
        'type': 'i',
        'prefix': ('off', 'hlt'),
        'bins': 6,
        'range': (0, 6)
        },
    
#     'nprongs': {
#         'name': 'nprongs',
#         'root': 'Number of Prongs',
#         'type': 'i',
#         'prefix': ('true'),
#         'bins': 6,
#         'range': (0, 6)
#         },

    'centFrac': {
        'name': 'centFrac',
        'root': 'f_{core}',
        'mpl': r'$f_{core}$',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 80,
        'range': (-0.1, 1.5)
        },
    
    'centFracCorrected': {
        'name': 'centFracCorrected',
        'root': 'f_{core} (PU corrected)',
        'mpl': r'$f_{core}$ (PU corrected)',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 80,
        'range': (-0.1, 1.5)
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

    'innerTrkAvgDist': {
        'name': 'innerTrkAvgDist',
        'root': 'Inner R_{track}',
        'prefix': ('off', 'hlt'),
        'type': 'f',
        'bins': 60,
        'range': (-0.1, 0.2),
        },
   
    'innerTrkAvgDistCorrected': {
        'name': 'innerTrkAvgDistCorrected',
        'root': 'Inner R_{track} (PU corrected)',
        'prefix': ('off', 'hlt'),
        'type': 'f',
        'bins': 60,
        'range': (-0.1, 0.2),
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
        'bins' : 44,
        'range': (-0.1, 1),
        },

    'SumPtTrkFracCorrected': {
        'name': 'SumPtTrkFracCorrected',
        'root': '1 - p_{T}^{trk in 0.2} / p_{T}^{trk in 0.4} (PU corrected)',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins' : 44,
        'range': (-0.1, 1),
        },

    'ChPiEMEOverCaloEME': {
        'name': 'ChPiEMEOverCaloEME',
        'root': 'E_{#pi^{#pm}}/E_{calo}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 80,
        'range': (-40, 40),
        },
    
    'ChPiEMEOverCaloEMECorrected': {
        'name': 'ChPiEMEOverCaloEMECorrected',
        'root': 'E_{#pi^{#pm}}/E_{calo} (PU corrected)',
        'type': 'f',
        'prefix': ('hlt', 'off'),
        'bins': 80,
        'range': (-40, 40),
        },

    'etOverPtLeadTrk': {
        'name': 'etOverPtLeadTrk',
        'root': '1./f_{track}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 30,
        'range': (-1, 5),
        },
    
    'etOverPtLeadTrkCorrected': {
        'name': 'etOverPtLeadTrkCorrected',
        'root': '1./f_{track} (PU corrected)',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 30,
        'range': (-1, 5),
        },

    'EMPOverTrkSysP': {
        'name' : 'EMPOverTrkSysP',
        'root': 'p_{EM}/p_{tracks}',
        'prefix': ('off', 'hlt'),
        'type': 'f',
        'bins': 120,
        'range': (-1, 10),
        },

    'EMPOverTrkSysPCorrected': {
        'name' : 'EMPOverTrkSysPCorrected',
        'root': 'p_{EM}/p_{tracks} (PU corrected)',
        'prefix': ('hlt', 'off'),
        'type': 'f',
        'bins': 120,
        'range': (-1, 10),
        },

    'ipSigLeadTrk': {
        'name' : 'ipSigLeadTrk',
        'root': 'S_{lead track}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (-5, 5),
        },
    
    'ipSigLeadTrkCorrected': {
        'name' : 'ipSigLeadTrkCorrected',
        'root': '|S_{lead track}| (PU Corrected)',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 44,
        'range': (-2, 20),
        },

#     'ipSigLeadTrk_BS': {
#         'name' : 'ipSigLeadTrk_BS',
#         'root': 'S_{lead track} (Beamspot)',
#         'type': 'f',
#         'prefix': ('hlt'),
#         'bins': 20,
#         'range': (-5, 5),
#         },

    'AbsipSigLeadTrk': {
        'name' : 'AbsipSigLeadTrk',
        'root': '|S_{lead track}|',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 44,
        'range': (-2, 20),
        },
    
#     'AbsipSigLeadTrk_BS': {
#         'name' : 'AbsipSigLeadTrk_BS',
#         'root': '|S_{lead track} (Beamspot)|',
#         'type': 'f',
#         'prefix': ('hlt'),
#         'bins': 20,
#         'range': (0, 5),
#         },

#     'AbsipSigLeadTrk_BS_pileup_corrected': {
#         'name' : 'AbsipSigLeadTrk_BS_pileup_corrected',
#         'root': '|S_{lead track} (Beamspot)|  (PU corrected)',
#         'type': 'f',
#         'prefix': ('hlt'),
#         'bins': 20,
#         'range': (0, 5),
#         },

    'dRmax': {
        'name' : 'dRmax',
        'root': '#DeltaR_{max}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 20,
        'range': (0, 0.2),
        },
    
    'dRmaxCorrected': {
        'name' : 'dRmaxCorrected',
        'root': '#DeltaR_{max} (PU corrected)',
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
    
    'trFlightPathSigCorrected': {
        'name' : 'trFlightPathSigCorrected',
        'root': 'S_{T}^{flight} (PU corrected)',
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
        'bins': 25,
        # 'scale': 0.001,
        'units': 'MeV',
        'range': (0., 5000.),
        },

    'massTrkSysCorrected': {
        'name' : 'massTrkSysCorrected',
        'root': 'm_{track} (PU corrected)',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 25,
        # 'scale': 0.001,
        'units': 'MeV',
        'range': (0, 5000.),
        },

    'ptRatioEflowApprox': {
        'name' : 'ptRatioEflowApprox',
        'root': 'Approximated p_{T} ratio',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 50,
        'range': (-1, 4),
        },

    'ptRatioEflowApproxCorrected': {
        'name' : 'ptRatioEflowApproxCorrected',
        'root': 'Approximated p_{T} ratio (PU corrected)',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 50,
        'range': (-1, 4),
        },

    'mEflowApprox': {
        'name' : 'mEflowApprox',
        'root': 'Approximated m_{#tau}^{vis}',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 30,
        # 'scale': 0.001,
        'units': 'MeV',
        'range': (-100, 5000.),
        },

    'mEflowApproxCorrected': {
        'name' : 'mEflowApproxCorrected',
        'root': 'Approximated m_{#tau}^{vis} (PU corrected)',
        'type': 'f',
        'prefix': ('off', 'hlt'),
        'bins': 30,
        'units': 'MeV',
        'range': (-100, 5000.),
        },

}
