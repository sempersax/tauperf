def get_label(variable):
    label = variable['root']
    if 'units' in variable.keys():
        label += ' [{0}]'.format(variable['units'])
    return label

pt = {
    'name': 'pt',
    'root': 'p_{T}',
    'type': 'f',
    'units': 'MeV',
    'bins': 20,
    'range': (10, 1e5)
    }

eta = {
    'name': 'eta',
    'root': '#eta',
    'type': 'f',
    'bins': 25,
    'range': (-2.5, 2.5)
    }

npv = {
    'name': 'npv',
    'root': 'Number of Primary Vertices',
    'type': 'i',
    'bins': 20,
    'range': (0, 80)
    }

mu = {
    'name': 'mu',
    'root': 'Average Interactions Per Bunch Crossing',
    'type': 'i',
    'bins': 100,
    'range': (0, 100)
    }

ntracks = {
    'name': 'numTrack',
    'root': 'Number of Tracks',
    'type': 'i',
    'bins': 6,
    'range': (0, 6)
    }
centfrac = {
    'name': 'centFrac',
    'root': 'f_{core}',
    'training': 'off_centFrac',
    'training_and': 'centFrac',
    'training_old': 'centFrac',
    'type': 'f',
    'bins': 20,
    'range': (0, 1)
    }

pssfraction = {
    'name': 'PSSFraction',
    'root': 'E_{PSS}/E_{cluster}',
    'training': 'off_PSSFraction',
    'training_and': 'PSSFraction',
    'training_old': 'PSSFraction',
    'type': 'f',
    'bins': 20,
    'range': (0, 1)
    }

nstrip = {
    'name': 'nStrip',
    'root': 'Number of cells in the strips layer',
    'training': 'off_nStrip',
    'training_old': 'nStrip',
    'type': 'i',
    'bins': 20,
    'range': (0, 20)
    }

emradius = {
    'name': 'EMRadius',
    'root': 'Radius in the ECAL',
    'training': 'off_EMRadius',
    'training_old': 'EMRadius',
    'type': 'f',
    'bins': 20,
    'range': (0, 0.4)
    }

hadradius = {
    'name': 'HadRadius',
    'root': 'Radius in the HCAL',
    'training': 'off_HadRadius',
    'training_old': 'HadRadius',
    'type': 'f',
    'bins': 20,
    'range': (0, 0.4)
    }

emfraction = {
    'name': 'EMFractionAtEMScale',
    'root': 'f_{EM} (EMSCALE)',
    'training': 'off_EMFractionAtEMScale',
    'training_old': 'EMFractionAtEMScale',
    'type': 'f',
    'bins': 20,
    'range': (0, 1)
    }

hadfraction = {
    'name': 'HADFractionAtEMScale',
    'root': 'f_{HAD} (EMSCALE)',
    'training': 'off_HADFractionAtEMScale',
    'type': 'f',
    'bins': 20,
    'range': (0, 1),
    'units': 'MeV'
    }

hadenergy = {
    'name': 'HadEnergy',
    'root': 'E_{HAD}',
    'training': 'off_HadEnergy',
    'training_old': 'HadEnergy',
    'type': 'f',
    'bins': 20,
    'range': (0, 40000),
    'units': 'MeV'
    }

stripwidth = {
    'name': 'stripWidth2',
    'root': 'Energy weighed width in the S1 of theECAL',
    'training': 'off_stripWidth2',
    'training_old': 'stripWidth2',
    'type': 'f',
    'bins': 20,
    'range': (-0.2, 0.2)
    }
lead2clustereoverallclusterE = {
    'name': 'lead2ClusterEOverAllClusterE',
    'root': 'E_(2nd cluster)/E_{tot}',
    'training': 'off_lead2ClusterEOverAllClusterE',
    'training_old': 'lead2ClusterEOverAllClusterE',
    'type': 'f',
    'bins': 20,
    'range': (0, 1)
    }

lead3clustereoverallclusterE = {
    'name': 'lead3ClusterEOverAllClusterE',
    'root': 'E_(3nd cluster)/E_{tot}',
    'training': 'off_lead3ClusterEOverAllClusterE',
    'training_old': 'lead3ClusterEOverAllClusterE',
    'type': 'f',
    'bins': 20,
    'range': (0, 1)
    }

numtopoclusters = {
    'name': 'numTopoClusters',
    'root': 'Number of topo clusters',
    'training': 'off_numTopoClusters',
    'type': 'i',
    'bins': 20,
    'range': (0, 10)
    }

topoinvmass = {
    'name': 'topoInvMass',
    'root': 'Invariant Mass of The Topoclusters',
    'training': 'off_topoInvMass',
    'type': 'f',
    'bins': 20,
    'range': (0, 40000),
    'units': 'MeV'
    }

topomeandeltar = {
    'name': 'topoMeanDeltaR',
    'root': 'Mean Radius of The Topoclusters',
    'training': 'off_topoMeanDeltaR',
    'type': 'f',
    'bins': 20,
    'range': (0, 0.4),
    }

numefftopoclusters = {
    'name': 'numEffTopoClusters',
    'root': 'Effective Number of Topoclusters',
    'training': 'off_numEffTopoClusters',
    'training_old': 'numEffTopoClusters',
    'type': 'f',
    'bins': 20,
    'range': (0, 10)
    }

efftopoinvmass = {
    'name': 'effTopoInvMass',
    'root': 'Invariant Mass of The Effective Topoclusters',
    'training': 'off_effTopoInvMass',
    'training_old': 'effTopoInvMass',
    'type': 'f',
    'bins': 20,
    'range': (0, 40000),
    'units': 'MeV'
    }

efftopomeandeltar = {
    'name': 'effTopoMeanDeltaR',
    'root': 'Mean Radius of The Effective Topoclusters',
    'training': 'off_effTopoMeanDeltaR',
    'training_old': 'effTopoMeanDeltaR',
    'type': 'f',
    'bins': 20,
    'range': (0, 0.4),
    }

trkavgdist = {
    'name': 'trkAvgDist',
    'root': 'R_{track}',
    'training': 'off_trkAvgDist',
    'training_and': 'trkAvgDist',
    'type': 'f',
    'bins': 20,
    'range': (0, 0.4),
    }
   
nwidetrk = {
    'name': 'nWideTrk',
    'root': 'N_{track}^{iso}',
    'training': 'off_nWideTrk',
    'training_and': 'nWideTrk',
    'type': 'i',
    'bins' : 20,
    'range': (0, 20),
    }

chpiemeovercaloeme = {
    'name': 'ChPiEMEOverCaloEME',
    'root': 'E_{#pi^{#pm}}/E_{calo}',
    'training': 'off_ChPiEMEOverCaloEME',
    'training_and': 'ChPiEMEOverCaloEME',
    'type': 'f',
    'bins': 20,
    'range': (0, 1),
    }
etoverptleadtrk = {
    'name': 'EtOverLeadTrackPt',
    'root': '1./f_{track}',
    'training': 'off_EtOverLeadTrackPt',
    'training_and': 'etOverPtLeadTrk',
    'type': 'f',
    'bins': 20,
    'range': (0, 3),
    }
    
empovertrksysp = {
    'name' : 'EMPOverTrkSysP',
    'root': 'p_{EM}/p_{tracks}',
    'training': 'off_EMPOverTrkSysP',
    'training_and': 'EMPOverTrkSysP',
    'type': 'f',
    'bins': 20,
    'range': (0, 3),
    }

ipsigleadtrk = {
    'name' : 'ipSigLeadTrk',
    'root': 'S_{lead track}',
    'training': 'off_ipSigLeadTrk',
    'training_and': 'ipSigLeadTrk',
    'type': 'f',
    'bins': 20,
    'range': (0, 5),
    }

drmax = {
    'name' : 'dRmax',
    'root': '#DeltaR_{max}',
    'training': 'off_dRmax',
    'training_and': 'dRmax',
    'type': 'f',
    'bins': 20,
    'range': (0, 0.2),
    }

trflightpathsig = {
    'name' : 'trFlightPathSig',
    'root': 'S_{T}^{flight}',
    'training': 'off_trFlightPathSig',
    'training_and': 'trFlightPathSig',
    'type': 'f',
    'bins': 20,
    'range': (-10, 30),
    }

masstrksys = {
    'name' : 'massTrkSys',
    'root': 'm_{track}',
    'training': 'off_massTrkSys',
    'training_and': 'massTrkSys',
    'type': 'f',
    'bins': 20,
    'units': 'MeV',
    'range': (0, 20e3),
    }
