from array import array
import ROOT
from helpers import ordereddict


class FlatTree(ROOT.TTree):
    """
    A class to create a tree from a dictionnary of variables (int or float)
    and to provide a reset function for the branches
    The branches are made from python arrays
    """
    # ---------------------------------------------------
    def __init__(self,name,title):
        ROOT.TTree.__init__(self,name,title)
        # --> Declare the variables
        self.variables = ordereddict.OrderedDict()
        """A variable is added as [ array('i',[0]),'runnumber/I' ]"""
    # ---------------------------------------------------
    def ResetBranches(self):
        """Reset all the tree branches to dummy values"""
        for varName,var in self.variables.iteritems():
            if var[0].typecode is 'i':
                var[0][0] = -9999
            elif var[0].typecode is 'f':
                var[0][0] = -9999.
            else:
                raise RuntimeError('Branch should be an int or a float')
    # ---------------------------------------------------
    def SetTreeBranches(self):
        """Create tree branches for all variables"""
        # --> Event level variables 
        for varName, var in self.variables.iteritems():
            self.Branch( varName,var[0],var[1])


#----------------------------------------------------------------------
class SlimTree(FlatTree):
    """A class to dump tau related informations"""
    def __init__(self,name,title,include,isData):
        """Constructor"""
        FlatTree.__init__(self,name,title)
        self.include = include
        self._isData = isData


        # --> Basic Event level variables 
        self.variables['runnumber'] = [ array('i',[0]),'runnumber/I' ]
        self.variables['evtnumber'] = [ array('i',[0]),'evtnumber/I' ]
        self.variables['lumiblock'] = [ array('i',[0]),'lumiblock/I']
        self.variables['npv'      ] = [ array('i',[0]),'npv/I'      ]
        self.variables['mu'       ] = [ array('f',[0.]),'mu/F'       ]
        
        # --> Trigger chain
        self.variables['EF_tau20_medium1'] = [array('i',[0]),'EF_tau20_medium1/I']
        self.variables['EF_tauNoCut']      = [array('i',[0]),'EF_tauNoCut/I']
        self.variables['L2_tauNoCut']      = [array('i',[0]),'L2_tauNoCut/I']
        self.variables['L1_TAU8']          = [array('i',[0]),'L1_TAU8/I']
        self.variables['L1_TAU11I']        = [array('i',[0]),'L1_TAU11I/I']
        self.variables['L2_tau18Ti_loose2_e18vh_medium1'] = [array('i',[0]),'L2_tau18Ti_loose2_e18vh_medium1/I']      
        # --> Truth branches if MC
        if self._isData == False:
            self.TruthVariables()
        # --> Reco branches
        self.RecoVariables()
        # --> EF branches
        self.EFVariables()
        # --> L2 branches
        self.L2Variables()
        # --> L1 branches
        self.L1Variables()

        # --> Set the tree branches
        self.SetTreeBranches()


    # ---------------------------------------------------
    def TruthVariables(self):
        # Define array length for the tree
        maxn = 10000
        if 'basic' in self.include:
            # ---> Truth based variables
            self.variables[ 'truth_ismatched'] = [ array('i',[0]),   'truth_ismatched/I' ]
            self.variables[ 'truth_ismatched_dr'] = [ array('i',[0]),   'truth_ismatched_dr/I' ]
            self.variables[ 'truth_index'] = [ array('i',[0]),   'truth_index/I' ]
            self.variables[ 'truth_index_dr'] = [ array('i',[0]),   'truth_index_dr/I' ]
            self.variables[ 'truth_p'   ] = [ array('f',[0.]), 'truth_p/F' ]
            self.variables[ 'truth_pt'  ] = [ array('f',[0.]), 'truth_pt/F' ]
            self.variables[ 'truth_mass'] = [ array('f',[0.]), 'truth_mass/F' ]
            self.variables[ 'truth_eta']  = [ array('f',[0.]), 'truth_eta/F' ]
            self.variables[ 'truth_phi']  = [ array('f',[0.]), 'truth_phi/F' ]
            self.variables[ 'nProngs'   ] = [array('i',[0])     , 'nProngs/I' ]
            self.variables[ 'nPi0s'     ] = [array('i',[0]),      'nPi0s/I' ]

        if 'decays' in self.include:
            # ---> Truth based variables describing the decay of the pi0s (max 2 pi0s)
            self.variables[ 'pi01_photons_E'  ] = [array('f',maxn*[0.]), 'pi01_photons_E[2]/F' ]
            self.variables[ 'pi01_photons_eta'] = [array('f',maxn*[0.]), 'pi01_photons_eta[2]/F' ]
            self.variables[ 'pi01_photons_phi'] = [array('f',maxn*[0.]), 'pi01_photons_phi[2]/F' ]
            self.variables[ 'pi02_photons_E'  ] = [array('f',maxn*[0.]), 'pi02_photons_E[2]/F' ]
            self.variables[ 'pi02_photons_eta'] = [array('f',maxn*[0.]), 'pi02_photons_eta[2]/F' ]
            self.variables[ 'pi02_photons_phi'] = [array('f',maxn*[0.]), 'pi02_photons_phi[2]/F' ]
            # ---> Truth based variables describing the properties of the charged pions 
            self.variables[ 'pi_ch_E'  ] = [array('f',maxn*[0.]), 'pi_ch_E[nProngs]/F' ]
            self.variables[ 'pi_ch_eta'] = [array('f',maxn*[0.]), 'pi_ch_eta[nProngs]/F' ]
            self.variables[ 'pi_ch_phi'] = [array('f',maxn*[0.]), 'pi_ch_phi[nProngs]/F' ]

    # ---------------------------------------------------
    def RecoVariables(self):
        # Define array length for the tree
        maxn = 10000
        # ---> Basics tau kinematics (from tau_pt,tau_eta,tau_phi) D3PD branches
        if 'basic' in self.include:
            self.variables[ 'p'  ]      = [ array('f',[0.]),'p/F'   ]
            self.variables[ 'pt' ]      = [ array('f',[0.]),'pt/F'  ]
            self.variables[ 'eta']      = [ array('f',[0.]),'eta/F' ]
            self.variables[ 'phi']      = [ array('f',[0.]),'phi/F' ]
            self.variables[ 'numTrack'] = [ array('i',[0]) , 'numTrack/I']
            self.variables[ 'nTracks' ] = [ array('i',[0]) , 'nTracks/I' ]
            self.variables[ 'nWideTrk'] = [ array('i',[0]), 'nWideTrk/I']
            self.variables[ 'notherTrk' ]   = [array('i',[0]),'notherTrk/I' ]
            self.variables[ 'hasL1matched'] = [ array('i',[0]),'hasL1matched/I' ]
            self.variables[ 'L1matched_pt'] = [ array('f',[0]),'L1matched_pt/F' ]
            self.variables[ 'clbased_pt' ]  = [ array('f',[0.]),'clbased_pt/F']

        if 'EDMVariables' in self.include:
            # ---> Michel's BDT pi0s counting outputs
            self.variables[ 'pi0BDTPrimary'  ] = [array('f',[0.]), 'pi0BDTPrimary/F' ]
            self.variables[ 'pi0BDTSecondary'] = [array('f',[0.]), 'pi0BDTSecondary/F' ]
            # ---> Identification BDT results (3 working points: loose, medium and tight)
            self.variables[ 'BDTloose' ] = [array('f',[0.]),'BDTloose/F' ]
            self.variables[ 'BDTmedium'] = [array('f',[0.]),'BDTmedium/F']
            self.variables[ 'BDTtight' ] = [array('f',[0.]),'BDTtight/F' ]
            # ---> Input variables for Michel's pi0 counting algorithm
            self.variables[ 'EMPOverTrkSysP'    ] = [array('f',[0.]), 'EMPOverTrkSysP/F' ]
            self.variables[ 'ChPiEMEOverCaloEME'] = [array('f',[0.]), 'ChPiEMEOverCaloEME/F' ]
            self.variables[ 'PSSFraction'       ] = [array('f',[0.]), 'PSSFraction/F' ]
            self.variables[ 'EtOverLeadTrackPt' ] = [array('f',[0.]), 'EtOverLeadTrackPt/F' ]
            self.variables[ 'nStrip'        ]     = [array('i',[0]), 'nStrip/I' ]
            self.variables[ 'nEffStripCells'    ] = [array('f',[0.]), 'nEffStripCells/F' ]

        if 'TauID' in self.include:
            # ---> Input variables for the ID BDT
            self.variables[ 'corrCentFrac']    = [ array('f',[0.]), 'corrCentFrac/F' ]
            self.variables[ 'centFrac'    ]    = [ array('f',[0.]), 'centFrac/F'     ]                
            self.variables[ 'isolFrac'    ]    = [ array('f',[0.]), 'isolFrac/F'     ]                
            self.variables[ 'corrFTrk'    ]    = [ array('f',[0.]), 'corrFTrk/F'     ]                        
            self.variables[ 'trkAvgDist'  ]    = [ array('f',[0.]), 'trkAvgDist/F'   ]                      
            self.variables[ 'ipSigLeadTrk']    = [ array('f',[0.]), 'ipSigLeadTrk/F' ]                    
            self.variables[ 'pi0_ptratio' ]    = [ array('f',[0.]), 'pi0_ptratio/F'  ]                     
            self.variables[ 'pi0_vistau_m'   ] = [array('f',[0.]) ,'pi0_vistau_m/F' ]                    
            self.variables[ 'pi0_n_reco' ]     = [ array('i',[0]) , 'pi0_n_reco/I'  ]                     
            self.variables[ 'trFlightPathSig'] = [array('f',[0.]) ,'trFlightPathSig/F' ]                 
            self.variables[ 'massTrkSys'     ] = [array('f',[0.]) ,'massTrkSys/F' ]                      
            self.variables[ 'CaloRadius' ]     = [array('f',[0.]), 'CaloRadius/F']
            self.variables[ 'HADRadius' ]      = [array('f',[0.]), 'HADRadius/F']
            self.variables[ 'IsoFrac' ]        = [array('f',[0.]), 'IsoFrac/F']
            self.variables[ 'EMFrac' ]         = [array('f',[0.]), 'EMFrac/F']
            self.variables[ 'stripWidth' ]     = [array('f',[0.]), 'stripWidth/F']
            self.variables[ 'dRmax' ]          = [array('f',[0.]), 'dRmax/F']
            self.variables['EMRadius'  ]       = [ array('f',[0.]),'EMRadius/F' ]       
            self.variables['HadRadius' ]       = [ array('f',[0.]),'HadRadius/F']       
            self.variables['EMEnergy'  ]       = [ array('f',[0.]),'EMEnergy/F']        
            self.variables['HadEnergy'  ]      = [ array('f',[0.]),'HadEnergy/F']        
            self.variables['CaloRadius']       = [ array('f',[0.]),'CaloRadius/F']      
            self.variables['stripWidth2']      = [ array('f',[0.]),'stripWidth2/F']
            self.variables['numTopoClusters']    = [ array('i',[0]), 'numTopoClusters/I']
            self.variables['numEffTopoClusters'] = [ array('f',[0.]), 'numEffTopoClusters/F' ]
            self.variables['topoInvMass']        = [ array('f',[0.]), 'topoInvMass/F']
            self.variables['effTopoInvMass']     = [ array('f',[0.]), 'effTopoInvMass/F']
            self.variables['topoMeanDeltaR']     = [ array('f',[0.]),'topoMeanDeltaR/F']
            self.variables['effTopoMeanDeltaR']  = [ array('f',[0.]),'effTopoMeanDeltaR/F']
            self.variables['lead2ClusterEOverAllClusterE' ] = [ array('f',[0.]),'lead2ClusterEOverAllClusterE/F' ]
            self.variables['lead3ClusterEOverAllClusterE' ] = [ array('f',[0.]),'lead3ClusterEOverAllClusterE/F' ]
            self.variables['EMFractionAtEMScale']           = [ array('f',[0.]),'EMFractionAtEMScale/F' ]          

        if 'cellObjects' in self.include:
            # ---> Cells variables 
            self.variables[ 'celln'         ] = [ array('i',[0]),	'celln/I' ]
            self.variables[ 'cellE'         ] = [ array('f',maxn*[0.]),	'cellE[celln]/F' ]
            self.variables[ 'celleta'       ] = [ array('f',maxn*[0.]),	'celleta[celln]/F' ]
            self.variables[ 'cellphi'       ] = [ array('f',maxn*[0.]),	'cellphi[celln]/F' ]
            self.variables[ 'cellsamplingID'] = [ array('f',maxn*[0.]), 	'cellsamplingID[celln]/F' ]
            # ---> Cells variables for cells belonging to the first layer of the ECAL 
            self.variables['stripn'         ] = [array('i',[0])     ,      'stripn/I' ]
            self.variables['stripE'         ] = [array('f',maxn*[0.]),      'stripE[stripn]/F' ]
            self.variables['stripeta'       ] = [array('f',maxn*[0.]),      'stripeta[stripn]/F' ]
            self.variables['stripphi'       ] = [array('f',maxn*[0.]),      'stripphi[stripn]/F' ]
            self.variables['stripsamplingID'] = [array('f',maxn*[0.]), 	'stripsamplingID[stripn]/F' ]

        if 'recoObjects' in self.include:
            # ---> Clusters variables for clusters used to form the tau candidate
            self.variables['clustern'  ] = [array('i',[0])      , 'clustern/I'             ]
            self.variables['clusterE'  ] = [array('f',maxn*[0.]), 'clusterE[clustern]/F'   ]
            self.variables['clustereta'] = [array('f',maxn*[0.]), 'clustereta[clustern]/F' ]
            self.variables['clusterphi'] = [array('f',maxn*[0.]), 'clusterphi[clustern]/F' ]
            self.variables['clusters_m']     = [array('f',[0.])  , 'clusters_m/F'           ]
            self.variables['clusters_eff_m'] = [array('f',[0.])  , 'clusters_eff_m/F'       ]
            # ---> tracks variables (used TJVA corrected tracks)
            self.variables[ 'trackM'  ] = [ array('f',maxn*[0.]), 'trackM[nTracks]/F' ]
            self.variables[ 'trackPt' ] = [ array('f',maxn*[0.]), 'trackPt[nTracks]/F' ]
            self.variables[ 'trackE'  ] = [ array('f',maxn*[0.]), 'trackE[nTracks]/F' ]
            self.variables[ 'tracketa'] = [ array('f',maxn*[0.]), 'tracketa[nTracks]/F' ]
            self.variables[ 'trackphi'] = [ array('f',maxn*[0.]), 'trackphi[nTracks]/F' ]
            self.variables[ 'trackd0' ] = [ array('f',maxn*[0.]), 'trackd0[nTracks]/F' ]
            self.variables[ 'trackz0' ] = [ array('f',maxn*[0.]), 'trackz0[nTracks]/F' ]

        if 'wideTracks' in self.include:
            # ---> wide tracks variables 
            self.variables[ 'wide_track_M'  ] = [ array('f',maxn*[0.]), 'wide_track_M[nWideTrk]/F' ]
            self.variables[ 'wide_track_Pt' ] = [ array('f',maxn*[0.]), 'wide_track_Pt[nWideTrk]/F' ]
            self.variables[ 'wide_track_E'  ] = [ array('f',maxn*[0.]), 'wide_track_E[nWideTrk]/F' ]
            self.variables[ 'wide_track_eta'] = [ array('f',maxn*[0.]), 'wide_track_eta[nWideTrk]/F' ]
            self.variables[ 'wide_track_phi'] = [ array('f',maxn*[0.]), 'wide_track_phi[nWideTrk]/F' ]
            self.variables[ 'wide_track_d0' ] = [ array('f',maxn*[0.]), 'wide_track_d0[nWideTrk]/F' ]
            self.variables[ 'wide_track_z0' ] = [ array('f',maxn*[0.]), 'wide_track_z0[nWideTrk]/F' ]
            self.variables[ 'wide_track_nBLHits'  ] = [ array('i',maxn*[0]), 'wide_track_nBLHits[nWideTrk]/I' ]
            self.variables[ 'wide_track_nPixHits' ] = [ array('i',maxn*[0]), 'wide_track_nPixHits[nWideTrk]/I' ]
            self.variables[ 'wide_track_nSCTHits' ] = [ array('i',maxn*[0]), 'wide_track_nSCTHits[nWideTrk]/I' ]
            self.variables[ 'wide_track_nTRTHits' ] = [ array('i',maxn*[0]), 'wide_track_nTRTHits[nWideTrk]/I' ]
            self.variables[ 'wide_track_nHits'    ] = [ array('i',maxn*[0]), 'wide_track_nHits[nWideTrk]/I' ]

        if 'otherTracks' in self.include:
            # ---> other tracks variables 
            self.variables[ 'other_track_M'  ] = [ array('f',maxn*[0.]), 'other_track_M[notherTrk]/F' ]
            self.variables[ 'other_track_Pt' ] = [ array('f',maxn*[0.]), 'other_track_Pt[notherTrk]/F' ]
            self.variables[ 'other_track_E'  ] = [ array('f',maxn*[0.]), 'other_track_E[notherTrk]/F' ]
            self.variables[ 'other_track_eta'] = [ array('f',maxn*[0.]), 'other_track_eta[notherTrk]/F' ]
            self.variables[ 'other_track_phi'] = [ array('f',maxn*[0.]), 'other_track_phi[notherTrk]/F' ]
            self.variables[ 'other_track_d0' ] = [ array('f',maxn*[0.]), 'other_track_d0[notherTrk]/F' ]
            self.variables[ 'other_track_z0' ] = [ array('f',maxn*[0.]), 'other_track_z0[notherTrk]/F' ]
            self.variables[ 'other_track_nBLHits'  ] = [ array('i',maxn*[0]), 'other_track_nBLHits[notherTrk]/I' ]
            self.variables[ 'other_track_nPixHits' ] = [ array('i',maxn*[0]), 'other_track_nPixHits[notherTrk]/I' ]
            self.variables[ 'other_track_nSCTHits' ] = [ array('i',maxn*[0]), 'other_track_nSCTHits[notherTrk]/I' ]
            self.variables[ 'other_track_nTRTHits' ] = [ array('i',maxn*[0]), 'other_track_nTRTHits[notherTrk]/I' ]
            self.variables[ 'other_track_nHits'    ] = [ array('i',maxn*[0]), 'other_track_nHits[notherTrk]/I' ]

        if 'CaloDetails' in self.include:
            # --> Calorimeter details
            self.variables['PSEnergy'  ] = [ array('f',[0.]),'PSEnergy/F'  ]
            self.variables['S1Energy'  ] = [ array('f',[0.]),'S1Energy/F'  ]
            self.variables['S2Energy'  ] = [ array('f',[0.]),'S2Energy/F'  ]
            self.variables['S3Energy'  ] = [ array('f',[0.]),'S3Energy/F'  ]
            self.variables['HADEnergy']  = [ array('f',[0.]),'HADEnergy/F']
            self.variables['CaloEnergy'] = [ array('f',[0.]),'CaloEnergy/F']

        if 'Pi0Finder' in self.include:
            self.variables['pi0_vistau_m_alt'] = [ array('f',[0.]),'pi0_vistau_m_alt/F' ]
            self.variables['pi0_ptratio_alt']  = [ array('f',[0.]),'pi0_ptratio_alt/F'  ]

    # ---------------------------------------------------
    def EFVariables(self):
        # Define array length for the tree
        maxn = 10000
        # ---> Offline/EF matching variables
        self.variables[ 'EF_ismatched'    ] = [ array('i',[0]),   'EF_ismatched/I' ]
        self.variables[ 'EF_DeltaR_EF_off'] = [ array('f',[-9999.]),  'EF_DeltaR_EF_off/F' ]
        # --> Matching to the trigger chaina
        self.variables[ 'EF_EF_tau20_medium1' ] = [ array('i',[0]), 'EF_EF_tau20_medium1/I' ]
        self.variables[ 'EF_EF_tauNoCut'      ] = [ array('i',[0]), 'EF_EF_tauNoCut/I' ]

        if 'basic' in self.include:
            # ---> Basics tau kinematics at EventFilter level
            self.variables[ 'EF_p'         ] = [ array('f',[0.]),  'EF_p/F'   ]
            self.variables[ 'EF_pt'        ] = [ array('f',[0.]),  'EF_pt/F'  ]
            self.variables[ 'EF_eta'       ] = [ array('f',[0.]),  'EF_eta/F' ]
            self.variables[ 'EF_phi'       ] = [ array('f',[0.]),  'EF_phi/F' ]
            self.variables[ 'EF_numTrack'  ] = [ array('i',[0] ), 'EF_numTrack/I'   ]
            self.variables[ 'EF_nTracks'   ] = [ array('i',[0] ), 'EF_nTracks/I'   ] 		
            self.variables[ 'EF_nWideTrk'  ] = [ array('i',[0]), 'EF_nWideTrk/I'     ]                        
            self.variables[ 'EF_notherTrk' ] = [ array('i',[0]),'EF_notherTrk/I' ]                     

        if 'EDMVariables' in self.include:
            # ---> Michel's BDT pi0s counting outputs
            self.variables[ 'EF_pi0BDTPrimary'  ] = [ array('f',[0.]), 'EF_pi0BDTPrimary/F'   ]
            self.variables[ 'EF_pi0BDTSecondary'] = [ array('f',[0.]), 'EF_pi0BDTSecondary/F' ]
            # ---> Input variables for Michel's pi0 counting algorithm
            self.variables[ 'EF_EMPOverTrkSysP'    ] = [array('f',[0.]), 'EF_EMPOverTrkSysP/F' ]
            self.variables[ 'EF_ChPiEMEOverCaloEME'] = [array('f',[0.]), 'EF_ChPiEMEOverCaloEME/F' ]
            self.variables[ 'EF_PSSFraction'       ] = [array('f',[0.]), 'EF_PSSFraction/F' ]
            self.variables[ 'EF_EtOverLeadTrackPt' ] = [array('f',[0.]), 'EF_EtOverLeadTrackPt/F' ]
            self.variables[ 'EF_nStrip'        ]     = [array('i',[0] ), 'EF_nStrip/I' ]
            # variables[ 'EF_nEffStripCells'    ] = [array('f',[0.]), 'EF_nEffStripCells/F' ]

        if 'TauID' in self.include:
            # ---> Input variables for the ID BDT
            self.variables[ 'EF_corrCentFrac'   ] = [ array('f',[0.]), 'EF_corrCentFrac/F' ]
            self.variables[ 'EF_centFrac'       ] = [ array('f',[0.]), 'EF_centFrac/F'     ]                
            self.variables[ 'EF_corrFTrk'       ] = [ array('f',[0.]), 'EF_corrFTrk/F'     ]                        
            self.variables[ 'EF_FTrk'           ] = [ array('f',[0.]), 'EF_FTrk/F'         ]                        
            self.variables[ 'EF_trkAvgDist'     ] = [ array('f',[0.]), 'EF_trkAvgDist/F'   ]                      
            self.variables[ 'EF_ipSigLeadTrk'   ] = [ array('f',[0.]), 'EF_ipSigLeadTrk/F' ]                    
            self.variables[ 'EF_pi0_ptratio'    ] = [ array('f',[0.]), 'EF_pi0_ptratio/F'  ]                     
            self.variables[ 'EF_pi0_vistau_m'   ] = [array('f',[0.]) ,'EF_pi0_vistau_m/F' ]                    
            self.variables[ 'EF_trFlightPathSig'] = [array('f',[0.]) ,'EF_trFlightPathSig/F' ]                 
            self.variables[ 'EF_massTrkSys'     ] = [array('f',[0.]) ,'EF_massTrkSys/F' ]                      
            self.variables[ 'EF_topoMeanDeltaR' ] = [array('f',[0.]) ,'EF_topoMeanDeltaR/F' ]                      
            self.variables[ 'EF_CaloRadius' ]     = [array('f',[0.]), 'EF_CaloRadius/F']
            self.variables[ 'EF_HADRadius' ]      = [array('f',[0.]), 'EF_HADRadius/F']
            self.variables[ 'EF_IsoFrac' ]        = [array('f',[0.]), 'EF_IsoFrac/F']
            self.variables[ 'EF_EMFrac' ]         = [array('f',[0.]), 'EF_EMFrac/F']
            self.variables[ 'EF_stripWidth' ]     = [array('f',[0.]), 'EF_stripWidth/F']
            self.variables[ 'EF_dRmax' ]          = [array('f',[0.]), 'EF_dRmax/F']

        if 'recoObjects' in self.include:
            # ---> tracks variables 
            self.variables[ 'EF_trackM'  ] = [ array('f',maxn*[0.]),  'EF_trackM[EF_nTracks]/F' ]	        
            self.variables[ 'EF_trackE'  ] = [ array('f',maxn*[0.]),  'EF_trackE[EF_nTracks]/F'   ] 		
            self.variables[ 'EF_trackPt' ] = [ array('f',maxn*[0.]),  'EF_trackPt[EF_nTracks]/F' ]	        
            self.variables[ 'EF_tracketa'] = [ array('f',maxn*[0.]),  'EF_tracketa[EF_nTracks]/F' ]       	
            self.variables[ 'EF_trackphi'] = [ array('f',maxn*[0.]),  'EF_trackphi[EF_nTracks]/F' ]       	
            self.variables[ 'EF_trackd0' ] = [ array('f',maxn*[0.]),  'EF_trackd0[EF_nTracks]/F' ]
            self.variables[ 'EF_trackz0' ] = [ array('f',maxn*[0.]),  'EF_trackz0[EF_nTracks]/F' ]		

        if 'wideTracks' in self.include:
            # ---> wide tracks variables 
            self.variables[ 'EF_wide_track_M'  ] = [ array('f',maxn*[0.]), 'EF_wide_track_M[EF_nWideTrk]/F' ]
            self.variables[ 'EF_wide_track_Pt' ] = [ array('f',maxn*[0.]), 'EF_wide_track_Pt[EF_nWideTrk]/F' ]
            self.variables[ 'EF_wide_track_E'  ] = [ array('f',maxn*[0.]), 'EF_wide_track_E[EF_nWideTrk]/F' ]
            self.variables[ 'EF_wide_track_eta'] = [ array('f',maxn*[0.]), 'EF_wide_track_eta[EF_nWideTrk]/F' ]
            self.variables[ 'EF_wide_track_phi'] = [ array('f',maxn*[0.]), 'EF_wide_track_phi[EF_nWideTrk]/F' ]
            self.variables[ 'EF_wide_track_d0' ] = [ array('f',maxn*[0.]), 'EF_wide_track_d0[EF_nWideTrk]/F' ]
            self.variables[ 'EF_wide_track_z0' ] = [ array('f',maxn*[0.]), 'EF_wide_track_z0[EF_nWideTrk]/F' ]
            self.variables[ 'EF_wide_track_nBLHits'  ] = [ array('i',maxn*[0]), 'EF_wide_track_nBLHits[EF_nWideTrk]/I' ]
            self.variables[ 'EF_wide_track_nPixHits' ] = [ array('i',maxn*[0]), 'EF_wide_track_nPixHits[EF_nWideTrk]/I' ]
            self.variables[ 'EF_wide_track_nSCTHits' ] = [ array('i',maxn*[0]), 'EF_wide_track_nSCTHits[EF_nWideTrk]/I' ]
            self.variables[ 'EF_wide_track_nTRTHits' ] = [ array('i',maxn*[0]), 'EF_wide_track_nTRTHits[EF_nWideTrk]/I' ]
            self.variables[ 'EF_wide_track_nHits'    ] = [ array('i',maxn*[0]), 'EF_wide_track_nHits[EF_nWideTrk]/I' ]

        if 'otherTracks' in self.include:
            # ---> other tracks variables 
            self.variables[ 'EF_other_track_M'  ] = [ array('f',maxn*[0.]), 'EF_other_track_M[EF_notherTrk]/F' ]
            self.variables[ 'EF_other_track_Pt' ] = [ array('f',maxn*[0.]), 'EF_other_track_Pt[EF_notherTrk]/F' ]
            self.variables[ 'EF_other_track_E'  ] = [ array('f',maxn*[0.]), 'EF_other_track_E[EF_notherTrk]/F' ]
            self.variables[ 'EF_other_track_eta'] = [ array('f',maxn*[0.]), 'EF_other_track_eta[EF_notherTrk]/F' ]
            self.variables[ 'EF_other_track_phi'] = [ array('f',maxn*[0.]), 'EF_other_track_phi[EF_notherTrk]/F' ]
            self.variables[ 'EF_other_track_d0' ] = [ array('f',maxn*[0.]), 'EF_other_track_d0[EF_notherTrk]/F' ]
            self.variables[ 'EF_other_track_z0' ] = [ array('f',maxn*[0.]), 'EF_other_track_z0[EF_notherTrk]/F' ]
            self.variables[ 'EF_other_track_nBLHits'  ] = [ array('i',maxn*[0]), 'EF_other_track_nBLHits[EF_notherTrk]/I' ]
            self.variables[ 'EF_other_track_nPixHits' ] = [ array('i',maxn*[0]), 'EF_other_track_nPixHits[EF_notherTrk]/I' ]
            self.variables[ 'EF_other_track_nSCTHits' ] = [ array('i',maxn*[0]), 'EF_other_track_nSCTHits[EF_notherTrk]/I' ]
            self.variables[ 'EF_other_track_nTRTHits' ] = [ array('i',maxn*[0]), 'EF_other_track_nTRTHits[EF_notherTrk]/I' ]
            self.variables[ 'EF_other_track_nHits'    ] = [ array('i',maxn*[0]), 'EF_other_track_nHits[EF_notherTrk]/I' ]

        if 'CaloDetails' in self.include:
            # --> Calorimeter details
            self.variables['EF_PSEnergy'  ] = [ array('f',[0.]),'EF_PSEnergy/F'  ]
            self.variables['EF_S1Energy'  ] = [ array('f',[0.]),'EF_S1Energy/F'  ]
            self.variables['EF_S2Energy'  ] = [ array('f',[0.]),'EF_S2Energy/F'  ]
            self.variables['EF_S3Energy'  ] = [ array('f',[0.]),'EF_S3Energy/F'  ]
            self.variables['EF_HADEnergy' ] = [ array('f',[0.]),'EF_HADEnergy/F']
            self.variables['EF_CaloEnergy'] = [ array('f',[0.]),'EF_CaloEnergy/F']



    # ---------------------------------------------------
    def L2Variables(self):
        # ---> EF/L2 matching variables
        self.variables[ 'L2_ismatched'    ] = [ array('i',[0]),   'L2_ismatched/I' ]
        self.variables[ 'L2_DeltaR_L2_EF']  = [ array('f',[-9999.]),  'L2_DeltaR_L2_EF/F' ]
        self.variables['L2_L2_tau20_medium']                  = [ array('i',[0]),   'L2_L2_tau20_medium/I' ]                 
        self.variables['L2_L2_tau20_medium1']                 = [ array('i',[0]),   'L2_L2_tau20_medium1/I' ]                 
        self.variables['L2_L2_tauNoCut'    ]                  = [ array('i',[0]),   'L2_L2_tauNoCut/I' ]                 
        self.variables['L2_L2_tau18Ti_loose2_e18vh_medium1' ] = [ array('i',[0]),   'L2_L2_tau18Ti_loose2_e18vh_medium1/I' ]
        if 'basic' in self.include:
            # ---> Basics tau kinematics at EventFilter level
            self.variables[ 'L2_p'  ] = [ array('f',[0.]),  'L2_p/F'   ]
            self.variables[ 'L2_pt' ] = [ array('f',[0.]),  'L2_pt/F'  ]
            self.variables[ 'L2_eta'] = [ array('f',[0.]),  'L2_eta/F' ]
            self.variables[ 'L2_phi'] = [ array('f',[0.]),  'L2_phi/F' ]

        if 'TauID' in self.include:
            self.variables[ 'L2_CaloRadius' ]     = [array('f',[0.]), 'L2_CaloRadius/F']
            self.variables[ 'L2_HADRadius' ]      = [array('f',[0.]), 'L2_HADRadius/F']
            self.variables[ 'L2_IsoFrac' ]        = [array('f',[0.]), 'L2_IsoFrac/F']
            self.variables[ 'L2_EMFrac' ]         = [array('f',[0.]), 'L2_EMFrac/F']
            self.variables[ 'L2_stripWidth' ]     = [array('f',[0.]), 'L2_stripWidth/F']
            self.variables[ 'L2_HADtoEMEnergy' ]  = [array('f',[0.]), 'L2_HADtoEMEnergy/F']
            self.variables[ 'L2_EnergyTonCells' ] = [array('f',[0.]), 'L2_EnergyTonCells/F']


    # ---------------------------------------------------
    def L1Variables(self):
        # ---> EF/L1 matching variables
        self.variables[ 'L1_ismatched'    ] = [ array('i',[0]),   'L1_ismatched/I' ]
        self.variables[ 'L1_DeltaR_L1_L2']  = [ array('f',[-9999.]),  'L1_DeltaR_L1_L2/F' ]
        if 'basic' in self.include:
            # ---> Basics tau kinematics at EventFilter level
            self.variables[ 'L1_p'  ] = [ array('f',[0.]),  'L1_p/F'   ]
            self.variables[ 'L1_pt' ] = [ array('f',[0.]),  'L1_pt/F'  ]
            self.variables[ 'L1_eta'] = [ array('f',[0.]),  'L1_eta/F' ]
            self.variables[ 'L1_phi'] = [ array('f',[0.]),  'L1_phi/F' ]




        


