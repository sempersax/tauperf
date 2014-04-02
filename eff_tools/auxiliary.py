from array import array

class aux:
    """ A class to handle auxiliary objects used in various modules """
    # ---> binning
    bins = {}
    bins["mu"]          = [array('d', [0 ,5 ,10,15,20,25,30,35,40 ,45 ,50,55,60,65,70,75,80,85,90 ]), "Average Interaction Per Bunch Crossing"  ]
    bins["off_pt"]          = [array('d', [20e3,22.5e3,25e3,27.5e3,30e3,32.5e3,35e3,37.5e3,40e3,42.5e3,45e3,50e3,60e3,70e3,80e3,100e3]), "Offline p_{T} [MeV]"]
#     bins["pt"]          = [ array('d', [10e3,15e3,16e3,17e3,18e3,19e3,20e3,21e3,22e3,23e3,24e3,25e3,26e3,27e3,28e3,30e3,32.5e3,35e3,37.5e3,40e3,50e3,60e3,80e3,100e3,150e3]),
#                             "Offline p_{T} [MeV]"]
#     bins["clbased_pt"]  = [ array('d', [10e3,15e3,16e3,17e3,18e3,19e3,20e3,21e3,22e3,23e3,24e3,25e3,26e3,27e3,28e3,30e3,32.5e3,35e3,37.5e3,40e3,50e3,60e3,80e3,100e3,150e3]),
#                             "Offline Cluster-Based p_{T} [MeV]" ]
    bins["off_eta"]         = [array('d', [-2.5,-2.25,-2.0,-1.6,-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2,1.6,2.0,2.25,2.5]), "#eta" ]
    bins["EF_pt"]       = [array('d', [20e3,22.5e3,25e3,27.5e3,30e3,32.5e3,35e3,37.5e3,40e3,42.5e3,45e3,50e3,60e3,70e3,80e3,100e3,120e3,140e3]), "Event Filter p_{T} [MeV]"]
    bins["npv"]         = [array('d', [0 ,5 ,10,15,20,25,30,35,40,45,50])   , "Number of Primary Vertices" ]
    bins["EF_eta"]      = [array('d', [-2.4,-2.0,-1.6,-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2,1.6,2.0,2.4]), "Event Filter #eta" ]

    N = 500
    bins["bdt_andrew_1"]   = [array('d',[2*x/float(10*N) for x in range(N)]), "BDT cut value" ]
    bins["bdt_andrew_2"]   = [array('d',[3*x/float(10*N) for x in range(N)]), "BDT cut value" ]
    bins["bdt_andrew_3"]   = [array('d',[3*x/float(10*N) for x in range(N)]), "BDT cut value" ]
    bins["bdt_quentin_1"]  = [array('d',[x/float(N) for x in range(N)])   , "BDT cut value" ]
    bins["bdt_quentin_2"]  = [array('d',[x/float(N) for x in range(N)])   , "BDT cut value" ]

    # ---> Categories 
    prong_cat    = ["1p","2p","3p","mp"]
    prongpi0_cat = ["1p_0n","2p_0n","3p_0n","mp_0n","1p_Xn","2p_Xn","3p_Xn","mp_Xn"]
    neutral_cat  = ["0n","1n","2n","Xn"]
    eta_cat      = ["central","endcap"]
    mu_cat       = ["low_mu","medium_mu","high_mu","crazy_mu"]


    cat_label = {}
    cat_label["1p"] = "1 prong" 
    cat_label["2p"] = "2 prongs"
    cat_label["3p"] = "3 prongs"
    cat_label["mp"] = "multi-prongs"
    cat_label["0n"] = "1 neutral" 
    cat_label["1n"] = "2 neutrals"
    cat_label["2n"] = "3 neutrals"
    cat_label["Xn"] = "X neutrals"
    cat_label["central"] = "|#eta|<1.37"
    cat_label["endcap"]  = "|#eta|>1.37" 
    cat_label["low_mu"]  = "<#mu> #in [0,20]"
    cat_label["medium_mu"] = "<#mu> #in [20,40]"
    cat_label["high_mu"] = "<#mu> #in [40,60]"
    cat_label["crazy_mu"] = "<#mu> #in [60,inf]"
    cat_label["all"] = ""
