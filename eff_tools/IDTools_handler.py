# ---> cut values list
cutvals = {}
# ---> BDT for EF taus
cutvals["bdt_andrew_1"]  = {'all':0.1360}
cutvals["bdt_andrew_2"]  = {'1p':9.211731e-02,'mp':1.032295e-01}
cutvals["bdt_andrew_3"]  = {'1p':1.000061e-01,'mp':1.133631e-01}
cutvals["bdt_quentin_1"] = {'1p':0.5028,'mp':0.5885}
cutvals["bdt_quentin_2"] = {'1p_0n':0.515,'1p_Xn':0.525,'mp_0n':0.665,'mp_Xn':0.635}

# ---> BDT for offline taus
cutvals["bdt_presel_3var"]                 = {'all':0.371004164981}# mu = 60
cutvals["bdt_presel_5var"]                 = {'all':0.3781426086}# mu = 60
cutvals["bdt_presel_fullvarlist"]          = {'all':0.395105310024}# mu = 60
cutvals["bdt_presel_fullvarlist_michel1"]  = {'all':0.388201399769}# mu = 60
cutvals["bdt_presel_fullvarlist_michel2"]  = {'all':0.391921428212}# mu = 60
cutvals["bdt_presel_fullvarlist_michel3"]  = {'all':0.389722714377}# mu = 60

cutvals["bdt_full_quentin_new"] = {'1p':0.446326885537,'mp':0.558639397958}


# --> List of necessary inputs for the ID decision tool (BDT_name,weight file, variable list file, cut value)
inputs_lists = {}
inputs_lists["bdt_presel_3var"] = {}
inputs_lists["bdt_presel_3var"]["all"]=[ "BDT","weights_prod/presel_3var_all_14TeV_offline_BDT_AlekseyParams.weights.xml",
                                         'variables_list/variables_quentin_bdt_preselection.txt', cutvals["bdt_presel_3var"]["all"] ]
inputs_lists["bdt_presel_5var"] = {}
inputs_lists["bdt_presel_5var"]["all"]=[ "BDT","weights_prod/presel_5var_all_14TeV_offline_BDT_AlekseyParams.weights.xml",
                                         'variables_list/variables_quentin_bdt_preselection_5var.txt', cutvals["bdt_presel_5var"]["all"] ]

inputs_lists["bdt_presel_fullvarlist"] = {}
inputs_lists["bdt_presel_fullvarlist"]["all"]=[ "BDT","weights_prod/presel_fullvarlist_all_14TeV_offline_BDT_AlekseyParams.weights.xml",
                                                'variables_list/variables_quentin_bdt_preselection_fullvariablelist.txt', cutvals["bdt_presel_fullvarlist"]["all"] ]

inputs_lists["bdt_presel_fullvarlist_michel1"] = {}
inputs_lists["bdt_presel_fullvarlist_michel1"]["all"]=[ "BDT","weights_prod/presel_fullvarlist_michel1_all_14TeV_offline_BDT_AlekseyParams.weights.xml",
                                                       'variables_list/variables_quentin_bdt_preselection_fullvariablelist_michel1.txt', cutvals["bdt_presel_fullvarlist_michel1"]["all"] ]

inputs_lists["bdt_presel_fullvarlist_michel2"] = {}
inputs_lists["bdt_presel_fullvarlist_michel2"]["all"]=[ "BDT","weights_prod/presel_fullvarlist_michel2_all_14TeV_offline_BDT_AlekseyParams.weights.xml",
                                                       'variables_list/variables_quentin_bdt_preselection_fullvariablelist_michel2.txt',
                                                        cutvals["bdt_presel_fullvarlist_michel2"]["all"] ]

inputs_lists["bdt_presel_fullvarlist_michel3"] = {}
inputs_lists["bdt_presel_fullvarlist_michel3"]["all"]=[ "BDT","weights_prod/presel_fullvarlist_michel3_all_14TeV_offline_BDT_AlekseyParams.weights.xml",
                                                       'variables_list/variables_quentin_bdt_preselection_fullvariablelist_michel3.txt',
                                                        cutvals["bdt_presel_fullvarlist_michel3"]["all"] ]

# # --> andrew_1
# inputs_lists["bdt_andrew_1"] = {}
# inputs_lists["bdt_andrew_1"]["all"]=[ "BDT","weights/bdt_andrew_1/TMVAClassification_BDT.weights_Andrew_BDT1.xml",
#                                       'variables_list/variables_andrew_bdt1.txt', cutvals["bdt_andrew_1"]["all"] ]
# # --> andrew_2
# inputs_lists["bdt_andrew_2"] = {}
# inputs_lists["bdt_andrew_2"]["1p"] = ["BDT","weights/bdt_andrew_2/case1_sp.xml","variables_list/variables_andrew_bdt2_1p.txt",cutvals["bdt_andrew_2"]["1p"]]
# inputs_lists["bdt_andrew_2"]["mp"] = ["BDT","weights/bdt_andrew_2/case1_mp.xml","variables_list/variables_andrew_bdt2_mp.txt",cutvals["bdt_andrew_2"]["1p"]]

# # --> andrew_3
# inputs_lists["bdt_andrew_3"] = {}
# inputs_lists["bdt_andrew_3"]["1p"] = ["BDT","weights/bdt_andrew_3/case2_sp.xml","variables_list/variables_andrew_bdt3_1p.txt",cutvals["bdt_andrew_3"]["1p"]]
# inputs_lists["bdt_andrew_3"]["mp"] = ["BDT","weights/bdt_andrew_3/case2_mp.xml","variables_list/variables_andrew_bdt3_mp.txt",cutvals["bdt_andrew_3"]["mp"]]

# # --> quentin_1
# inputs_lists["bdt_quentin_1"] = {}
# inputs_lists["bdt_quentin_1"]["1p"] = ["BDT","weights/quentin_1p_14TeV_BDT.weights.xml","variables_list/variables_quentin_bdt_1_1p.txt",cutvals["bdt_quentin_1"]["1p"]]
# inputs_lists["bdt_quentin_1"]["mp"] = ["BDT","weights/quentin_3p_14TeV_BDT.weights.xml","variables_list/variables_quentin_bdt_1_3p.txt",cutvals["bdt_quentin_1"]["mp"]]

# # --> quentin_2
# inputs_lists["bdt_quentin_2"] = {}
# inputs_lists["bdt_quentin_2"]["1p_0n"] = ["BDT","weights/quentin_1p_0n_14TeV_BDT.weights.xml","variables_list/variables_quentin_bdt_1_1p.txt",cutvals["bdt_quentin_2"]["1p_0n"]]
# inputs_lists["bdt_quentin_2"]["1p_Xn"] = ["BDT","weights/quentin_1p_Xn_14TeV_BDT.weights.xml","variables_list/variables_quentin_bdt_1_1p.txt",cutvals["bdt_quentin_2"]["1p_Xn"]]
# inputs_lists["bdt_quentin_2"]["mp_0n"] = ["BDT","weights/quentin_3p_0n_14TeV_BDT.weights.xml","variables_list/variables_quentin_bdt_1_3p.txt",cutvals["bdt_quentin_2"]["mp_0n"]]
# inputs_lists["bdt_quentin_2"]["mp_Xn"] = ["BDT","weights/quentin_3p_Xn_14TeV_BDT.weights.xml","variables_list/variables_quentin_bdt_1_3p.txt",cutvals["bdt_quentin_2"]["mp_Xn"]]

# # --> quentin_new
inputs_lists["bdt_full_quentin_new"] = {}
inputs_lists["bdt_full_quentin_new"]["1p"] = ["BDT","weights_prod/quentin_new_1p_14TeV_offline_BDT_AlekseyParams.weights.xml",
                                              "variables_list/variables_quentin_bdt_1_1p_offline.txt",cutvals["bdt_full_quentin_new"]["1p"]]
inputs_lists["bdt_full_quentin_new"]["mp"] = ["BDT","weights_prod/quentin_new_3p_14TeV_offline_BDT_AlekseyParams.weights.xml",
                                              "variables_list/variables_quentin_bdt_1_3p_offline.txt",cutvals["bdt_full_quentin_new"]["mp"]]






