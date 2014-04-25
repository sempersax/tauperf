import sys
import fileinput
from array import array

import ROOT

from rootpy.extern import argparse

from eff_tools import IDTools_handler
from eff_tools.auxiliary      import aux
from eff_tools.TauIDTool      import TauIDTool
from eff_tools.EFTau_Category import Category

#-----------------------------------------------------------

#--> Receive and parse argument
parser = argparse.ArgumentParser()
parser.add_argument("input_file_list", help="the list (txt file) of the input files")
parser.add_argument("output_file", help="the name of the output root file")
parser.add_argument("sample_type", help="Sample type (signal_8TeV,signal_14TeV,background_14TeV,background_data_8TeV)")
parser.add_argument("-N","--Nentries", type=int,default=-1,help="Specify the number of events use to run")
args = parser.parse_args()
parser.print_usage()

# --> Import data files
tauCell=ROOT.TChain('tauCell_test')

inputfile= open(args.input_file_list)
for ifile in inputfile:
    tauCell.Add(ifile.strip())

# Get number of entries in data file
print 'The input chain contains ',tauCell.GetEntries(),' entries'
if args.Nentries==-1:
    entries = tauCell.GetEntries()
else:
    entries = args.Nentries
print 'The loop will use ',entries,' entries'


# andrew_cutval = 1.516797e-01
# my_cutval = 0.135


# ---> cut values list
cutvals = IDTools_handler.cutvals


# --> Declaration of the different id tools
ID_Tools = {}
for input in IDTools_handler.inputs_lists:
    ID_Tools[input]  = TauIDTool(tauCell,IDTools_handler.inputs_lists[input])

# --> Plotting category
# plot_cat = aux.prong_cat+aux.prongpi0_cat+aux.eta_cat+aux.mu_cat+["all"]
plot_cat = aux.prong_cat+aux.prongpi0_cat+["all"]

# --> Declaration of the list (python dictionary) of TEfficiency objects
Efficiencies = {}
Efficiencies_old = {}

for tool in ID_Tools:
    Efficiencies[tool] = {}

for cat in plot_cat:
    for var in  aux.bins:
        if "bdt" in var: continue
        Efficiencies_old[var+"_"+cat] = ROOT.TEfficiency( "Efficiency_old_"+var+"_"+cat ,"", len(aux.bins[var][0])-1 ,aux.bins[var][0] )
        for tool in ID_Tools:
            Efficiencies[tool][var+"_"+cat] = ROOT.TEfficiency( "Efficiency_"+tool+"_"+var+"_"+cat,"",len(aux.bins[var][0])-1 ,aux.bins[var][0] )



# --> Number of tracks histogram (to combine efficiency values)
h_nTracks = ROOT.TH1F("h_nTracks","h_nTracks",10,0,10)
h_BDT     =  {}
for cutval in cutvals:
    h_BDT[cutval] = ROOT.TH1F("h_BDT_"+cutval,"h_BDT",len(aux.bins[cutval][0])-1 ,aux.bins[cutval][0])

#--------------------------------------------------------------
#-------------> LOOP OVER THE EVENTS OF THE INPUT TREE --------
#--------------------------------------------------------------
for entry in xrange(entries):
    tauCell.GetEntry( entry )

    if tauCell.EF_ismatched != 1: continue
    if tauCell.L2_ismatched != 1: continue
    if tauCell.L1_ismatched != 1: continue
    if 'signal' in args.sample_type and tauCell.truth_ismatched!=1:continue

    #     if tauCell.BDTmedium != 1 : continue

    # --> tauNoCut is not implemented in 14 TeV MC
    # --> L2_tau20_medium is the 'backup' solution for now
    isTrigger = False
    if '14TeV' in args.sample_type:
        isTrigger = tauCell.L2_L2_tau20_medium
    elif '8TeV' in args.sample_type:
        isTrigger = tauCell.EF_EF_tauNoCut
        
    if isTrigger !=1: continue

    h_nTracks.Fill ( tauCell.nTracks )
    tau_cat = Category(tauCell)
    for tool in ID_Tools:
        ID_Tools[tool].SetCutValues(cutvals[tool])
        h_BDT[tool].Fill (ID_Tools[tool].BDTScore())
        # --> Fill TEfficiency graph if it belongs to the intersection of the tau_cat and plot_cat
        for cat in (set(tau_cat.categories+["all"])&set(plot_cat)):
            for var in  aux.bins:
                if "bdt" in var: continue
                Efficiencies[tool][var+"_"+cat].Fill( ID_Tools[tool].Decision(), getattr(tauCell,var) )

    for cat in (set(tau_cat.categories+["all"])&set(plot_cat)):
        for var in  aux.bins:
            if "bdt" in var: continue
            Efficiencies_old[var+"_"+cat].Fill( tauCell.EF_EF_tau20_medium1,getattr(tauCell,var))


#--------------------------------------------------------------
#-------------> END OF THE LOOP OVER THE EVENTS        --------
#--------------------------------------------------------------



#--------------------------------------------------------------
#-------------> EFFICIENCY STORING       - --------------------
#--------------------------------------------------------------

output = ROOT.TFile(args.output_file,"recreate")
root_directory = output.CurrentDirectory()
for tool in Efficiencies:
    directory = output.mkdir(tool)
    directory.cd()
    for var in Efficiencies[tool]:
        Efficiencies[tool][var].Write()
    h_BDT[tool].Write()
    root_directory.cd()

old_directory = output.mkdir("old")
old_directory.cd()
for var in Efficiencies_old: Efficiencies_old[var].Write()
root_directory.cd()
h_nTracks.Write()
output.Close()

        
