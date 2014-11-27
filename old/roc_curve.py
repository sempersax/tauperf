import argparse
import ROOT
from array import array
import sys

import fileinput
import AnalysisTools
import AtlasStyle
from auxiliary      import aux
from DecisionTool   import DecisionTool
from TauIDTool      import TauIDTool
from EFTau_Category import Category
from eff_plotting_tools import RejectionCurve
from eff_plotting_tools import RoCcurve

import IDTools_handler

#-----------------------------------------------------------

#--> Receive and parse argument
parser = argparse.ArgumentParser()
parser.add_argument("input_file_signal", help="the list (txt file) of the input signal files")
parser.add_argument("input_file_bkg", help="the list (txt file) of the input bkg files")
parser.add_argument("output_file", help="the name of the output root file")
parser.add_argument("sample_type", help="Sample type (8TeV,14TeV)")
parser.add_argument("-N","--Nentries", type=int,default=-1,help="Specify the number of events use to run")
args = parser.parse_args()
parser.print_usage()

# --> Import data files
tauCell_s=ROOT.TChain('tauCell_test')
tauCell_b=ROOT.TChain('tauCell_test')

for ifile in open(args.input_file_signal):
    tauCell_s.Add(ifile.strip())
for ifile in open(args.input_file_bkg):
    tauCell_b.Add(ifile.strip())

# Get number of entries in data file
print 'The signal input chain contains ',tauCell_s.GetEntries(),' entries'
print 'The bkg input chain contains ',tauCell_b.GetEntries(),' entries'
if args.Nentries==-1:
    entries_s = tauCell_s.GetEntries()
    entries_b = tauCell_b.GetEntries()
else:
    entries_s = args.Nentries
    entries_b = args.Nentries
print 'The loops will use ',entries_s,' for signal and ',entries_b,' entries for bkg'

# --> Declaration of the different id tools
ID_Tools_s = {}
ID_Tools_b = {}
for input in IDTools_handler.inputs_lists:
    ID_Tools_s[input]  = TauIDTool(tauCell_s,IDTools_handler.inputs_lists[input])
    ID_Tools_b[input]  = TauIDTool(tauCell_b,IDTools_handler.inputs_lists[input])

# --> Plotting category
plot_cat = aux.prong_cat+aux.prongpi0_cat+["all"]

# --> Declaration of the list (python dictionary) of TEfficiency objects
Efficiencies_s = {}
for tool in ID_Tools_s:
    Efficiencies_s[tool] = {}
    Efficiencies_s[tool][tool+"_all"] = ROOT.TEfficiency( "Efficiency_"+tool+"_all_bdt_cuts_signal","",
                                                          len(aux.bins[tool][0])-1 ,aux.bins[tool][0] )
    for cat in aux.prong_cat+aux.prongpi0_cat:
        Efficiencies_s[tool][tool+"_"+cat] = ROOT.TEfficiency( "Efficiency_"+tool+"_"+cat+"_bdt_cuts_signal","",
                                                               len(aux.bins[tool][0])-1 ,aux.bins[tool][0] )

#--------------------------------------------------------------
#-------------> LOOP OVER THE EVENTS OF THE INPUT TREE --------
#--------------------------------------------------------------
for entry in xrange(entries_s):
    AnalysisTools.Processing(entry,entries_s,float(entries_s)/100.)
    tauCell_s.GetEntry( entry )

    if tauCell_s.EF_ismatched != 1: continue
    if tauCell_s.L2_ismatched != 1: continue
    if tauCell_s.L1_ismatched != 1: continue
    if tauCell_s.truth_ismatched!=1:continue

    # --> tauNoCut is not implemented in 14 TeV MC
    # --> L2_tau20_medium is the 'backup' solution for now
    isTrigger = False
    if '14TeV' in args.sample_type:
        isTrigger = tauCell_s.L2_L2_tau20_medium
    elif '8TeV' in args.sample_type:
        isTrigger = tauCell_s.EF_EF_tauNoCut
        
    if isTrigger !=1: continue

    tau_cat = Category(tauCell_s)
    for tool in ID_Tools_b:
        for cutval in aux.bins[tool][0]:
            cutvals_tmp = {}
            for key in IDTools_handler.cutvals[tool].keys(): cutvals_tmp[key]=cutval
            ID_Tools_s[tool].SetCutValues(cutvals_tmp)
            Efficiencies_s[tool][tool+"_all"].Fill( ID_Tools_s[tool].Decision(),cutval )
            for cat in tau_cat.prong_cat:
                Efficiencies_s[tool][tool+"_"+cat].Fill( ID_Tools_s[tool].Decision(),cutval ) 
            for cat in tau_cat.prongpi0_cat:
                Efficiencies_s[tool][tool+"_"+cat].Fill( ID_Tools_s[tool].Decision(),cutval ) 

#--------------------------------------------------------------
#-------------> END OF THE LOOP OVER THE EVENTS        --------
#--------------------------------------------------------------

# --> Declaration of the list (python dictionary) of TEfficiency objects
Efficiencies_b = {}
for tool in ID_Tools_b:
    Efficiencies_b[tool] = {}
    Efficiencies_b[tool][tool+"_all"] = ROOT.TEfficiency( "Efficiency_"+tool+"all_bdt_cuts_bkg","",len(aux.bins[tool][0])-1 ,aux.bins[tool][0] )
    for cat in aux.prong_cat+aux.prongpi0_cat:
        Efficiencies_b[tool][tool+"_"+cat] = ROOT.TEfficiency( "Efficiency_"+tool+"_"+cat+"_bdt_cuts_bkg","",len(aux.bins[tool][0])-1 ,aux.bins[tool][0] )

#--------------------------------------------------------------
#-------------> LOOP OVER THE EVENTS OF THE INPUT TREE --------
#--------------------------------------------------------------
for entry in xrange(entries_b):
    AnalysisTools.Processing(entry,entries_b,float(entries_b)/100.)
    tauCell_b.GetEntry( entry )

    if tauCell_b.EF_ismatched != 1: continue
    if tauCell_b.L2_ismatched != 1: continue
    if tauCell_b.L1_ismatched != 1: continue

    # --> tauNoCut is not implemented in 14 TeV MC
    # --> L2_tau20_medium is the 'backup' solution for now
    isTrigger = False
    if '14TeV' in args.sample_type:
        isTrigger = tauCell_b.L2_L2_tau20_medium
    elif '8TeV' in args.sample_type:
        isTrigger = tauCell_b.EF_EF_tauNoCut
        
    if isTrigger !=1: continue
    tau_cat = Category(tauCell_b)
    for tool in ID_Tools_b:
        for cutval in aux.bins[tool][0]:
            cutvals_tmp = {}
            for key in IDTools_handler.cutvals[tool].keys(): cutvals_tmp[key]=cutval
            ID_Tools_b[tool].SetCutValues(cutvals_tmp)
            Efficiencies_b[tool][tool+"_all"].Fill( ID_Tools_b[tool].Decision(),cutval )
            for cat in tau_cat.prong_cat:
                Efficiencies_b[tool][tool+"_"+cat].Fill( ID_Tools_b[tool].Decision(),cutval ) 
            for cat in tau_cat.prongpi0_cat:
                Efficiencies_b[tool][tool+"_"+cat].Fill( ID_Tools_b[tool].Decision(),cutval ) 

#--------------------------------------------------------------
#-------------> END OF THE LOOP OVER THE EVENTS        --------
#--------------------------------------------------------------

#--------------------------------------------------------------
#-------------> ROC CURVE CALCULATION                  --------
#--------------------------------------------------------------
roc_curves = {}
for tool in Efficiencies_s:
    for var in Efficiencies_s[tool]:
        roc_curves[tool+"_"+var] = RoCcurve(Efficiencies_s[tool][var],Efficiencies_b[tool][var])
        roc_curves[tool+"_"+var].SetName("roc_"+var)
        roc_curves[tool+"_"+var].SetTitle("roc_"+var)
        
#--------------------------------------------------------------
#-------------> EFFICIENCY STORING       - --------------------
#--------------------------------------------------------------

output=ROOT.TFile(args.output_file,"recreate")
root_directory = output.CurrentDirectory()
for tool in Efficiencies_s:
    directory = output.mkdir(tool)
    directory.cd()
    for var in Efficiencies_s[tool]:
        Efficiencies_s[tool][var].Write()
        Efficiencies_b[tool][var].Write()
        roc_curves[tool+"_"+var].Write()
    root_directory.cd()
output.Close()

        
