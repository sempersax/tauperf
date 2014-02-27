import ROOT
import sys

from helpers import ordereddict
from eff_tools.auxiliary import aux
from helpers import AnalysisTools 
from eff_tools.eff_plotting_tools import RejectionCurve
from eff_tools.eff_plotting_tools import RoCcurve
from eff_tools.eff_plotting_tools import SvsB_Perf_Canvas
from eff_tools.eff_plotting_tools import DiscriVar_Canvas
from eff_tools.eff_plotting_tools import EfficiencyPlot

from helpers import AtlasStyle
AtlasStyle.SetAtlasStyle()
ROOT.gROOT.SetBatch()



#-----> Open the rootfiles
files = ordereddict.OrderedDict()


files["Z_14TeV_all"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_all_v12.root")
files["Z_14TeV_mu20"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_mu20_v12.root")
files["Z_14TeV_mu40"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_mu40_v12.root")
files["Z_14TeV_mu60"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_mu60_v12.root")
files["Z_14TeV_mu80"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_mu80_v12.root")
files["bkg_JF_14TeV_mu40"] = ROOT.TFile("efficiencies/efficiencies_presel_JF17_14TeV_mu40_v12.root")
files["bkg_JF_14TeV_mu60"] = ROOT.TFile("efficiencies/efficiencies_presel_JF17_14TeV_mu60_v12.root")
files["bkg_JF_14TeV_all"]  = ROOT.TFile("efficiencies/efficiencies_presel_JF17_14TeV_all_v12.root")



files_sb = ordereddict.OrderedDict()
# files_sig_bkg["8TeV"]       = {'sig':files["Z_8TeV"]      ,'bkg':files["bkg_data_8TeV"] }
files_sb["14TeV_mu40"] = {'sig':files["Z_14TeV_mu40"],'bkg':files["bkg_JF_14TeV_mu40"] }
files_sb["14TeV_mu60"] = {'sig':files["Z_14TeV_mu60"],'bkg':files["bkg_JF_14TeV_mu60"] }
files_sb["14TeV_all"]  = {'sig':files["Z_14TeV_all"],'bkg':files["bkg_JF_14TeV_all"] }



#--------------------------------------------------------------
#-------------> EFFICIENCY PLOTTING        --------------------
#--------------------------------------------------------------


categories = ["all"]
bdt_name   = ["bdt_presel_3var"]
bdt_name   += ["bdt_presel_5var"]
bdt_name   += ["bdt_presel_fullvarlist"]
bdt_name   += ["bdt_presel_fullvarlist_michel1"]
bdt_name   += ["bdt_presel_fullvarlist_michel2"]

style_config = {}
style_config["bdt_presel"]    = [ ROOT.kBlack, ROOT.kFullCircle ]
style_config["bdt_presel_fullvarlist"]    = [ ROOT.kBlack, ROOT.kFullSquare ]

#--------------------------------------------------------------
#-------------> EFFICIENCY PLOTTING        --------------------
#--------------------------------------------------------------
# for file in files:
#     for var in aux.bins:
#         if 'bdt' in var: continue
#         if 'EF' in var: continue
#         for cat in categories:
#             print "---> plot "+var+" in category "+cat
#             teff_list = ordereddict.OrderedDict()
#             for name in bdt_name:
#                 teff_list[name] = [ files[file].Get(name+'/Efficiency_'+name+'_'+var+'_'+cat),
#                                     style_config[name][0],style_config[name][1], aux.bins[var][1],"Efficiency", name ]
#             if "bkg" in file:
#                 for name, eff in teff_list.iteritems():
#                     eff[0] = RejectionCurve(eff[0])
#                     eff[4] = "Rejection = 1 - #epsilon"

#             # ---> Efficiency/rejection plot supperposed for various BDT
#             EP = EfficiencyPlot('plot_'+var+'_'+cat+"_"+file,var+'_'+cat+'_'+file,teff_list,"")




##------------------------------------------------------------------------------
##      ---->  SIGNAL/BKG EFFICIENCY COMPARISON PLOTS
##------------------------------------------------------------------------------

plots = {}
for var in aux.bins:
    if 'bdt' in var: continue
    if 'EF' in var: continue
    if 'npv' in var: continue
    for cat in categories:
        print "---> plot "+var+" in category "+cat
        for name in bdt_name:
            ## --------------------------------------------
            sig_file = ordereddict.OrderedDict()
            sig_file["mu20"]  = [ files["Z_14TeV_mu20"].Get( name+'/Efficiency_'+name+'_'+var+'_'+cat  ),2,23,aux.bins[var][1],"Efficiency","mu=20"]
            sig_file["mu40"]  = [ files["Z_14TeV_mu40"].Get( name+'/Efficiency_'+name+'_'+var+'_'+cat ),3,24,aux.bins[var][1],"Efficiency","mu=40"]
            sig_file["mu60"]  = [ files["Z_14TeV_mu60"].Get( name+'/Efficiency_'+name+'_'+var+'_'+cat ),4,25,aux.bins[var][1],"Efficiency","mu=60"]
            sig_file["mu80"]  = [ files["Z_14TeV_mu80"].Get( name+'/Efficiency_'+name+'_'+var+'_'+cat ),ROOT.kViolet,26,aux.bins[var][1],"Efficiency","mu=80"]
            plots["pileup_signal_"+var+"_"+cat+"_"+name] = EfficiencyPlot("plot_sig_"+var+"_"+cat+"_"+name,"signal "+var+" "+cat,sig_file,"")
            for pair in files_sb:
                # --------------------------------------------
                print "svsb_"+pair+"_"+var+"_"+cat+"_"+name
                plots["svsb_"+pair+"_"+var+"_"+cat+"_"+name] = SvsB_Perf_Canvas( files_sb[pair]['sig'].Get(name+'/Efficiency_'+name+'_'+var+'_'+cat),
                                                                                 files_sb[pair]['bkg'].Get(name+'/Efficiency_'+name+'_'+var+'_'+cat),
                                                                                 aux.bins[var][1])


        plots["svsb_14TeV_mu60_"+var+"_"+cat+"_bdt_presel_3var"].cd()
        rej_2 = RejectionCurve(files_sb["14TeV_mu60"]['bkg'].Get('bdt_presel_5var/Efficiency_bdt_presel_5var_'+var+'_'+cat))
        rej_3 = RejectionCurve(files_sb["14TeV_mu60"]['bkg'].Get('bdt_presel_fullvarlist/Efficiency_bdt_presel_fullvarlist_'+var+'_'+cat))
        rej_4 = RejectionCurve(files_sb["14TeV_mu60"]['bkg'].Get('bdt_presel_fullvarlist_michel1/Efficiency_bdt_presel_fullvarlist_michel1_'+var+'_'+cat))
        rej_5 = RejectionCurve(files_sb["14TeV_mu60"]['bkg'].Get('bdt_presel_fullvarlist_michel2/Efficiency_bdt_presel_fullvarlist_michel2_'+var+'_'+cat))
        rej_2.SetMarkerColor(ROOT.kGreen)
        rej_2.SetLineColor(ROOT.kGreen)
        rej_2.SetMarkerStyle(ROOT.kFullTriangleUp)
        rej_2.SetLineStyle(ROOT.kDashed)
        rej_3.SetMarkerColor(ROOT.kViolet)
        rej_3.SetLineColor(ROOT.kViolet)
        rej_3.SetMarkerStyle(ROOT.kFullTriangleUp)
        rej_3.SetLineStyle(ROOT.kDashed)
        rej_4.SetMarkerColor(ROOT.kBlue)
        rej_4.SetLineColor(ROOT.kBlue)
        rej_4.SetMarkerStyle(ROOT.kFullTriangleUp)
        rej_4.SetLineStyle(ROOT.kDashed)
        rej_5.SetMarkerColor(ROOT.kGray+1)
        rej_5.SetLineColor(ROOT.kGray+1)
        rej_5.SetMarkerStyle(ROOT.kFullTriangleUp)
        rej_5.SetLineStyle(ROOT.kDashed)
        rej_2.Draw("sameP")
        rej_3.Draw("sameP")
        rej_4.Draw("sameP")
        rej_5.Draw("sameP")
        
for key,plot in plots.iteritems():
    plot.SaveAs("./plots/"+key+".eps")

