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


from rootpy.plotting.style import set_style
set_style('ATLAS')
# from array import array

def GetGlobalEfficiency(t_eff):
    htot = t_eff.GetTotalHistogram()
    total_events=0
    total_eff=0
    for bin in xrange(htot.GetNbinsX()+1):
        total_events += htot.GetBinContent(bin)
        total_eff += htot.GetBinContent(bin)*t_eff.GetEfficiency(bin)
    total_eff *= 1./total_events
    return total_eff

class RoC_Canvas(ROOT.TCanvas):
    """ A class to plot RoC curves """

    def __init__(self,signal_file_st,bkg_file_st,cat):
        ROOT.TCanvas.__init__(self,signal_file_st+"_"+cat,signal_file_st+"_"+cat)
        signal_file = ROOT.TFile( signal_file_st )
        bkg_file    = ROOT.TFile( bkg_file_st    )
        rocs = ordereddict.OrderedDict()
        rocs['andrew_1']=[ RoCcurve( signal_file.Get( 'Efficiency_bdt_andrew_1_'+cat+'_bdt_cuts'),bkg_file.Get( 'Efficiency_bdt_andrew_1_'+cat+'_bdt_cuts')),
                           ROOT.kRed, 'Andrew BDT-1']
        rocs['andrew_2']=[ RoCcurve( signal_file.Get( 'Efficiency_bdt_andrew_2_'+cat+'_bdt_cuts'),bkg_file.Get( 'Efficiency_bdt_andrew_2_'+cat+'_bdt_cuts')),
                           ROOT.kBlue, 'Andrew BDT-2']
        rocs['andrew_3']=[ RoCcurve( signal_file.Get( 'Efficiency_bdt_andrew_3_'+cat+'_bdt_cuts'),bkg_file.Get( 'Efficiency_bdt_andrew_3_'+cat+'_bdt_cuts')),
                           ROOT.kGreen, 'Andrew BDT-3']
        rocs['quentin_1']=[RoCcurve( signal_file.Get('Efficiency_bdt_quentin_1_'+cat+'_bdt_cuts'),bkg_file.Get('Efficiency_bdt_quentin_1_'+cat+'_bdt_cuts')),
                           ROOT.kViolet,'Quentin BDT-1']
#         rocs['quentin_2']=[RoCcurve( signal_file.Get('Efficiency_bdt_quentin_2_'+cat+'_bdt_cuts'),bkg_file.Get('Efficiency_bdt_quentin_2_'+cat+'_bdt_cuts')),
#                            ROOT.kYellow,'Quentin BDT-2']
        self.roc_old = ROOT.TGraph()
        self.roc_old.SetPoint(0,GetGlobalEfficiency( signal_file.Get('Efficiency_old_EF_eta_'+cat)),
                              1-GetGlobalEfficiency( bkg_file.Get('Efficiency_old_EF_eta_'+cat)) )
        print 1-GetGlobalEfficiency( bkg_file.Get('Efficiency_old_EF_eta_'+cat))
        self.legend = ROOT.TLegend(0.14,0.18,0.55,0.40)
        self.legend.SetFillColor(0)
        self.label = ROOT.TLatex()
        self.label.SetNDC()
        for roc in rocs:
            rocs[roc][0].SetLineColor(rocs[roc][1])
            self.legend.AddEntry( rocs[roc][0] , rocs[roc][2], 'l' )
        self.legend.AddEntry( self.roc_old , "2012 BDT", 'p' )
        

        self.SetGridx()
        self.SetGridy()
        self.cd()
        for roc in rocs:
            if 'andrew_1' in roc: rocs[roc][0].Draw('AL')
            else: rocs[roc][0].Draw('SAMEL')
        self.roc_old.Draw('SAMEP')
        self.legend.Draw("same")
        self.label.DrawLatex( 0.75,0.9,cat)
        ROOT.gPad.Update()
        return None


#--------------------------------------------------------------
#-------------> EFFICIENCY PLOTTING        --------------------
#--------------------------------------------------------------


#-----> Open the rootfiles
files = ordereddict.OrderedDict()
# files["Z_8TeV"]            = ROOT.TFile("efficiencies/efficiencies_noroc_Ztautau_8TeV_v6.root")
# files["Z_14TeV_mu20"]      = ROOT.TFile("efficiencies/efficiencies_noroc_Ztautau_14TeV_mu20_v6.root")
# files["Z_14TeV_mu40"]      = ROOT.TFile("efficiencies/efficiencies_noroc_Ztautau_14TeV_mu40_v6.root")
# files["Z_14TeV_mu60"]      = ROOT.TFile("efficiencies/efficiencies_noroc_Ztautau_14TeV_mu60_v6.root")
# files["Z_14TeV_mu80"]      = ROOT.TFile("efficiencies/efficiencies_noroc_Ztautau_14TeV_mu80_v6.root")
# files["Z_14TeV_all"]      = ROOT.TFile("efficiencies/efficiencies_noroc_Ztautau_14TeV_all_v6.root")
# files["bkg_JF_14TeV_mu40"] = ROOT.TFile("efficiencies/efficiencies_noroc_JF17_14TeV_mu40_v6.root")
# files["bkg_JF_14TeV_mu60"] = ROOT.TFile("efficiencies/efficiencies_noroc_JF17_14TeV_mu60_v6.root")
# files["bkg_data_8TeV"]     = ROOT.TFile("efficiencies/efficiencies_data_v6.root")


files["Z_14TeV_all"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_all_v10.root")
files["Z_14TeV_mu20"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_mu20_v10.root")
files["Z_14TeV_mu40"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_mu40_v10.root")
files["Z_14TeV_mu60"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_mu60_v10.root")
files["Z_14TeV_mu80"]      = ROOT.TFile("efficiencies/efficiencies_presel_Ztautau_14TeV_mu80_v10.root")
files["bkg_JF_14TeV_mu40"] = ROOT.TFile("efficiencies/efficiencies_presel_JF17_14TeV_mu40_v10.root")
files["bkg_JF_14TeV_mu60"] = ROOT.TFile("efficiencies/efficiencies_presel_JF17_14TeV_mu60_v10.root")
files["bkg_JF_14TeV_all"]  = ROOT.TFile("efficiencies/efficiencies_presel_JF17_14TeV_all_v10.root")



files_sig_bkg = ordereddict.OrderedDict()
# files_sig_bkg["8TeV"]       = {'sig':files["Z_8TeV"]      ,'bkg':files["bkg_data_8TeV"] }
files_sb["14TeV_mu40"] = {'sig':files["Z_14TeV_mu40"],'bkg':files["bkg_JF_14TeV_mu40"] }
files_sb["14TeV_mu60"] = {'sig':files["Z_14TeV_mu60"],'bkg':files["bkg_JF_14TeV_mu60"] }
files_sb["14TeV_all"]  = {'sig':files["Z_14TeV_all"],'bkg':files["bkg_JF_14TeV_all"] }





# categories = aux.prong_cat+aux.eta_cat+aux.mu_cat+["all"]
categories = ["all"]
# bdt_name   = ["bdt_andrew_1","bdt_quentin_1","bdt_quentin_2","old"]
# bdt_name   = ["bdt_andrew_1","bdt_quentin_1","bdt_quentin_2"]
# bdt_name   = ["bdt_andrew_1","old"]
bdt_name   = ["bdt_presel"]

style_config = {}
style_config["bdt_andrew_1"]  = [ ROOT.kRed,    ROOT.kFullSquare ]
style_config["bdt_andrew_2"]  = [ ROOT.kBlue,   ROOT.kFullSquare ]
style_config["bdt_andrew_3"]  = [ ROOT.kGreen,  ROOT.kFullSquare ]
style_config["bdt_quentin_1"] = [ ROOT.kViolet, ROOT.kFullSquare ]
style_config["bdt_quentin_2"] = [ ROOT.kOrange , ROOT.kFullSquare ]
style_config["old"]           = [ ROOT.kBlack,  ROOT.kFullCircle ]
style_config["bdt_presel"]    = [ ROOT.kBlack, ROOT.kBlack ]

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
##      ----> ROC CANVAS AND SIGNAL/BKG EFFICIENCY COMPARISON
##------------------------------------------------------------------------------
# roc_file = ROOT.TFile("rocs/roc_Ztautau_14TeV_mu40_v6.root")
# roc_cat = ["1p_0n","1p_Xn","3p_0n","3p_Xn"]
# for cat in roc_cat:
#     rocs = {}
#     for name in bdt_name:
#         rocs[name] = roc_file.Get(name+'/roc_'+name+'_'+cat)
#         rocs[name].SetLineColor(style_config[name][0])
#         rocs[name].SetLineWidth(4)
#     canvas_roc = ROOT.TCanvas()
#     canvas_roc.SetGridx()
#     canvas_roc.SetGridy()
#     canvas_roc.cd()
#     rocs['bdt_andrew_1'].Draw('AL')
#     rocs['bdt_quentin_1'].Draw('SAMEL')
#     rocs['bdt_quentin_2'].Draw('SAMEL')



# for pair in files_sb:
#     RoC_Canvas( files_sb[pair]['sig'].GetName(),files_sb[pair]['bkg'].GetName(),'1p' )
#     RoC_Canvas( files_sb[pair]['sig'].GetName(),files_sb[pair]['bkg'].GetName(),'3p' )
#     SvsB = {}
#     SvsB["1p"] = SvsB_Perf_Canvas( files_sb[pair]['sig'].Get('Efficiency_bdt_quentin_1_1p_bdt_cuts'),
#                                    files_sb[pair]['bkg'].Get('Efficiency_bdt_quentin_1_1p_bdt_cuts') )

#     SvsB["1p"].cd()
#     cut_line_1p = ROOT.TLine(SvsB["1p"].GetOptimalCutValue(1-GetGlobalEfficiency( files_sb[pair]['bkg'].Get('Efficiency_old_EF_eta_1p'))),0,
#                              SvsB["1p"].GetOptimalCutValue(1-GetGlobalEfficiency( files_sb[pair]['bkg'].Get('Efficiency_old_EF_eta_1p'))),1 )
#     cut_line_1p.Draw("same")
#     SvsB["3p"] = SvsB_Perf_Canvas( files_sb[pair]['sig'].Get('Efficiency_bdt_quentin_1_3p_bdt_cuts'),
#                                    files_sb[pair]['bkg'].Get('Efficiency_bdt_quentin_1_3p_bdt_cuts') )
#     SvsB["3p"].cd()
#     cut_line_3p = ROOT.TLine(SvsB["3p"].GetOptimalCutValue(1-GetGlobalEfficiency( files_sb[pair]['bkg'].Get('Efficiency_old_EF_eta_3p'))),0,
#                              SvsB["3p"].GetOptimalCutValue(1-GetGlobalEfficiency( files_sb[pair]['bkg'].Get('Efficiency_old_EF_eta_3p'))),1 )
#     cut_line_3p.Draw("same")
#     RoC_Canvas( files_sb[pair]['sig'].GetName(),files_sb[pair]['bkg'].GetName(),'1p_0n' )
#     RoC_Canvas( files_sb[pair]['sig'].GetName(),files_sb[pair]['bkg'].GetName(),'1p_Xn' )
#     RoC_Canvas( files_sb[pair]['sig'].GetName(),files_sb[pair]['bkg'].GetName(),'3p_0n' )
#     RoC_Canvas( files_sb[pair]['sig'].GetName(),files_sb[pair]['bkg'].GetName(),'3p_Xn' )
#     SvsB = {}
#     SvsB["all"] = SvsB_Perf_Canvas( files_sb[pair]['sig'].Get('Efficiency_bdt_andrew_1_bdt_cuts'),
#                                     files_sb[pair]['bkg'].Get('Efficiency_bdt_andrew_1_bdt_cuts') )
#     print 'optimal cut all : ',SvsB["all"].GetOptimalCutValue(1-GetGlobalEfficiency( files_sb[pair]['bkg'].Get('Efficiency_old_EF_eta_all')) )

#     SvsB["1p"] = SvsB_Perf_Canvas( files_sb[pair]['sig'].Get('Efficiency_bdt_andrew_1_1p_bdt_cuts'),
#                                    files_sb[pair]['bkg'].Get('Efficiency_bdt_andrew_1_1p_bdt_cuts') )
#     print 'toto'
#     SvsB["1p"] = SvsB_Perf_Canvas( files_sb[pair]['sig'].Get('Efficiency_bdt_andrew_1_1p_Xn_bdt_cuts'),
#                                    files_sb[pair]['bkg'].Get('Efficiency_bdt_andrew_1_1p_Xn_bdt_cuts') )
#     print 'optimal cut 1p_Xn: ',SvsB["1p"].GetOptimalCutValue(1-GetGlobalEfficiency( files_sb[pair]['bkg'].Get('Efficiency_old_EF_eta_1p')) )
#     SvsB["3p"] = SvsB_Perf_Canvas( files_sb[pair]['sig'].Get('Efficiency_bdt_andrew_1_3p_bdt_cuts'),
#                                    files_sb[pair]['bkg'].Get('Efficiency_bdt_andrew_1_3p_bdt_cuts') )
#     print 'optimal cut 3p: ',SvsB["3p"].GetOptimalCutValue(1-GetGlobalEfficiency( files_sb[pair]['bkg'].Get('Efficiency_old_EF_eta_3p')) )
#     SvsB["3p_Xn"] = SvsB_Perf_Canvas( files_sb[pair]['sig'].Get('Efficiency_bdt_andrew_1_3p_Xn_bdt_cuts'),
#                                       files_sb[pair]['bkg'].Get('Efficiency_bdt_andrew_1_3p_Xn_bdt_cuts') )
#     print 'optimal cut 3p_Xn: ',SvsB["3p_Xn"].GetOptimalCutValue(1-GetGlobalEfficiency( files_sb[pair]['bkg'].Get('Efficiency_old_EF_eta_3p_Xn')) )
#     label = ROOT.TLatex()
#     for key in SvsB:
#         SvsB[key].SetTitle(key)
#         SvsB[key].cd()
#         label.DrawLatex(0.05,0.8,key)
#         SvsB[key].Update()


            



# for var in aux.bins:
#     if 'bdt' in var: continue
#     for cat in categories:
        # --> Efficiency plots for various mu values
#         sig_file = ordereddict.OrderedDict()
#         sig_file["8TeV"]  = [ files["Z_8TeV"]      .Get("Efficiency_bdt_quentin_2_"+var+"_"+cat),1,22,aux.bins[var][1],"Efficiency","8 TeV"]
#         sig_file["mu20"]  = [ files["Z_14TeV_mu20"].Get("Efficiency_quentin_2_"+var+"_"+cat),2,23,aux.bins[var][1],"Efficiency","mu=20"]
#         sig_file["mu40"]  = [ files["Z_14TeV_mu40"].Get("Efficiency_bdt_quentin_1_"+var+"_"+cat),3,24,aux.bins[var][1],"Efficiency","mu=40"]
#         sig_file["mu60"]  = [ files["Z_14TeV_mu60"].Get("Efficiency_bdt_quentin_1_"+var+"_"+cat),4,25,aux.bins[var][1],"Efficiency","mu=60"]
#         sig_file["mu80"]  = [ files["Z_14TeV_mu80"].Get("Efficiency_bdt_quentin_1_"+var+"_"+cat),ROOT.kViolet,26,aux.bins[var][1],"Efficiency","mu=80"]
#         EP_mu_sig = EfficiencyPlot("plot_sig_"+var+"_"+cat,"signal "+var+" "+cat,sig_file,"")
        

#         sig_file_2 = ordereddict.OrderedDict()
#         sig_file_2["mu40"]  = [ files["Z_14TeV_mu40"].Get("Efficiency_bdt_andrew_1_"+var+"_"+cat),3,24,aux.bins[var][1],"Efficiency","mu=40"]
#         sig_file_2["mu60"]  = [ files["Z_14TeV_mu60"].Get("Efficiency_bdt_andrew_1_"+var+"_"+cat),4,25,aux.bins[var][1],"Efficiency","mu=60"]
#         sig_file_2["mu80"]  = [ files["Z_14TeV_mu80"].Get("Efficiency_bdt_andrew_1_"+var+"_"+cat),ROOT.kViolet,26,aux.bins[var][1],"Efficiency","mu=80"]
#         EP_mu_sig_2 = EfficiencyPlot("plot_sig_2"+var+"_"+cat,"signal and "+var+" "+cat,sig_file_2,"")


#         # --> Rejection plots for various mu values
#         bkg_file = ordereddict.OrderedDict()
#         bkg_file["mu40"]  = [ RejectionCurve(files["bkg_JF_14TeV_mu40"].Get("Efficiency_quentin_2_"+var+"_"+cat)),3,23,aux.bins[var][1],"Rejection","mu=40"]
#         bkg_file["mu60"]  = [ RejectionCurve(files["bkg_JF_14TeV_mu60"].Get("Efficiency_quentin_2_"+var+"_"+cat)),4,23,aux.bins[var][1],"Rejection","mu=60"]
#         EP_mu_rej = EfficiencyPlot("plot_bkg_"+var+"_"+cat,"background "+var+" "+cat,bkg_file,"")



# roc = {}
# roc["mu40"] = RoCcurve( files["Z_14TeV_mu40"].Get('Efficiency_Andrew1_bdt_1_cuts_all'), files["bkg_JF_14TeV_mu40"].Get('Efficiency_Andrew1_bdt_1_cuts_all') )
# roc["mu60"] = RoCcurve( files["Z_14TeV_mu60"].Get('Efficiency_Andrew1_bdt_1_cuts_all'), files["bkg_JF_14TeV_mu60"].Get('Efficiency_Andrew1_bdt_1_cuts_all')  )

# roc["mu40"].SetLineColor(ROOT.kRed)
# roc["mu60"].SetLineColor(ROOT.kRed)
# roc["mu60"].SetLineStyle(ROOT.kDashed)


# roc_old = {}
# roc_old["mu40"] = ROOT.TGraph()
# roc_old["mu40"].SetPoint(0,GetGlobalEfficiency( files["Z_14TeV_mu40"].Get('Efficiency_old_EF_eta_all')),
#                          1-GetGlobalEfficiency( files["bkg_JF_14TeV_mu40"].Get('Efficiency_old_EF_eta_all')) )
# roc_old["mu60"] = ROOT.TGraph()
# roc_old["mu60"].SetPoint(0,GetGlobalEfficiency( files["Z_14TeV_mu60"].Get('Efficiency_old_EF_eta_all')),
#                          1-GetGlobalEfficiency( files["bkg_JF_14TeV_mu60"].Get('Efficiency_old_EF_eta_all')) )

# roc_old["mu40"].SetMarkerColor(ROOT.kRed)
# roc_old["mu40"].SetMarkerStyle(ROOT.kFullSquare)
# roc_old["mu60"].SetMarkerColor(ROOT.kRed)

# legend = ROOT.TLegend(0.14,0.18,0.55,0.40)
# legend.SetFillColor(0)
# legend.AddEntry( roc["mu40"], "Andrew BDT-1 mu=40", 'l' )
# legend.AddEntry( roc["mu60"], "Andrew BDT-1 mu=60", 'l' )
# legend.AddEntry( roc_old["mu40"], "Old BDT mu=40", 'p' )
# legend.AddEntry( roc_old["mu60"], "Old BDT mu=60", 'p' )




# DiscriVar_Canvas( [files_sb["14TeV_mu40"]["sig"],files_sb["14TeV_mu40"]["bkg"],"h_BDT_bdt_andrew_1","BDT Score"] )
# DiscriVar_Canvas( [files_sb["14TeV_mu40"]["sig"],files_sb["14TeV_mu40"]["bkg"],"h_BDT_bdt_andrew_2","BDT Score"] )
# DiscriVar_Canvas( [files_sb["14TeV_mu40"]["sig"],files_sb["14TeV_mu40"]["bkg"],"h_BDT_bdt_andrew_3","BDT Score"] )
# DiscriVar_Canvas( [files_sb["14TeV_mu40"]["sig"],files_sb["14TeV_mu40"]["bkg"],"h_BDT_bdt_quentin_1","BDT Score"] )
# DiscriVar_Canvas( [files_sb["14TeV_mu40"]["sig"],files_sb["14TeV_mu40"]["bkg"],"h_BDT_bdt_quentin_2","BDT Score"] )

# h_sig = files["Z_14TeV_mu40"].Get("h_BDT_quentin_1")      
# h_bkg = files["bkg_JF_14TeV_mu40"].Get("h_BDT")
# h_bkg.SetLineColor(ROOT.kRed)

# h_sig.Scale(1./h_sig.Integral())
# h_bkg.Scale(1./h_bkg.Integral())

# c_BDT = ROOT.TCanvas()
# h_bkg.Draw("HIST")
# h_sig.Draw("SAMEHIST")
