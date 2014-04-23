
from rootpy.plotting import Canvas
from rootpy.extern import argparse
from ROOT import TFile,TCanvas
import ROOT


parser=argparse.ArgumentParser()
parser.add_argument("dir", help="The graph dir name")
parser.add_argument("name_old", help="The name of the old graph")
parser.add_argument("name_new", help="The name of the new graph")

args=parser.parse_args()

dir=args.dir
name_old=args.name_old
name_new=args.name_new

file_old=TFile('efficiencies/efficiencies_presel_Ztautau_14TeV_mu20_v13.root', 'read')
file_new=TFile('efficiencies/efficiencies_presel_Ztautau_14TeV_mu20_v16.root', 'read')

Eff_old=file_old.Get( "%s/%s" % (dir,name_old)  )
Eff_new=file_new.Get( "%s/%s" % (dir,name_new)  )
Eff_new.SetLineStyle(ROOT.kDashed)
Eff_new.SetLineColor(ROOT.kRed)
Eff_new.SetMarkerColor(ROOT.kRed)

c = TCanvas()
c.cd()
Eff_old.Draw("AP")
Eff_new.Draw("sameP")
