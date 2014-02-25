import ROOT


def SetAtlasStyle():
    AtlasStyle = ROOT.TStyle("ATLAS","Atlas style")

    # use plain black on white colors
    iColor = iMode = 0
    AtlasStyle.SetFrameBorderMode(iMode)
    AtlasStyle.SetCanvasBorderMode(iMode)
    AtlasStyle.SetPadBorderMode(iMode)
    AtlasStyle.SetPadColor(iColor)
    AtlasStyle.SetCanvasColor(iColor)
    AtlasStyle.SetStatColor(iColor)
    # AtlasStyle.SetFillColor(iColor)
    # set the paper & margin sizes
    AtlasStyle.SetPaperSize(20,26)
    AtlasStyle.SetPadTopMargin(0.05)
    AtlasStyle.SetPadRightMargin(0.1)
    AtlasStyle.SetPadBottomMargin(0.16)
    AtlasStyle.SetPadLeftMargin(0.12)
    # use large fonts
    iFont = 42
    TextSize = 0.05
    AtlasStyle.SetTextFont(iFont)
    AtlasStyle.SetTextSize(TextSize)
    AtlasStyle.SetLabelFont(iFont,"x")
    AtlasStyle.SetTitleFont(iFont,"x")
    AtlasStyle.SetLabelFont(iFont,"y")
    AtlasStyle.SetTitleFont(iFont,"y")
    AtlasStyle.SetLabelFont(iFont,"z")
    AtlasStyle.SetTitleFont(iFont,"z")
    AtlasStyle.SetLabelSize(TextSize,"x")
    AtlasStyle.SetTitleSize(TextSize,"x")
    AtlasStyle.SetLabelSize(TextSize,"y")
    AtlasStyle.SetTitleSize(TextSize,"y")
    AtlasStyle.SetLabelSize(TextSize,"z")
    AtlasStyle.SetTitleSize(TextSize,"z")
    # use bold lines and markers
    AtlasStyle.SetMarkerStyle(20)
    AtlasStyle.SetMarkerSize(1.2)
    AtlasStyle.SetHistLineWidth(2)
    AtlasStyle.SetLineStyleString(2,"[12 12]") # postscript dashes
    # get rid of X error bars and y error bar caps
    # AtlasStyle.SetErrorX(0.001)

    # Palette
    AtlasStyle.SetPalette(1)

    # do not display any of the standard histogram decorations
    AtlasStyle.SetOptTitle(0)
    # AtlasStyle.SetOptStat(1111)
    AtlasStyle.SetOptStat(0)
    # AtlasStyle.SetOptFit(1111)
    AtlasStyle.SetOptFit(0)

    # put tick marks on top and RHS of plots
    AtlasStyle.SetPadTickX(1)
    AtlasStyle.SetPadTickY(1)

    ROOT.gROOT.SetStyle("ATLAS")
    ROOT.gROOT.ForceStyle()

##gStyle->SetPadTickX(1)
##gStyle->SetPadTickY(1)
# from ROOT import *
# ROOT.gROOT.LoadMacro("AtlasStyle.C") 
# #SetAtlasStyle()
