from rootpy.plotting import Legend, Hist, Graph, Canvas
from rootpy.plotting.style.atlas.labels import ATLAS_label

import ROOT

from ..variables import VARIABLES, get_label
from .templates import RatioPlot, SimplePlot
from .. import ATLAS_LABEL


def draw_ratio(a, b, field, category,
               textsize=22,
               ratio_range=(0, 2),
               ratio_line_values=[0.5, 1, 1.5],
               optional_label_text=None,
               normalize=True,
               logy=False):
    """
    Draw a canvas with two Hists normalized to unity on top
    and a ratio plot between the two hist
    Parameters:
    - a: Nominal Hist (denominator in the ratio)
    - b: Shifted Hist (numerator in the ratio)
    - field: variable field (see variables.py)
    - category: analysis category (see categories/*)
    """
    if field in VARIABLES:
        xtitle = get_label(VARIABLES[field])
    else:
        xtitle = field
    plot = RatioPlot(xtitle=xtitle,
                     ytitle='{0}Events'.format(
                         'Normalized ' if normalize else ''),
                     ratio_title='A / B',
                     ratio_limits=ratio_range,
                     ratio_line_values=ratio_line_values,
                     logy=logy)
    if normalize:
        a_integral = a.integral()
        if a_integral != 0:
            a /= a_integral
        b_integral = b.integral()
        if b_integral != 0:
            b /= b_integral
    a.title = 'A: ' + a.title
    b.title = 'B: ' + b.title
    a.color = 'black'
    b.color = 'red'
    a.legendstyle = 'L'
    b.legendstyle = 'L'
    a.markersize = 0
    b.markersize = 0
    a.linewidth = 2
    b.linewidth = 2
    a.fillstyle = 'hollow'
    b.fillstyle = 'hollow'
    a.linestyle = 'solid'
    b.linestyle = 'dashed'
    a.drawstyle='hist E0'
    b.drawstyle='hist E0'
    plot.draw('main', [a, b], ypadding=(0.3, 0.))
    ratio = Hist.divide(a, b, fill_value=-1)
    ratio.drawstyle = 'hist'
    ratio.color = 'black'
    ratio_band = Graph(ratio, fillstyle='/', fillcolor='black', linewidth=0)
    ratio_band.drawstyle = '20'
    plot.draw('ratio', [ratio_band, ratio])
    with plot.pad('main') as pad:
        # legend
        #         leg = Legend([a, b], 0.2, 0.2, 0.45,
        #                      margin=0.35, textsize=textsize)
        leg = Legend([a, b])
        leg.Draw()
        # draw the category label
        if category is not None:
            label = ROOT.TLatex(
                pad.GetLeftMargin() + 0.04, 0.87,
                category.label)
            label.SetNDC()
            label.SetTextFont(43)
            label.SetTextSize(textsize)
            label.Draw()
        # show p-value and chi^2
        if a.integral() != 0 and b.integral() != 0:
            pvalue = a.Chi2Test(b, 'WW')
            chi2 = a.Chi2Test(b, 'WW CHI2/NDF')
        else:
            pvalue = -9999.
            chi2 = -9999.
        pvalue_label = ROOT.TLatex(
            pad.GetLeftMargin() + 0.04, 0.8,
            "p-value={0:.2f}".format(pvalue))
        pvalue_label.SetNDC(True)
        pvalue_label.SetTextFont(43)
        pvalue_label.SetTextSize(textsize)
        pvalue_label.Draw()
        
        chi2_label = ROOT.TLatex(
            pad.GetLeftMargin() + 0.04, 0.72,
            "#frac{{#chi^{{2}}}}{{ndf}}={0:.2f}".format(chi2))
        chi2_label.SetNDC(True)
        chi2_label.SetTextFont(43)
        chi2_label.SetTextSize(textsize)
        chi2_label.Draw()
        if optional_label_text is not None:
            optional_label = ROOT.TLatex(pad.GetLeftMargin()+0.55,0.87,
                                         optional_label_text )
            optional_label.SetNDC(True)
            optional_label.SetTextFont(43)
            optional_label.SetTextSize(textsize)
            optional_label.Draw()
        if ATLAS_LABEL.lower() == 'internal':
            x = 0.67
            y = 1-pad.GetTopMargin()+0.005
        else:
            x = (1. - pad.GetRightMargin() - 0.03) - len(ATLAS_LABEL) * 0.025
            y = 1-pad.GetTopMargin()+0.01
        ATLAS_label(x, y,
                    sep=0.132, pad=pad, sqrts=None,
                    text=ATLAS_LABEL,
                    textsize=textsize)
    return plot


def draw_shape(a, b, field, category,
               textsize=22,
               optional_label_text=None,
               normalize=True,
               logy=False):
    """
    Draw a canvas with two Hists normalized to unity
    Parameters:
    - a: Nominal Hist (denominator in the ratio)
    - b: Shifted Hist (numerator in the ratio)
    - field: variable field (see variables.py)
    - category: analysis category (see categories/*)
    """
    if field in VARIABLES:
        xtitle = get_label(VARIABLES[field])
    else:
        xtitle = field
    plot = SimplePlot(xtitle=xtitle,
                     ytitle='{0}Events'.format(
                         'Normalized ' if normalize else ''),
                     logy=logy)
    if normalize:
        a_integral = a.integral()
        if a_integral != 0:
            a /= a_integral
        b_integral = b.integral()
        if b_integral != 0:
            b /= b_integral
    a.title = 'A: ' + a.title
    b.title = 'B: ' + b.title
    a.color = 'black'
    b.color = 'red'
    a.legendstyle = 'L'
    b.legendstyle = 'L'
    a.markersize = 0
    b.markersize = 0
    a.linewidth = 2
    b.linewidth = 2
    a.fillstyle = 'hollow'
    b.fillstyle = 'hollow'
    a.linestyle = 'solid'
    b.linestyle = 'dashed'
    a.drawstyle='hist E0'
    b.drawstyle='hist E0'
    plot.draw('main', [a, b], ypadding=(0.3, 0.))
    with plot.pad('main') as pad:
        # legend
        #         leg = Legend([a, b], 0.2, 0.2, 0.45,
        #                      margin=0.35, textsize=textsize)
        leg = Legend([a, b])
        leg.Draw()
        # draw the category label
        if category is not None:
            label = ROOT.TLatex(
                pad.GetLeftMargin() + 0.04, 0.87,
                category.label)
            label.SetNDC()
            label.SetTextFont(43)
            label.SetTextSize(textsize)
            label.Draw()
        # show p-value and chi^2
        pvalue = a.Chi2Test(b, 'WW')
        pvalue_label = ROOT.TLatex(
            pad.GetLeftMargin() + 0.04, 0.8,
            "p-value={0:.2f}".format(pvalue))
        pvalue_label.SetNDC(True)
        pvalue_label.SetTextFont(43)
        pvalue_label.SetTextSize(textsize)
        pvalue_label.Draw()
        chi2 = a.Chi2Test(b, 'WW CHI2/NDF')
        chi2_label = ROOT.TLatex(
            pad.GetLeftMargin() + 0.04, 0.72,
            "#frac{{#chi^{{2}}}}{{ndf}}={0:.2f}".format(chi2))
        chi2_label.SetNDC(True)
        chi2_label.SetTextFont(43)
        chi2_label.SetTextSize(textsize)
        chi2_label.Draw()
        if optional_label_text is not None:
            optional_label = ROOT.TLatex(pad.GetLeftMargin()+0.55,0.87,
                                         optional_label_text )
            optional_label.SetNDC(True)
            optional_label.SetTextFont(43)
            optional_label.SetTextSize(textsize)
            optional_label.Draw()
        if ATLAS_LABEL.lower() == 'internal':
            x = 0.67
            y = 1-pad.GetTopMargin()+0.005
        else:
            x = (1. - pad.GetRightMargin() - 0.03) - len(ATLAS_LABEL) * 0.025
            y = 1-pad.GetTopMargin()+0.01
        ATLAS_label(x, y,
                    sep=0.132, pad=pad, sqrts=None,
                    text=ATLAS_LABEL,
                    textsize=textsize)
    return plot


def draw_efficiency(
    eff_s, rej_b, field, 
    category, textsize=22):

    if field in VARIABLES:
        xtitle = get_label(VARIABLES[field])
    else:
        xtitle = field

    c = Canvas()
    c.SetGridx()
    c.SetGridy()
    eff_s.painted_graph.yaxis.SetRangeUser(0,1.10)
    eff_s.painted_graph.yaxis.title = 'Efficiency'
    eff_s.painted_graph.xaxis.title = xtitle
    eff_s.painted_graph.Draw('AP')
    rej_b.color = 'red'
    rej_b.markerstyle = 'square'
    rej_b.painted_graph.Draw('sameP')
    right_axis = ROOT.TGaxis(
        ROOT.gPad.GetUxmax(), ROOT.gPad.GetUymin(),
        ROOT.gPad.GetUxmax(), ROOT.gPad.GetUymax(), 
        0, 1.10, 510,"+L")
    right_axis.SetLineColor(ROOT.kRed)
    right_axis.SetLabelColor(ROOT.kRed)
    right_axis.SetTextColor(ROOT.kRed)
    right_axis.SetTitle('Rejection = 1 - #epsilon_{B}')
    right_axis.Draw('same')
    ROOT.gStyle.SetPadTickY(0)
    ROOT.gPad.Update()
    ROOT.gStyle.SetPadTickY(1)
    label = ROOT.TLatex(
        c.GetLeftMargin() + 0.04, 0.9,
        category.label)
    label.SetNDC()
    label.SetTextFont(43)
    label.SetTextSize(textsize)
    label.Draw()
    leg = Legend(
        [eff_s, rej_b], pad=c)
        # textsize=20, leftmargin=0.6, topmargin=0.6)
    leg.Draw('same')
    return c

def draw_efficiencies(
    effs, field, 
    category, textsize=22):

    if field in VARIABLES:
        xtitle = get_label(VARIABLES[field])
    else:
        xtitle = field

    c = Canvas()
    c.SetGridx()
    c.SetGridy()
    if not isinstance(effs, (list, tuple)):
        effs = [effs]

    h = Hist(
        10, 
        effs[0].painted_graph.xaxis.min,
        effs[0].painted_graph.xaxis.max)
    h.Draw('HIST')

    colors = ['black', 'red', 'blue', 'green', 'purple']
    if len(effs) > len(colors):
        colors = len(effs) * colors
    for eff, col in zip(effs, colors):
        eff.painted_graph.yaxis.SetRangeUser(0, 1.10)
        eff.painted_graph.yaxis.title = 'Efficiency'
        eff.painted_graph.xaxis.title = xtitle
        eff.painted_graph.color = col
        eff.painted_graph.legendstyle = 'l'
        eff.painted_graph.Draw('SAMEP')
    label = ROOT.TLatex(
        c.GetLeftMargin() + 0.04, 0.9,
        category.label)
    label.SetNDC()
    label.SetTextFont(43)
    label.SetTextSize(textsize)
    label.Draw()
    leg = Legend(effs, pad=c)
        # textsize=20, leftmargin=0.6, topmargin=0.6)
    leg.Draw('same')
    return c
