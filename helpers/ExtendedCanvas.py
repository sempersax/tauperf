from ROOT import TCanvas
from ROOT import TLatex
from ROOT import TPad

class ExtendedCanvas(TCanvas):
    """A class to plot canvas with 2 pads and some options"""

    def __init__(self,npads):
        """Constructor"""
        TCanvas.__init__(self)
        # self._canv = TCanvas()
        self._npads=npads
        self._haslumilabel = False
        self._hascdmlabel = False
        self._hasatlaslabel = False
        self._lat = TLatex();
        self._lat.SetTextSize(0.042);
        self._lat.SetNDC(True);
        if(self._npads !=1):
            self.SetMultiplePads(self._npads)
        return None




    # ------------------------------------------------------------------
    def SetLumiLabel(self, X, Y,lumi):
        self._haslumilabel = True
        self._lat.DrawLatex( X,Y,Form("#int Ldt = %1.1f fb^{-1}",lumi) )

    # ------------------------------------------------------------------
    def SetCdmLabel(self,X,Y,status):
        self._hascdmlabel = True
        if (status == "2010" or status == "2011" ):
            self._lat.DrawLatex(X,Y,"#sqrt{s} = 7 TeV")
        elif( status == "2012" ):
            self._lat.DrawLatex(X,Y,"#sqrt{s} = 8 TeV")
        else:
            Fatal("ExtendedCanvas::SetCdmLabel()","Wrong status !")


    # ------------------------------------------------------------------
    def SetAtlasLabel(self,X,Y,status):
        self._hasatlaslabel = True
        if (status == "internal"):
            self._lat.DrawLatex(X,Y,"#font[72]{ATLAS Internal}")
        elif( status == "progress" ):
            self._lat.DrawLatex(X,Y,"#font[72]{ATLAS Work In Progress}")
        elif( status == "approval"):
            self._lat.DrawLatex(X,Y,"#font[72]{ATLAS For Approval}")
        elif( status == "conference" ):
            self._lat.DrawLatex(X,Y,"#font[72]{ATLAS Preliminary}")
        elif( status == "paper" ):
            self._lat.DrawLatex(X,Y,"#font[72]{ATLAS}")
        else:
            Fatal("ExtendedCanvas::SetAtlasLabel()","Wrong status !")

    # ------------------------------------------------------------------
    def SetGenericLabel(self,X,Y,label):
        self._lat.DrawLatex(self,X,Y,label)

    # ------------------------------------------------------------------
    def SetEtaCategoryLabel(self,X,Y,cat):
        self._lat.SetTextSize(0.037);
        if(cat=="CC"): self._lat.DrawLatex(X,Y,"#it{Central-Central}");
        elif(cat=="CE"): self._lat.DrawLatex(X,Y,"#it{Central-Endcap}");
        elif(cat=="EC"): self._lat.DrawLatex(X,Y,"#it{Endcap-Central}");
        elif(cat=="EE"): self._lat.DrawLatex(X,Y,"#it{Endcap-Endcap}");
        elif(cat=="EE_S"): self._lat.DrawLatex(X,Y,"#it{Endcap-Endcap (same sign)}");
        elif(cat=="EE_O"): self._lat.DrawLatex(X,Y,"#it{Endcap-Endcap (opposite sign)}");
        elif(cat=="NONE"): self._lat.DrawLatex(X,Y,"#it{No categorization}");
        elif(cat=="allcat"): self._lat.DrawLatex(X,Y,"All categories");
        else: Fatal("ExtendedCanvas::SetEtaCategoryLabel","Wrong Eta category !!")
        self._lat.SetTextSize(0.042);

    # ------------------------------------------------------------------
    def SetMultiplePads(self,npads):
        if( npads !=2):
            Fatal("ExtendedCanvas::SetMultiplePads()","Number of pads not supported" )
        self._lat.SetTextSize(0.062)
        self.Divide(1,2)
        p1 = self.GetPad(1)
        p1.SetPad(0.05,0.30,0.97,0.97)
        p1.SetBottomMargin(0.00)
        p2 = self.GetPad(2)
        p2.SetPad(0.05,0.0,0.97,0.30)
        p2.SetTopMargin(0.0)
        p2.SetBottomMargin(0.40)
        p1.SetTicks(1,1)
        p2.SetTicks(1,1)
        p1.Update()
        p2.SetGridy()
        p2.Update()
