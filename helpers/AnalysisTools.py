def SetMaximumHist(hlist):
    # --> Set the maximum of histograms on a list
    # --> to make them all plotable on a canvas
    maximum = -1e6
    for h in hlist:
        if h.GetBinContent(h.GetMaximumBin())>maximum:
            maximum = h.GetBinContent(h.GetMaximumBin())
    for h in hlist:
        h.SetMaximum(maximum+0.1*maximum)

def Efficiency(k,n):
    # --> Returns the efficiency computed following
    # --> http://arxiv.org/abs/0908.0130
    eff = (k+1)/(n+2)
    r1=(k+1)/(n+2);
    r2=(k+2)/(n+3);
    r3=(k+1)/(n+2);
    err_eff=sqrt(r1*r2-r3*r3);
    return eff,err_eff

def Processing(jentry,nentries,period):
    if jentry%period==0 and jentry!=0:
        r = int(float(jentry)/float(nentries)*100)
        print str(jentry)+"/"+str(nentries)+" events processed ---> ("+str(r)+")%"

