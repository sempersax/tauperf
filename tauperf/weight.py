import rootpy.compiled as C


C.register_code(
"""
#include "TH1F.h"
#include "TGraph.h"

class Weight 
{

public:
Weight(const TH1F* h) {m_gr = new TGraph(h);}
~Weight() {delete m_gr;}
double w(const double & pt) {return m_gr->Eval(pt / 1000.);}

private:
TGraph * m_gr;

};


""", ['Weight'])

from rootpy.compiled import Weight
