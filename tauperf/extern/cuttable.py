import os
import rootpy.compiled as C

C.register_code(
"""
#include <map>

int get_bin(double pt_mev)
{

if (pt_mev < 30000.) return 0;
else if (pt_mev > 30000. && pt_mev < 40000.) return 1;
else if (pt_mev > 40000. && pt_mev < 50000.) return 2;
else if (pt_mev > 50000. && pt_mev < 60000.) return 3;
else if (pt_mev > 60000. && pt_mev < 70000.) return 4;
else if (pt_mev > 70000. && pt_mev < 80000.) return 5;
else if (pt_mev > 80000. && pt_mev < 90000.) return 6;
else if (pt_mev > 90000. && pt_mev < 100000.) return 7;
else return 8;

}

std::map<int, std::map<double, double> > get_table_1p() 
{

std::map<int, std::map<double, double> > table_1p;

table_1p[0][0.3] = 0.64425;
table_1p[0][0.4] = 0.60925;
table_1p[0][0.5] = 0.57525;
table_1p[0][0.6] = 0.53775;
table_1p[0][0.7] = 0.49825;
table_1p[0][0.8] = 0.45175;
table_1p[0][0.9] = 0.39425;
table_1p[1][0.3] = 0.65375;
table_1p[1][0.4] = 0.62025;
table_1p[1][0.5] = 0.58575;
table_1p[1][0.6] = 0.54975;
table_1p[1][0.7] = 0.51025;
table_1p[1][0.8] = 0.45775;
table_1p[1][0.9] = 0.40075;
table_1p[2][0.3] = 0.66175;
table_1p[2][0.4] = 0.63275;
table_1p[2][0.5] = 0.60175;
table_1p[2][0.6] = 0.56525;
table_1p[2][0.7] = 0.52825;
table_1p[2][0.8] = 0.47775;
table_1p[2][0.9] = 0.40925;
table_1p[3][0.3] = 0.67075;
table_1p[3][0.4] = 0.64325;
table_1p[3][0.5] = 0.61425;
table_1p[3][0.6] = 0.58125;
table_1p[3][0.7] = 0.53875;
table_1p[3][0.8] = 0.49075;
table_1p[3][0.9] = 0.41475;
table_1p[4][0.3] = 0.68325;
table_1p[4][0.4] = 0.66125;
table_1p[4][0.5] = 0.63625;
table_1p[4][0.6] = 0.60725;
table_1p[4][0.7] = 0.56575;
table_1p[4][0.8] = 0.52225;
table_1p[4][0.9] = 0.43125;
table_1p[5][0.3] = 0.69175;
table_1p[5][0.4] = 0.68075;
table_1p[5][0.5] = 0.65775;
table_1p[5][0.6] = 0.63125;
table_1p[5][0.7] = 0.59675;
table_1p[5][0.8] = 0.54825;
table_1p[5][0.9] = 0.46675;
table_1p[6][0.3] = 0.69575;
table_1p[6][0.4] = 0.68775;
table_1p[6][0.5] = 0.67375;
table_1p[6][0.6] = 0.65325;
table_1p[6][0.7] = 0.62475;
table_1p[6][0.8] = 0.57925;
table_1p[6][0.9] = 0.50875;
table_1p[7][0.3] = 0.69725;
table_1p[7][0.4] = 0.69125;
table_1p[7][0.5] = 0.68075;
table_1p[7][0.6] = 0.66075;
table_1p[7][0.7] = 0.63625;
table_1p[7][0.8] = 0.59975;
table_1p[7][0.9] = 0.53425;
table_1p[8][0.3] = 0.70025;
table_1p[8][0.4] = 0.69175;
table_1p[8][0.5] = 0.68475;
table_1p[8][0.6] = 0.66975;
table_1p[8][0.7] = 0.64675;
table_1p[8][0.8] = 0.61475;
table_1p[8][0.9] = 0.55475;

return table_1p;
}

std::map<int, std::map<double, double> > get_table_mp()
{

std::map<int, std::map<double, double> > table_mp;

table_mp[0][0.3] = 0.64925;
table_mp[0][0.4] = 0.60725;
table_mp[0][0.5] = 0.57225;
table_mp[0][0.6] = 0.53375;
table_mp[0][0.7] = 0.49325;
table_mp[0][0.8] = 0.45075;
table_mp[0][0.9] = 0.40025;
table_mp[1][0.3] = 0.66375;
table_mp[1][0.4] = 0.62425;
table_mp[1][0.5] = 0.58625;
table_mp[1][0.6] = 0.54875;
table_mp[1][0.7] = 0.50875;
table_mp[1][0.8] = 0.46575;
table_mp[1][0.9] = 0.41075;
table_mp[2][0.3] = 0.68425;
table_mp[2][0.4] = 0.64925;
table_mp[2][0.5] = 0.60925;
table_mp[2][0.6] = 0.57175;
table_mp[2][0.7] = 0.53225;
table_mp[2][0.8] = 0.48825;
table_mp[2][0.9] = 0.42925;
table_mp[3][0.3] = 0.69725;
table_mp[3][0.4] = 0.66525;
table_mp[3][0.5] = 0.62725;
table_mp[3][0.6] = 0.58875;
table_mp[3][0.7] = 0.55075;
table_mp[3][0.8] = 0.50625;
table_mp[3][0.9] = 0.44525;
table_mp[4][0.3] = 0.70675;
table_mp[4][0.4] = 0.68675;
table_mp[4][0.5] = 0.65375;
table_mp[4][0.6] = 0.61225;
table_mp[4][0.7] = 0.57375;
table_mp[4][0.8] = 0.53275;
table_mp[4][0.9] = 0.46925;
table_mp[5][0.3] = 0.71225;
table_mp[5][0.4] = 0.69725;
table_mp[5][0.5] = 0.66625;
table_mp[5][0.6] = 0.63225;
table_mp[5][0.7] = 0.59225;
table_mp[5][0.8] = 0.55375;
table_mp[5][0.9] = 0.49825;
table_mp[6][0.3] = 0.71475;
table_mp[6][0.4] = 0.70175;
table_mp[6][0.5] = 0.68225;
table_mp[6][0.6] = 0.64725;
table_mp[6][0.7] = 0.61175;
table_mp[6][0.8] = 0.57125;
table_mp[6][0.9] = 0.52125;
table_mp[7][0.3] = 0.71875;
table_mp[7][0.4] = 0.70575;
table_mp[7][0.5] = 0.68925;
table_mp[7][0.6] = 0.66025;
table_mp[7][0.7] = 0.61725;
table_mp[7][0.8] = 0.57925;
table_mp[7][0.9] = 0.53625;
table_mp[8][0.3] = 0.72225;
table_mp[8][0.4] = 0.71125;
table_mp[8][0.5] = 0.69925;
table_mp[8][0.6] = 0.67875;
table_mp[8][0.7] = 0.63575;
table_mp[8][0.8] = 0.59775;
table_mp[8][0.9] = 0.55625;

return table_mp;

}

// BDT_Cut function
double BDT_Cut(double pt_mev, int ntracks, double target)
{
   std::map<int, std::map<double, double> > table_1p = get_table_1p();
   std::map<int, std::map<double, double> > table_mp = get_table_mp();
   if (ntracks < 2) {
      return table_1p[get_bin(pt_mev)][target];
   } else {
      return table_mp[get_bin(pt_mev)][target];
   }
}
""", ['BDT_Cut'])


from rootpy.compiled import BDT_Cut
