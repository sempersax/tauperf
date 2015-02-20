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

table_1p[0][0.3] = 0.68225;
table_1p[0][0.4] = 0.65975;
table_1p[0][0.5] = 0.63475;
table_1p[0][0.6] = 0.60625;
table_1p[0][0.7] = 0.57275;
table_1p[0][0.8] = 0.53175;
table_1p[0][0.9] = 0.45625;
table_1p[1][0.3] = 0.68525;
table_1p[1][0.4] = 0.66475;
table_1p[1][0.5] = 0.64175;
table_1p[1][0.6] = 0.61275;
table_1p[1][0.7] = 0.57925;
table_1p[1][0.8] = 0.53625;
table_1p[1][0.9] = 0.46125;
table_1p[2][0.3] = 0.68875;
table_1p[2][0.4] = 0.67125;
table_1p[2][0.5] = 0.64975;
table_1p[2][0.6] = 0.62525;
table_1p[2][0.7] = 0.59175;
table_1p[2][0.8] = 0.55075;
table_1p[2][0.9] = 0.47575;
table_1p[3][0.3] = 0.69375;
table_1p[3][0.4] = 0.67775;
table_1p[3][0.5] = 0.65725;
table_1p[3][0.6] = 0.63425;
table_1p[3][0.7] = 0.60425;
table_1p[3][0.8] = 0.56175;
table_1p[3][0.9] = 0.48375;
table_1p[4][0.3] = 0.70125;
table_1p[4][0.4] = 0.68775;
table_1p[4][0.5] = 0.67275;
table_1p[4][0.6] = 0.65175;
table_1p[4][0.7] = 0.62675;
table_1p[4][0.8] = 0.58675;
table_1p[4][0.9] = 0.50875;
table_1p[5][0.3] = 0.70975;
table_1p[5][0.4] = 0.69725;
table_1p[5][0.5] = 0.68525;
table_1p[5][0.6] = 0.66925;
table_1p[5][0.7] = 0.64725;
table_1p[5][0.8] = 0.61275;
table_1p[5][0.9] = 0.54475;
table_1p[6][0.3] = 0.71625;
table_1p[6][0.4] = 0.70375;
table_1p[6][0.5] = 0.69425;
table_1p[6][0.6] = 0.68175;
table_1p[6][0.7] = 0.66275;
table_1p[6][0.8] = 0.63525;
table_1p[6][0.9] = 0.57575;
table_1p[7][0.3] = 0.71825;
table_1p[7][0.4] = 0.70775;
table_1p[7][0.5] = 0.69675;
table_1p[7][0.6] = 0.68625;
table_1p[7][0.7] = 0.67175;
table_1p[7][0.8] = 0.64875;
table_1p[7][0.9] = 0.60025;
table_1p[8][0.3] = 0.72075;
table_1p[8][0.4] = 0.71175;
table_1p[8][0.5] = 0.70175;
table_1p[8][0.6] = 0.69275;
table_1p[8][0.7] = 0.67875;
table_1p[8][0.8] = 0.65825;
table_1p[8][0.9] = 0.61875;


return table_1p;
}

std::map<int, std::map<double, double> > get_table_mp()
{
std::map<int, std::map<double, double> > table_mp;

table_mp[0][0.3] = 0.65325;
table_mp[0][0.4] = 0.62725;
table_mp[0][0.5] = 0.60475;
table_mp[0][0.6] = 0.58075;
table_mp[0][0.7] = 0.55375;
table_mp[0][0.8] = 0.52175;
table_mp[0][0.9] = 0.48075;
table_mp[1][0.3] = 0.66425;
table_mp[1][0.4] = 0.64025;
table_mp[1][0.5] = 0.61725;
table_mp[1][0.6] = 0.59425;
table_mp[1][0.7] = 0.56675;
table_mp[1][0.8] = 0.53375;
table_mp[1][0.9] = 0.48875;
table_mp[2][0.3] = 0.67825;
table_mp[2][0.4] = 0.65775;
table_mp[2][0.5] = 0.63425;
table_mp[2][0.6] = 0.61225;
table_mp[2][0.7] = 0.58675;
table_mp[2][0.8] = 0.55275;
table_mp[2][0.9] = 0.50575;
table_mp[3][0.3] = 0.68525;
table_mp[3][0.4] = 0.67175;
table_mp[3][0.5] = 0.64975;
table_mp[3][0.6] = 0.62625;
table_mp[3][0.7] = 0.60175;
table_mp[3][0.8] = 0.56925;
table_mp[3][0.9] = 0.51975;
table_mp[4][0.3] = 0.69875;
table_mp[4][0.4] = 0.68375;
table_mp[4][0.5] = 0.66875;
table_mp[4][0.6] = 0.64275;
table_mp[4][0.7] = 0.61925;
table_mp[4][0.8] = 0.58775;
table_mp[4][0.9] = 0.53975;
table_mp[5][0.3] = 0.70475;
table_mp[5][0.4] = 0.69225;
table_mp[5][0.5] = 0.68175;
table_mp[5][0.6] = 0.65925;
table_mp[5][0.7] = 0.63075;
table_mp[5][0.8] = 0.60225;
table_mp[5][0.9] = 0.55525;
table_mp[6][0.3] = 0.70675;
table_mp[6][0.4] = 0.70175;
table_mp[6][0.5] = 0.68625;
table_mp[6][0.6] = 0.67125;
table_mp[6][0.7] = 0.64275;
table_mp[6][0.8] = 0.61375;
table_mp[6][0.9] = 0.57125;
table_mp[7][0.3] = 0.71525;
table_mp[7][0.4] = 0.70525;
table_mp[7][0.5] = 0.69575;
table_mp[7][0.6] = 0.68225;
table_mp[7][0.7] = 0.65575;
table_mp[7][0.8] = 0.62325;
table_mp[7][0.9] = 0.58275;
table_mp[8][0.3] = 0.72575;
table_mp[8][0.4] = 0.70625;
table_mp[8][0.5] = 0.70225;
table_mp[8][0.6] = 0.68875;
table_mp[8][0.7] = 0.66875;
table_mp[8][0.8] = 0.62925;
table_mp[8][0.9] = 0.59325;


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
