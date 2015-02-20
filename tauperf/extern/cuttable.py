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
else return 7;

}

std::map<int, std::map<double, double> > get_table_1p() 
{

std::map<int, std::map<double, double> > table_1p;
table_1p[0][0.3] = 0.5725;
table_1p[0][0.4] = 0.6025;
table_1p[0][0.5] = 0.6325;
table_1p[0][0.6] = 0.6575;
table_1p[0][0.7] = 0.6775;
table_1p[0][0.8] = 0.6975;
table_1p[0][0.9] = 0.7175;
table_1p[1][0.3] = 0.5775;
table_1p[1][0.4] = 0.6125;
table_1p[1][0.5] = 0.6375;
table_1p[1][0.6] = 0.6625;
table_1p[1][0.7] = 0.6825;
table_1p[1][0.8] = 0.7025;
table_1p[1][0.9] = 0.7175;
table_1p[2][0.3] = 0.5875;
table_1p[2][0.4] = 0.6225;
table_1p[2][0.5] = 0.6475;
table_1p[2][0.6] = 0.6675;
table_1p[2][0.7] = 0.6875;
table_1p[2][0.8] = 0.7025;
table_1p[2][0.9] = 0.7225;
table_1p[3][0.3] = 0.6025;
table_1p[3][0.4] = 0.6325;
table_1p[3][0.5] = 0.6575;
table_1p[3][0.6] = 0.6775;
table_1p[3][0.7] = 0.6925;
table_1p[3][0.8] = 0.7075;
table_1p[3][0.9] = 0.7225;
table_1p[4][0.3] = 0.6225;
table_1p[4][0.4] = 0.6475;
table_1p[4][0.5] = 0.6725;
table_1p[4][0.6] = 0.6875;
table_1p[4][0.7] = 0.6975;
table_1p[4][0.8] = 0.7125;
table_1p[4][0.9] = 0.7275;
table_1p[5][0.3] = 0.6425;
table_1p[5][0.4] = 0.6675;
table_1p[5][0.5] = 0.6825;
table_1p[5][0.6] = 0.6975;
table_1p[5][0.7] = 0.7075;
table_1p[5][0.8] = 0.7175;
table_1p[5][0.9] = 0.7275;
table_1p[6][0.3] = 0.6625;
table_1p[6][0.4] = 0.6775;
table_1p[6][0.5] = 0.6925;
table_1p[6][0.6] = 0.7025;
table_1p[6][0.7] = 0.7125;
table_1p[6][0.8] = 0.7225;
table_1p[6][0.9] = 0.7325;
table_1p[7][0.3] = 0.6725;
table_1p[7][0.4] = 0.6875;
table_1p[7][0.5] = 0.6975;
table_1p[7][0.6] = 0.7075;
table_1p[7][0.7] = 0.7175;
table_1p[7][0.8] = 0.7225;
table_1p[7][0.9] = 0.7325;
return table_1p;
}

std::map<int, std::map<double, double> > get_table_mp()
{


std::map<int, std::map<double, double> > table_mp;
table_mp[0][0.3] = 0.5525;
table_mp[0][0.4] = 0.5775;
table_mp[0][0.5] = 0.6025;
table_mp[0][0.6] = 0.6225;
table_mp[0][0.7] = 0.6525;
table_mp[0][0.8] = 0.6775;
table_mp[0][0.9] = 0.7025;
table_mp[1][0.3] = 0.5625;
table_mp[1][0.4] = 0.5925;
table_mp[1][0.5] = 0.6175;
table_mp[1][0.6] = 0.6375;
table_mp[1][0.7] = 0.6625;
table_mp[1][0.8] = 0.6825;
table_mp[1][0.9] = 0.7075;
table_mp[2][0.3] = 0.5825;
table_mp[2][0.4] = 0.6075;
table_mp[2][0.5] = 0.6325;
table_mp[2][0.6] = 0.6575;
table_mp[2][0.7] = 0.6775;
table_mp[2][0.8] = 0.6925;
table_mp[2][0.9] = 0.7225;
table_mp[3][0.3] = 0.5975;
table_mp[3][0.4] = 0.6225;
table_mp[3][0.5] = 0.6475;
table_mp[3][0.6] = 0.6675;
table_mp[3][0.7] = 0.6825;
table_mp[3][0.8] = 0.7025;
table_mp[3][0.9] = 0.7275;
table_mp[4][0.3] = 0.6175;
table_mp[4][0.4] = 0.6425;
table_mp[4][0.5] = 0.6675;
table_mp[4][0.6] = 0.6825;
table_mp[4][0.7] = 0.6975;
table_mp[4][0.8] = 0.7075;
table_mp[4][0.9] = 0.7425;
table_mp[5][0.3] = 0.6275;
table_mp[5][0.4] = 0.6575;
table_mp[5][0.5] = 0.6775;
table_mp[5][0.6] = 0.6925;
table_mp[5][0.7] = 0.7025;
table_mp[5][0.8] = 0.7225;
table_mp[5][0.9] = 0.7475;
table_mp[6][0.3] = 0.6425;
table_mp[6][0.4] = 0.6725;
table_mp[6][0.5] = 0.6825;
table_mp[6][0.6] = 0.6975;
table_mp[6][0.7] = 0.7075;
table_mp[6][0.8] = 0.7275;
table_mp[6][0.9] = 0.7525;
table_mp[7][0.3] = 0.6625;
table_mp[7][0.4] = 0.6825;
table_mp[7][0.5] = 0.6975;
table_mp[7][0.6] = 0.7025;
table_mp[7][0.7] = 0.7225;
table_mp[7][0.8] = 0.7425;
table_mp[7][0.9] = 0.7525;
return table_mp;

}

// single prong function
double BDT_Cut_1P(double pt_mev, double target)
{
std::map<int, std::map<double, double> > table = get_table_1p();
return table[get_bin(pt_mev)][target];
}

// --> Multiprong function
double BDT_Cut_MP(double pt_mev, double target)
{
std::map<int, std::map<double, double> > table = get_table_mp();
return table[get_bin(pt_mev)][target];
}

""", ['BDT_Cut_1P', 'BDT_Cut_MP'])


from rootpy.compiled import BDT_Cut_1P, BDT_Cut_MP
