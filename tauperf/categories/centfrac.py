import rootpy.compiled as C

C.register_code(
"""
double CentFrac_Cut(double pt_mev) {

if (pt_mev < 30000.) {
  return 0.68;
} else if (pt_mev > 30000 && pt_mev < 35000) {
  return 0.64;
} else if (pt_mev > 35000 && pt_mev < 40000) {
  return 0.62;
} else if (pt_mev > 40000 && pt_mev < 50000) {
  return 0.60;
} else {
  return 0.57;
}
}
""", ['CentFrac_Cut'])

from rootpy.compiled import CentFrac_Cut
