// stl_loader.h
#include<vector>

#ifdef __CINT__
#pragma link C++ class vector<vector<int> >;
#pragma link C++ class vector<vector<float> >;
#else
template class std::vector<std::vector<float> >;
template class std::vector<std::vector<int> >;
#endif

