// anisotropy.hpp
// Anisotropic energy and mobility functions
// Questions/comments to gruberja@gmail.com (Jason Gruber)

#ifndef ANISOTROPY
#define ANISOTROPY
#include<map>

// global energy and mobility storage
namespace anisotropy{
	std::map<int,std::map<int,double> > energy_table;
	std::map<int,std::map<int,double> > width_table;
}

template <typename T> T min(const T& a, const T& b) {return (a<b?a:b);}
template <typename T> T max(const T& a, const T& b) {return (a>b?a:b);}

double energy(int i, int j)
{
	using namespace anisotropy;

	// use computed value, if possible
	int a = min(i,j);
	int b = max(i,j);
	double energy = energy_table[a][b];
	if (energy==0.0) {
		// compute energy here...
		energy = 1.0/3;
		energy_table[a][b] = energy;
	}
	return energy;
}

double width(int i, int j)
{
	using namespace anisotropy;

	// use computed value, if possible
	int a = min(i,j);
	int b = max(i,j);
	double width = width_table[a][b];
	if (width==0.0) {
		// compute width here...
		width = 1.0;
		width_table[a][b] = width;
	}
	return width;
}

#endif
