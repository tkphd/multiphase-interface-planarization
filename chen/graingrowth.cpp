// graingrowth.hpp
// Asymmetric coarsening algorithms for 2D and 3D multiphase field methods
// Questions/comments to gruberja@gmail.com (Jason Gruber)

#ifndef GRAINGROWTH_UPDATE
#define GRAINGROWTH_UPDATE
#include"MMSP.hpp"
#include"anisotropy.hpp"
#include"graingrowth.hpp"
#include<cmath>

namespace MMSP{

void generate(int dim, const char* filename)
{
	if (dim==1) {
		MMSP::grid<1,vector<double> > grid(3,0,128);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			if (x[0]<32)      grid(n)[2] = 1.0;
			else if (x[0]>96) grid(n)[2] = 1.0;
			else              grid(n)[0] = 1.0;
			grid(n)[1] = 0.0;
		}

		output(grid,filename);
	}

	if (dim==2) {
		MMSP::grid<2,vector<double> > grid(3,0,128,0,128);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			if (std::pow(x[0]-64,2.0)+std::pow(x[1]-64,2.0) < 1024) {
				grid(n)[0] = 1.0;
				grid(n)[1] = 0.0;
				grid(n)[2] = 0.0;
			} else if (x[0]<64) {
				grid(n)[0] = 0.0;
				grid(n)[1] = 1.0;
				grid(n)[2] = 0.0;
			} else {
				grid(n)[0] = 0.0;
				grid(n)[1] = 0.0;
				grid(n)[2] = 1.0;
			}
		}

		MMSP::vector<double> mass(fields(grid));
		for (int n=0; n<nodes(grid); n++)
			for (int l=0; l<fields(grid); l++)
				mass[l] += grid(n)[l];
		for (int l=0; l<length(mass); l++)
			std::cout<<mass[l]<<'\t';
		std::cout<<std::endl;

		output(grid,filename);
	}

	if (dim==3) {
		MMSP::grid<3,vector<double> > grid(3,0,64,0,64,0,64);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			if (x[0]<16) {
				if (x[1]<32) grid(n)[1] = 1.0;
				else grid(n)[2] = 1.0;
				grid(n)[0] = 0.0;
			}
			else if (x[0]>48) {
				if (x[1]<32) grid(n)[1] = 1.0;
				else grid(n)[2] = 1.0;
				grid(n)[0] = 0.0;

			}
			else {
				if (x[1]<16 || x[1]>48) grid(n)[0] = 1.0;
				else grid(n)[0] = 1.0;
				grid(n)[1] = 0.0;
			}
		}

		output(grid,filename);
	}
}

template<typename T>
double multiwell(const MMSP::vector<T>& v)
{
	double ifce_nrg = 0.0;
	for (int i=0; i<length(v); i++) {
		ifce_nrg += pow(v[i],4.0)/4.0 - pow(v[i],2.0)/2.0;
		for (int j=i+1; j<length(v); j++)
			ifce_nrg += 2.0*energy(i,j)*pow(v[i],2.0)*pow(v[j],2.0);
	}
	return ifce_nrg;
}

template <int dim>
void update(MMSP::grid<dim,vector<double> >& grid, int steps)
{
	double dt = 0.01;
	double epsilon = 1.0e-8;

	for (int step=0; step<steps; step++) {
		print_progress(step, steps);

		ghostswap(grid);
		MMSP::grid<dim,vector<double> > update(grid);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			double omega = 1.0;
			double gamma = 3.0/2;
			double kappa = 1.0;

			// Symmetric EOM
			vector<double> lapPhi = laplacian(grid,x);
			vector<double> dFdp(fields(grid),0.0);

			for (int i=0; i<fields(grid); i++) {
				double phii = grid(x)[i];
				dFdp[i] = omega*(pow(phii,3.0) - phii) - kappa*lapPhi[i];
				for (int j=0; j<fields(grid); j++) {
					if (i==j) continue;
					dFdp[i] += 2.0*omega*gamma*phii*pow(grid(x)[j],2.0);
				}
			}

			for (int i=0; i<fields(grid); i++)
				update(x)[i] = grid(x)[i] - dt*dFdp[i];

			/*
			// Asymmetric EOM
			vector<vector<double> > gradPhi = gradient(grid,x);
			vector<double> lapPhi = laplacian(grid,x);

			double denom = 0.0, rdenom=0.0;
			for (int i=0; i<fields(grid); i++)
				for (int j=0; j<fields(grid); j++)
					denom += pow(grid(x)[i],2.0)*pow(grid(x)[j],2.0);
			if (denom>epsilon) rdenom = 1.0/denom;

			double allkap = 0.0, allomg = 0.0;
			for (int i=0; i<fields(grid); i++) {
				double phii = grid(x)[i];
				for (int j=0; j<fields(grid); j++) {
					double phij2 = pow(grid(x)[j],2.0);
					double gamij = energy(i,j);
					double delij = width(i,j);
					double kapij = kap0*gamij*delij; // epsilon squared(ij)
					double omgij = omg0*gamij/delij; // omega(ij)
					allkap += kapij*pow(phii,2.0)*phij2 * rdenom;
					allomg += omgij*pow(phii,2.0)*phij2 * rdenom;
				}
			}

			vector<double> dedp(fields(grid),0.0);
			vector<double> dwdp(fields(grid),0.0);
			vector<double> dgdp(fields(grid),0.0);
			for (int i=0; i<fields(grid); i++) {
				double phii = grid(x)[i];
				dgdp[i] = pow(phii,3.0) - phii;
				for (int j=0; j<fields(grid); j++) {
					if (i==j) continue;
					double phij2 = pow(grid(x)[j],2.0);
					double gamij = energy(i,j);
					double delij = width(i,j);
					double kapij = kap0*gamij*delij; // epsilon squared(ij)
					double omgij = omg0*gamij/delij; // omega(ij)
					dedp[i] += 2.0*phii*(kapij - allkap)*phij2 * rdenom;
					dwdp[i] += 2.0*phii*(omgij - allomg)*phij2 * rdenom;
					dgdp[i] += 2.0*gamij*phii*phij2;
				}
			}

			for (int i=0; i<fields(grid); i++) {
				double dFdp = allomg*dgdp[i] + multiwell(grid(x))*dwdp[i] - allkap*lapPhi[i];
				for (int j=0; j<fields(grid); j++)
					for (int d=0; d<dim; d++)
						dFdp += gradPhi[d][j] * (0.5*dedp[i]*gradPhi[d][j] - dedp[j]*gradPhi[d][i]);
				update(x)[i] = grid(x)[i] - dt*dFdp;
			}
			*/

		}
		swap(grid,update);
	}
	// In case vector calculations are necessary for mass or energy
	ghostswap(grid);

	MMSP::vector<double> mass(fields(grid));
	for (int n=0; n<nodes(grid); n++)
		for (int l=0; l<fields(grid); l++)
			mass[l] += grid(n)[l];
	for (int l=0; l<length(mass); l++)
		std::cout<<mass[l]<<'\t';
	std::cout<<std::endl;
}


} // namespace MMSP

#endif

#include"MMSP.main.hpp"
