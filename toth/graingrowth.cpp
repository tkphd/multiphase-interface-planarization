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

const double machine_epsilon = 1.0e-8;

void generate(int dim, const char* filename)
{
	if (dim==1) {
		MMSP::grid<1,sparse<double> > grid(0,0,128);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			if (x[0]<32)      set(grid(n),2) = 1.0;
			else if (x[0]>96) set(grid(n),2) = 1.0;
			else              set(grid(n),0) = 1.0;
		}

		output(grid,filename);
	}

	if (dim==2) {
		MMSP::grid<2,sparse<double> > grid(0,0,128,0,128);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			double rsq = std::pow(x[0]-64,2.0)+std::pow(x[1]-64,2.0);
			if (rsq < 1024)   set(grid(n),0) = 1.0;
			else if (x[0]<64) set(grid(n),1) = 1.0;
			else              set(grid(n),2) = 1.0;
		}

		MMSP::sparse<double> mass;
		for (int n=0; n<nodes(grid); n++)
			for (int k=0; k<length(grid(n)); k++) {
				int i = index(grid(n),k);
				set(mass,i) += grid(n)[i];
			}
		for (int l=0; l<length(mass); l++)
			std::cout<<mass.value(l)<<'\t';
		std::cout<<std::endl;

		output(grid,filename);
	}

	if (dim==3) {
		MMSP::grid<3,sparse<double> > grid(0,0,64,0,64,0,64);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			if (x[0]<16) {
				if (x[1]<32) set(grid(n),1) = 1.0;
				else set(grid(n),2) = 1.0;
			}
			else if (x[0]>48) {
				if (x[1]<32) set(grid(n),1) = 1.0;
				else set(grid(n),2) = 1.0;

			}
			else {
				if (x[1]<16 || x[1]>48) set(grid(n),0) = 1.0;
				else set(grid(n),0) = 1.0;
			}
		}

		output(grid,filename);
	}
}

template<typename T>
double multiwell(const MMSP::sparse<T>& v)
{
	// this is the 'g' function, Eqn. 30
	double ifce_nrg = 1.0/12;
	for (int k=0; k<length(v); k++) {
		ifce_nrg += pow(v.value(k),4.0)/4.0 - pow(v.value(k),3.0)/3.0;
		for (int l=k+1; l<length(v); l++)
			ifce_nrg += pow(v.value(k),2.0)*pow(v.value(l),2.0)/2.0;
	}
	return ifce_nrg;
}

template <int dim>
void update(MMSP::grid<dim,sparse<double> >& grid, int steps)
{
	double dt = 0.01;

	for (int step=0; step<steps; step++) {
		print_progress(step, steps);

		ghostswap(grid);
		MMSP::grid<dim,sparse<double> > update(grid);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			/*
			// Symmetric EOM
			double omega = 1.0;
			double epssq = 1.0;

			sparse<double> lapPhi = laplacian(grid,x);
			double sumPhiSq = 0.0;
			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				sumPhiSq += pow(grid(n)[i],2.0);
			}

			sparse<double> dFdp;
			double sumdFdp = 0.0;
			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				double phii = grid(n)[i];
				set(dFdp,i) = omega*phii*(sumPhiSq-phii) - epssq*lapPhi[i];
				sumdFdp += dFdp[i];
			}

			for (int k=0; k<length(dFdp); k++) {
				int i = index(dFdp,k);
				set(update(x),i) = grid(n)[i] + dt*(sumdFdp - double(length(dFdp))*dFdp[i]);
			}
			*/

			// Asymmetric EOM
			vector<sparse<double> > gradPhi = gradient(grid,x);
			sparse<double> lapPhi = laplacian(grid,x);

			double denom = 0.0;
			for (int k=0; k<length(lapPhi); k++)
				for (int l=0; l<length(lapPhi); l++)
					denom += pow(grid(n).value(k),2.0)*pow(grid(n).value(l),2.0);
			int rdenom = (denom>machine_epsilon)?1.0/denom:0.0;

			double eps0 = 3.0;
			double omg0 = 3.0;

			double alleps = 0.0, allomg = 0.0;
			for (int k=0; k<length(grid(n)); k++) {
				int i = index(grid(n),k);
				double phii2 = pow(grid(n)[i],2.0);
				for (int l=0; l<length(grid(n)); l++) {
					int j = index(grid(n),l);
					double phij2 = pow(grid(n)[j],2.0);
					double gamij = energy(i,j);
					double delij = width(i,j);
					double epsij = eps0*gamij*delij; // epsilon squared(ij)
					double omgij = omg0*gamij/delij; // omega(ij)
					alleps += epsij*phii2*phij2 * rdenom;
					allomg += omgij*phii2*phij2 * rdenom;
				}
			}

			sparse<double> dedp;
			sparse<double> dwdp;
			sparse<double> dgdp;
			for (int k=0; k<length(grid(n)); k++) {
				int i = index(grid(n),k);
				double phii = grid(n)[i];
				set(dgdp,i) = pow(phii,3.0) - pow(phii,2.0);
				for (int l=0; l<length(grid(n)); l++) {
					if (k==l) continue;
					int j = index(grid(n),l);
					double phij2 = pow(grid(n)[j],2.0);
					double gamij = energy(i,j);
					double delij = width(i,j);
					double epsij = eps0*gamij*delij; // epsilon squared(ij)
					double omgij = omg0*gamij/delij; // omega(ij)
					set(dedp,i) += 2.0*phii*(epsij - alleps)*phij2 * rdenom;
					set(dwdp,i) += 2.0*phii*(omgij - allomg)*phij2 * rdenom;
					set(dgdp,i) += phii*phij2;
				}
			}

			sparse<double> dFdp;
			double sumdFdp = 0.0;
			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				set(dFdp,i) = allomg*dgdp[i] + multiwell(grid(n))*dwdp[i] - alleps*lapPhi[i];
				for (int l=0; l<length(lapPhi); l++) {
					int j = index(lapPhi,l);
					for (int d=0; d<dim; d++)
						set(dFdp,i) += gradPhi[d][j] * (0.5*dedp[i]*gradPhi[d][j] - dedp[j]*gradPhi[d][i]);
				}
				sumdFdp += dFdp[i];
			}

			for (int k=0; k<length(dFdp); k++) {
				int i = index(dFdp,k);
				set(update(x),i) =  grid(n)[i] + dt*(sumdFdp - double(length(dFdp))*dFdp[i]);
			}

		}
		swap(grid,update);
	}
	// In case vector calculations are necessary for mass or energy
	ghostswap(grid);

	MMSP::sparse<double> mass;
	for (int n=0; n<nodes(grid); n++)
		for (int k=0; k<length(grid(n)); k++) {
			int i = index(grid(n),k);
			set(mass,i) += grid(n)[i];
		}
	for (int l=0; l<length(mass); l++)
		std::cout<<mass.value(l)<<'\t';
	std::cout<<std::endl;
}


} // namespace MMSP

#endif

#include"MMSP.main.hpp"
