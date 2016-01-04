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
			if (rsq < 1024)	  set(grid(n),0) = 1.0;
			else if (x[0]<64) set(grid(n),1) = 1.0;
			else              set(grid(n),2) = 1.0;
		}

		MMSP::sparse<double> mass;
		for (int n=0; n<nodes(grid); n++) {
			double local_mass = 0.0;
			for (int k=0; k<length(grid(n)); k++) {
				int i = index(grid(n),k);
				local_mass += pow(grid(n)[i],2.0);
			}
			double recip_mass = (local_mass>machine_epsilon)?1.0/local_mass:0.0;
			for (int k=0; k<length(grid(n)); k++) {
				int i = index(grid(n),k);
				set(mass,i) += pow(grid(n)[i],2.0)*recip_mass;
			}
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
	double ifce_nrg = 0.0;
	for (int k=0; k<length(v); k++) {
		int i = index(v,k);
		ifce_nrg += pow(v[i],4.0)/4.0 - pow(v[i],2.0)/2.0;
		for (int l=k+1; l<length(v); l++) {
			int j = index(v,l);
			ifce_nrg += 2.0*energy(i,j)*pow(v[i],2.0)*pow(v[j],2.0);
		}
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

			double kap0 = 1.0;
			double omg0 = 1.0;

			/*
			// Symmetric EOM
			double gamma = energy(0,1);
			double delta = width(0,1);
			double kappa = kap0*gamij*delij;
			double omega = omg0*gamij/delij;

			sparse<double> lapPhi = laplacian(grid,x);
			sparse<double> dFdp;

			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				double phii = grid(n)[i];
				set(dFdp,i) = omega*(pow(phii,3.0) - phii) - kappa*lapPhi[i];
				for (int l=0; l<length(lapPhi); l++) {
					int j = index(lapPhi,l);
					if (i==j) continue;
					set(dFdp,i) += 2.0*omega*gamma*phii*pow(grid(n)[j],2.0);
				}
			}

			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				set(update(x),i) = grid(n)[i] - dt*dFdp[i];
			}
			*/

			// Asymmetric EOM
			vector<sparse<double> > gradPhi = gradient(grid,x);
			sparse<double> lapPhi = laplacian(grid,x);

			double denom = 0.0;
			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				for (int l=0; l<length(lapPhi); l++) {
					int j = index(lapPhi,l);
					denom += pow(grid(n)[i],2.0)*pow(grid(n)[j],2.0);
				}
			}
			double rdenom = (denom>machine_epsilon)?1.0/denom:0.0;

			double allkap = 0.0, allomg = 0.0;
			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				double phii = grid(n)[i];
				for (int l=0; l<length(lapPhi); l++) {
					int j = index(lapPhi,l);
					double phij2 = pow(grid(n)[j],2.0);
					double gamij = energy(i,j);
					double delij = width(i,j);
					double kapij = kap0*gamij*delij; // epsilon squared(ij)
					double omgij = omg0*gamij/delij; // omega(ij)
					allkap += kapij*pow(phii,2.0)*phij2 * rdenom;
					allomg += omgij*pow(phii,2.0)*phij2 * rdenom;
				}
			}

			sparse<double> dedp;
			sparse<double> dwdp;
			sparse<double> dgdp;
			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				double phii = grid(n)[i];
				set(dgdp,i) = pow(phii,3.0) - phii;
				for (int l=0; l<length(lapPhi); l++) {
					int j = index(lapPhi,l);
					if (i==j) continue;
					double phij2 = pow(grid(n)[j],2.0);
					double gamij = energy(i,j);
					double delij = width(i,j);
					double kapij = kap0*gamij*delij; // epsilon squared(ij)
					double omgij = omg0*gamij/delij; // omega(ij)
					set(dedp,i) += 2.0*phii*(kapij - allkap)*phij2 * rdenom;
					set(dwdp,i) += 2.0*phii*(omgij - allomg)*phij2 * rdenom;
					set(dgdp,i) += 2.0*gamij*phii*phij2;
				}
			}

			for (int k=0; k<length(lapPhi); k++) {
				int i = index(lapPhi,k);
				double dFdp = allomg*dgdp[i] + multiwell(grid(n))*dwdp[i] - allkap*lapPhi[i];
				for (int l=0; l<length(lapPhi); l++) {
					int j = index(lapPhi,l);
					for (int d=0; d<dim; d++)
						dFdp += gradPhi[d][j] * (0.5*dedp[i]*gradPhi[d][j] - dedp[j]*gradPhi[d][i]);
				}
				set(update(x),i) = grid(n)[i] - dt*dFdp;
			}

		}
		swap(grid,update);
	}
	// In case vector calculations are necessary for mass or energy
	ghostswap(grid);

	MMSP::sparse<double> mass;
	for (int n=0; n<nodes(grid); n++) {
		double local_mass = 0.0;
		for (int k=0; k<length(grid(n)); k++) {
			int i = index(grid(n),k);
			local_mass += pow(grid(n)[i],2.0);
		}
		double recip_mass = (local_mass>machine_epsilon)?1.0/local_mass:0.0;
		for (int k=0; k<length(grid(n)); k++) {
			int i = index(grid(n),k);
			set(mass,i) += pow(grid(n)[i],2.0)*recip_mass;
		}
	}
	for (int l=0; l<length(mass); l++)
		std::cout<<mass.value(l)<<'\t';
	std::cout<<std::endl;
}


} // namespace MMSP

#endif

#include"MMSP.main.hpp"
