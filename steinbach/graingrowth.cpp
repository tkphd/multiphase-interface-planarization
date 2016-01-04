// graingrowth.hpp
// Anisotropic coarsening algorithms for 2D and 3D sparse phase field (sparsePF) methods
// Questions/comments to gruberja@gmail.com (Jason Gruber)

#ifndef GRAINGROWTH_UPDATE
#define GRAINGROWTH_UPDATE
#include"MMSP.hpp"
#include"anisotropy.hpp"
#include"graingrowth.hpp"
#include<cmath>

namespace MMSP{

double machine_epsilon = 1.0e-8;


void generate(int dim, const char* filename)
{
	if (dim==1) {
		MMSP::grid<1,sparse<double> > grid(0,0,128);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			if (x[0]<32)      set(grid(n),3) = 1.0;
			else if (x[0]>96) set(grid(n),3) = 1.0;
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

			// Plane
			double rsq = std::pow(x[0]-64,2.0)+std::pow(x[1]-64,2.0);
			if (rsq < 1024)   set(grid(n),1) = 1.0;
			else if (x[0]<64) set(grid(n),2) = 1.0;
			else              set(grid(n),3) = 1.0;

			/*
			// Honeycomb
			if (x[0]<32) {
				if (x[1]<64) set(grid(n),2) = 1.0;
				else set(grid(n),3) = 1.0;
			}
			else if (x[0]>96) {
				if (x[1]<64) set(grid(n),2) = 1.0;
				else set(grid(n),3) = 1.0;
			}
			else {
				if (x[1]<32 or x[1]>96) set(grid(n),1) = 1.0;
				else set(grid(n),0) = 1.0;
			}
			*/
		}

		MMSP::sparse<double> mass;
		for (int n=0; n<nodes(grid); n++)
			for (int l=0; l<length(grid(n)); l++) {
				int index = grid(n).index(l);
				set(mass,index) += grid(n)[index];
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
				if (x[1]<32) set(grid(n),2) = 1.0;
				else set(grid(n),3) = 1.0;
			}
			else if (x[0]>48) {
				if (x[1]<32) set(grid(n),2) = 1.0;
				else set(grid(n),3) = 1.0;
			}
			else {
				if (x[1]<16 or x[1]>48) set(grid(n),1) = 1.0;
				else set(grid(n),0) = 1.0;
			}
		}

		output(grid,filename);
	}
}

template <int dim> void update(MMSP::grid<dim,sparse<double> >& grid, int steps)
{
	double dt = 0.01;
	double width = 8.0;

	for (int step=0; step<steps; step++) {
		print_progress(step, steps);
		// update grid must be overwritten each time
		MMSP::grid<dim,sparse<double> > update(grid);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			// compute laplacian of each field
			sparse<double> lapPhi = laplacian(grid,n);

			double S = double(length(lapPhi));

			// if only one field is nonzero,
			// then copy this node to update
			if (S<2.0) update(x) = grid(n);

			else {

				// compute variational derivatives
				sparse<double> dFdp;
				for (int h=0; h<length(lapPhi); h++) {
					int hindex = MMSP::index(lapPhi,h);
					double phii = grid(n)[hindex];
					for (int j=h+1; j<length(lapPhi); j++) {
						int jindex = MMSP::index(lapPhi,j);
						double phij = grid(n)[jindex];
						double gamma = energy(hindex,jindex);
						double eps = 4.0/acos(-1.0)*sqrt(0.5*gamma*width);
						double w = 4.0*gamma/width;
						set(dFdp,hindex) += 0.5*eps*eps*lapPhi[jindex]+w*phij;
						set(dFdp,jindex) += 0.5*eps*eps*lapPhi[hindex]+w*phii;
						for (int k=j+1; k<length(lapPhi); k++) {
							int kindex = MMSP::index(lapPhi,k);
							double phik = grid(n)[kindex];
							set(dFdp,hindex) += 3.0*phij*phik;
							set(dFdp,jindex) += 3.0*phii*phik;
							set(dFdp,kindex) += 3.0*phii*phij;
						}
					}
				}

				// compute time derivatives
				sparse<double> dpdt;
				for (int h=0; h<length(lapPhi); h++) {
					int hindex = MMSP::index(lapPhi,h);
					for (int j=h+1; j<length(lapPhi); j++) {
						int jindex = MMSP::index(lapPhi,j);
						double mu = mobility(hindex,jindex);
						set(dpdt,hindex) -= mu*(dFdp[hindex]-dFdp[jindex]);
						set(dpdt,jindex) -= mu*(dFdp[jindex]-dFdp[hindex]);
					}
				}

				// compute update values
				double sum = 0.0;
				for (int h=0; h<length(dpdt); h++) {
					int index = MMSP::index(dpdt,h);
					double value = grid(n)[index]+dt*(2.0/S)*dpdt[index];
					if (value>1.0) value = 1.0;
					if (value<0.0) value = 0.0;
					if (value>machine_epsilon) set(update(x),index) = value;
					sum += update(x)[index];
				}

				// project onto Gibbs simplex
				double rsum = (fabs(sum)>machine_epsilon)?1.0/sum:0.0;
				for (int h=0; h<length(update(x)); h++) {
					int index = MMSP::index(update(x),h);
					set(update(x),index) *= rsum;
				}
			}
		}
		swap(grid,update);
		ghostswap(grid);
	}
	MMSP::sparse<double> mass;
	for (int n=0; n<nodes(grid); n++)
		for (int l=0; l<length(grid(n)); l++) {
			int index = grid(n).index(l);
			set(mass,index) += grid(n)[index];
		}
	for (int l=0; l<length(mass); l++)
		std::cout<<mass.value(l)<<'\t';
	std::cout<<std::endl;
}


} // namespace MMSP

#endif

#include"MMSP.main.hpp"
