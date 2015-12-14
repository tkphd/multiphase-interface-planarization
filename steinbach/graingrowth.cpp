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
			if (std::pow(x[0]-64,2)+std::pow(x[1]-64,2) < 1024) {
				set(grid(n),1) = 1.0;
				set(grid(n),2) = 0.0;
				set(grid(n),3) = 0.0;
			} else if (x[0]<64) {
				set(grid(n),1) = 0.0;
				set(grid(n),2) = 1.0;
				set(grid(n),3) = 0.0;
			} else {
				set(grid(n),1) = 0.0;
				set(grid(n),2) = 0.0;
				set(grid(n),3) = 1.0;
			}

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
	double epsilon = 1.0e-8;

	for (int step=0; step<steps; step++) {
		print_progress(step, steps);
		// update grid must be overwritten each time
		MMSP::grid<dim,sparse<double> > update(grid);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int i=0; i<nodes(grid); i++) {
			vector<int> x = position(grid,i);

			// determine nonzero fields within
			// the neighborhood of this node
			sparse<int> s;
			for (int j=0; j<dim; j++)
				for (int k=-1; k<=1; k++) {
					x[j] += k;
					for (int h=0; h<length(grid(x)); h++) {
						int index = MMSP::index(grid(x),h);
						set(s,index) = 1;
					}
					x[j] -= k;
				}
			double S = double(length(s));

			// if only one field is nonzero,
			// then copy this node to update
			if (S<2.0) update(i) = grid(i);

			else {
				// compute laplacian of each field
				sparse<double> lap = laplacian(grid,i);

				// compute variational derivatives
				sparse<double> dFdp;
				for (int h=0; h<length(s); h++) {
					int hindex = MMSP::index(s,h);
					for (int j=h+1; j<length(s); j++) {
						int jindex = MMSP::index(s,j);
						double gamma = energy(hindex,jindex);
						double eps = 4.0/acos(-1.0)*sqrt(0.5*gamma*width);
						double w = 4.0*gamma/width;
						set(dFdp,hindex) += 0.5*eps*eps*lap[jindex]+w*grid(i)[jindex];
						set(dFdp,jindex) += 0.5*eps*eps*lap[hindex]+w*grid(i)[hindex];
						for (int k=j+1; k<length(s); k++) {
							int kindex = MMSP::index(s,k);
							set(dFdp,hindex) += 3.0*grid(i)[jindex]*grid(i)[kindex];
							set(dFdp,jindex) += 3.0*grid(i)[kindex]*grid(i)[hindex];
							set(dFdp,kindex) += 3.0*grid(i)[hindex]*grid(i)[jindex];
						}
					}
				}

				// compute time derivatives
				sparse<double> dpdt;
				for (int h=0; h<length(s); h++) {
					int hindex = MMSP::index(s,h);
					for (int j=h+1; j<length(s); j++) {
						int jindex = MMSP::index(s,j);
						double mu = mobility(hindex,jindex);
						set(dpdt,hindex) -= mu*(dFdp[hindex]-dFdp[jindex]);
						set(dpdt,jindex) -= mu*(dFdp[jindex]-dFdp[hindex]);
					}
				}

				// compute update values
				double sum = 0.0;
				for (int h=0; h<length(s); h++) {
					int index = MMSP::index(s,h);
					double value = grid(i)[index]+dt*(2.0/S)*dpdt[index];
					if (value>1.0) value = 1.0;
					if (value<0.0) value = 0.0;
					if (value>epsilon) set(update(i),index) = value;
					sum += update(i)[index];
				}

				// project onto Gibbs simplex
				double rsum = 0.0;
				if (fabs(sum)>0.0) rsum = 1.0/sum;
				for (int h=0; h<length(update(i)); h++) {
					int index = MMSP::index(update(i),h);
					set(update(i),index) *= rsum;
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
