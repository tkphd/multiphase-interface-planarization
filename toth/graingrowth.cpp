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

			if (std::pow(x[0]-64,2)+std::pow(x[1]-64,2) < 625) {
				grid(n)[0] = 1.0;
				grid(n)[1] = 0.0;
				grid(n)[2] = 0.0;
			} else if (x[0]<64) {
				grid(n)[0] = 0.0001;
				grid(n)[1] = 0.9999;
				grid(n)[2] = 0.0;
			} else {
				grid(n)[0] = 0.0;
				grid(n)[1] = 0.0001;
				grid(n)[2] = 0.9999;
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
		MMSP::grid<3,vector<double> > grid(0,0,64,0,64,0,64);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			if (x[0]<16) {
				if (x[1]<32) grid(n)[1] = 1.0;
				else grid(n)[2] = 1.0;
			}
			else if (x[0]>48) {
				if (x[1]<32) grid(n)[1] = 1.0;
				else grid(n)[2] = 1.0;
			}
			else {
				if (x[1]<16 or x[1]>48) grid(n)[0] = 1.0;
				else grid(n)[0] = 1.0;
			}
		}

		output(grid,filename);
	}
}

template <int dim, typename T> vector<T> upside_grad(const grid<dim, T>& GRID, const vector<int>& x)
{
    vector<T> gradient(dim);
    vector<int> s = x;

    for (int i=0; i<dim; i++) {
        s[i] += 1;
        const T& yh = GRID(s);
        s[i] -= 1;
        const T& yl = GRID(s);

        double weight = 1.0 / dx(GRID, i);
        gradient[i] = weight * (yh - yl);
    }
    return gradient;
}

template <int dim, typename T> vector<T> downside_grad(const grid<dim, T>& GRID, const vector<int>& x)
{
    vector<T> gradient(dim);
    vector<int> s = x;

    for (int i=0; i<dim; i++) {
        const T& yh = GRID(s);
        s[i] -= 1;
        const T& yl = GRID(s);
		s[i] += 1;

        double weight = 1.0 / dx(GRID, i);
        gradient[i] = weight * (yh - yl);
    }
    return gradient;
}

template<typename T>
double g(const MMSP::vector<T>& v)
{
	double ifce_nrg = 1.0/12;
	for (int i=0; i<length(v); i++) {
		ifce_nrg += pow(v[i],4)/4.0 - pow(v[i],3)/3.0;
		for (int j=i+1; j<length(v); j++)
			ifce_nrg += 0.5*pow(v[i],2)*pow(v[j],2);
	}
	return ifce_nrg;
}

template <int dim>
void update(MMSP::grid<dim,vector<double> >& grid, int steps)
{
	double dt = 0.001;

	for (int step=0; step<steps; step++) {
		print_progress(step, steps);
		// update grid must be overwritten each time
		MMSP::grid<dim,vector<double> > update(grid);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(grid); n++) {
			vector<int> x = position(grid,n);

			// dot prod of gradients: centered or midpoint differences?
			vector<vector<double> > gradPhi = gradient(grid,x);
			//vector<vector<double> > gradPhiU = upside_grad(grid,x);
			//vector<vector<double> > gradPhiD = downside_grad(grid,x);
			vector<double> lapPhi = laplacian(grid,x);

			double alleps = 0.0, allomg = 0.0, denom = 0.0;
			for (int i=0; i<fields(grid); i++)
				for (int j=i+1; j<fields(grid); j++)
					denom += 2.0*pow(grid(x)[i],2)*pow(grid(x)[j],2);
			for (int i=0; i<fields(grid); i++)
				for (int j=i+1; j<fields(grid); j++) {
					if (denom < 1.0e-20 || i==j)
						continue;
					double gamij = energy(i,j);
					double delij = width(i,j);
					double epsij = 3.0*gamij*delij; // epsilon squared(ij)
					double omgij = 3.0*gamij/delij; // omega(ij)
					alleps += 2.0*epsij*pow(grid(x)[i],2)*pow(grid(x)[j],2) / denom;
					allomg += 2.0*omgij*pow(grid(x)[i],2)*pow(grid(x)[j],2) / denom;
				}
			vector<double> dedp(fields(grid),0.0);
			vector<double> dwdp(fields(grid),0.0);
			vector<double> dgdp(fields(grid),0.0);
			for (int i=0; i<fields(grid); i++) {
				dgdp[i] = pow(grid(x)[i],3) - pow(grid(x)[i],2);
				for (int j=0; j<fields(grid); j++) {
					if (j>i)
						dgdp[i] += grid(x)[i]*pow(grid(x)[j],2);
					if (denom < 1.0e-20 || i==j)
						continue;
					double gamij = energy(i,j);
					double delij = width(i,j);
					double epsij = 3.0*gamij*delij; // epsilon squared(ij)
					double omgij = 3.0*gamij/delij; // omega(ij)
					dedp[i] += 2.0*grid(x)[i]*(epsij - alleps)*pow(grid(x)[j],2) / denom;
					dwdp[i] += 2.0*grid(x)[i]*(omgij - allomg)*pow(grid(x)[j],2) / denom;
				}
			}
			vector<double> dFdp(fields(grid),0.0);
			double sumdFdp = 0.0;
			for (int i=0; i<fields(grid); i++) {
				dFdp[i] += allomg*dgdp[i] + dwdp[i]*g(grid(x)) - alleps*lapPhi[i];
				for (int j=0; j<fields(grid); j++) {
					for (int d=0; d<dim; d++)
						dFdp[i] += gradPhi[d][j] * (0.5*dedp[i]*gradPhi[d][j] - dedp[j]*gradPhi[d][i]);
				}
				sumdFdp += dFdp[i];
			}
			//double sum = 0.0;
			for (int i=0; i<fields(grid); i++) {
				update(x)[i] = grid(x)[i] + dt*(sumdFdp - 1.0*fields(grid)*dFdp[i]);
				/*
				update(x)[i] = grid(x)[i];
				for (int j=0; j<fields(grid); j++) {
					if (i==j)
						continue;
					update(x)[i] += dt*(dFdp[j] - dFdp[i]);
				}
				sum += update(x)[i];
				*/
			}
			/*
			// project onto Gibbs simplex
			double rsum = 0.0;
			if (fabs(sum)>0.0) rsum = 1.0/sum;
			for (int i=0; i<fields(update); i++)
                update(x)[i] *= rsum;
            */
		}
		swap(grid,update);
		ghostswap(grid);
	}

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
