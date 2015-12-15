# Makefile
# GNU makefile for example grain growth code
# Questions/comments to gruberja@gmail.com (Jason Gruber)

# includes
incdir = $(MMSP_PATH)/include

# compilers/flags
compiler = g++
flags = -O3 -std=c++11 -I $(incdir) -fopenmp
gflags = -g -I $(incdir)
pcompiler = mpic++
pflags = -O3 -std=c++11 -I $(incdir) -include mpi.h

# dependencies
core = $(incdir)/MMSP.main.hpp \
       $(incdir)/MMSP.utility.hpp \
       $(incdir)/MMSP.grid.hpp \
       $(incdir)/MMSP.sparse.hpp

# the program
graingrowth: graingrowth.cpp anisotropy.hpp $(core)
	$(compiler) $(flags) $< -o $@ -lz

debug: graingrowth.cpp anisotropy.hpp $(core)
	g++ $(gflags) $< -o $@ -lz

parallel: graingrowth.cpp anisotropy.hpp $(core)
	$(pcompiler) $(pflags) $< -o $@ -lz

clean:
	rm -f graingrowth parallel
