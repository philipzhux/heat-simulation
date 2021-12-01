all: seq cuda mpi pthread omp hybrid;
rempi: clean mpi;
clean: ; rm -rf hs_*;
clean_make: ;rm -rf ../cmakes/sequential/cmake_build/ ../cmakes/omp/cmake_build/ ../cmakes/mpi/cmake_build/ ../cmakes/pthread/cmake_build/ ../cmakes/hybrid/cmake_build/ ../cmakes/cuda/cmake_build/
seq: hs_seq;
cuda: hs_cuda;
mpi: hs_mpi;
pthread: hs_pthread;
hybrid: hs_hybrid;
omp: hs_omp;
hs_seq: ../cmakes/sequential/cmake_build/Makefile; cmake --build ../cmakes/sequential/cmake_build/
../cmakes/sequential/cmake_build/Makefile:../cmakes/sequential/cmake_build; cmake -H../cmakes/sequential -B../cmakes/sequential/cmake_build
../cmakes/sequential/cmake_build: ; mkdir -p ../cmakes/sequential/cmake_build
hs_cuda: ../cmakes/cuda/cmake_build/Makefile; cmake --build ../cmakes/cuda/cmake_build/
../cmakes/cuda/cmake_build/Makefile:../cmakes/cuda/cmake_build; cmake -H../cmakes/cuda -B../cmakes/cuda/cmake_build
../cmakes/cuda/cmake_build: ; mkdir -p ../cmakes/cuda/cmake_build
hs_mpi: ../cmakes/mpi/cmake_build/Makefile; cmake --build ../cmakes/mpi/cmake_build/
../cmakes/mpi/cmake_build/Makefile:../cmakes/mpi/cmake_build; cmake -H../cmakes/mpi -B../cmakes/mpi/cmake_build
../cmakes/mpi/cmake_build: ; mkdir -p ../cmakes/mpi/cmake_build
hs_pthread: ../cmakes/pthread/cmake_build/Makefile; cmake --build ../cmakes/pthread/cmake_build/
../cmakes/pthread/cmake_build/Makefile:../cmakes/pthread/cmake_build; cmake -H../cmakes/pthread -B../cmakes/pthread/cmake_build
../cmakes/pthread/cmake_build: ; mkdir -p ../cmakes/pthread/cmake_build
hs_hybrid: ../cmakes/hybrid/cmake_build/Makefile; cmake --build ../cmakes/hybrid/cmake_build/
../cmakes/hybrid/cmake_build/Makefile:../cmakes/hybrid/cmake_build; cmake -H../cmakes/hybrid -B../cmakes/hybrid/cmake_build
../cmakes/hybrid/cmake_build: ; mkdir -p ../cmakes/hybrid/cmake_build
hs_omp: ../cmakes/omp/cmake_build/Makefile; cmake --build ../cmakes/omp/cmake_build/
../cmakes/omp/cmake_build/Makefile:../cmakes/omp/cmake_build; cmake -H../cmakes/omp -B../cmakes/omp/cmake_build
../cmakes/omp/cmake_build: ; mkdir -p ../cmakes/omp/cmake_build