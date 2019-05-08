nvcc :
nvcc -c upwinding.cu
nvcc -c gradient_g.cu


ubuntu64:
gfortran -o M MeanCurvatureFlow-cuda.f90 upwinding.o gradient_g.o -L/usr/local/cuda/lib64  -lcudart


mac:
gfortran -o M MeanCurvatureFlow-cuda.f90 upwinding.o gradient_g.o -L/usr/local/cuda/lib64  -lcudart

