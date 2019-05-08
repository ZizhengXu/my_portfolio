#include <stdio.h>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h> 


__global__ void grkernel(double *d_g,double *d_gx,double *d_gy,double *d_gz,
                         int nx,int ny,int nz,  double* d_w) 
{//    do iz = 2, nz - 1
//        do iy = 2, ny - 1
//            do ix = 2, nx - 1
//                gx(ix, iy, iz) = w(-1) * g(ix - 1, iy, iz) + w(1) * g(ix + 1, iy, iz)
/*       
       for i = 1 to nz-2
    d_gx[ix*nz*ny+iy*nz+iz] = d_w[0] *d_gx[(ix-1)*nz*ny+iy*nz+iz] + w[1] * d_g[(ix+1)*nz*ny+iy*nz+iz]
*/
//                gy(ix, iy, iz) = w(-1) * g(ix, iy-1, iz) + w(1) * g(ix, iy + 1, iz)
//                gz(ix, iy, iz) = w(-1) * g(ix, iy, iz-1) + w(1) * g(ix, iy, iz + 1)
//            enddo
//        enddo
//    enddo
//    return
//end subroutine gradient_g
    int iz=blockIdx.x+1;int ix=threadIdx.x+1;int iy=threadIdx.y+1;
    d_gx[ix*nz*ny+iy*nz+iz] = d_w[0] *d_g[(ix-1)*nz*ny+iy*nz+iz] + d_w[1] *d_g[(ix+1)*nz*ny+iy*nz+iz];
    d_gy[ix*nz*ny+iy*nz+iz] = d_w[0] *d_g[ix*nz*ny+(iy-1)*nz+iz] + d_w[1] * d_g[ix*nz*ny+(iy+1)*nz+iz];
    d_gz[ix*nz*ny+iy*nz+iz] = d_w[0] *d_g[ix*nz*ny+iy*nz+iz-1] + d_w[1] * d_g[ix*nz*ny+iy*nz+ iz + 1];
    __syncthreads();
 }


extern "C"  void gradient_g_(double *nx,double* ny,double* nz,double* g,double* gx,
                       double* gy,double *gz,double* dx)
{//    implicit real*8 (a-h, o-z)
//    dimension g(nx, ny, nz), gx(nx, ny, nz), gy(nx, ny, nz), gz(nx, ny, nz)
   
    double *d_g,*d_gx,*d_gy,*d_gz;
    double *d_w;
    int N=(*nx)* (*ny)* (*nz);
//    dimension w(-1:1)
    
//    w(-1) = -5.d-1/dx; w(0) = 0.0d0; w(1) = 5.d-1/dx
    double w[3]={-5. -1/ *dx,0,5-1/ *dx};
      /* Allocate complex array on device */
    cudaMalloc ((void **) &d_g , sizeof(double)*N);
    cudaMalloc ((void **) &d_gx , sizeof(double)*N);
    cudaMalloc ((void **) &d_gy , sizeof(double)*N);
    cudaMalloc ((void **) &d_gz , sizeof(double)*N);
    cudaMalloc ((void **) &d_w , sizeof(double)*2);
  /* Copy array from host memory to device memory */
    cudaMemcpy( d_g, g,  sizeof(double)*N,cudaMemcpyHostToDevice);
   
  /* Compute execution configuration */

    dim3 dimGrid (*nz-2);    dim3 dimBlock(*nx-2,*ny-2);
    grkernel<<<dimGrid,dimBlock>>>(d_g,d_gx,d_gy,d_gz,*nx,*ny,*nz,d_w);

  /* Copy the result back */
   
    cudaMemcpy( gx,d_gx,sizeof(double)*N,cudaMemcpyDeviceToHost);  
    cudaMemcpy( gy,d_gy,sizeof(double)*N,cudaMemcpyDeviceToHost);  
    cudaMemcpy( gz,d_gz,sizeof(double)*N,cudaMemcpyDeviceToHost);  
   

  /* Free memory on the device */
    cudaFree(d_g);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_gz);
    cudaFree(d_w);
  return;
}
    

