
#include <stdio.h>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h> 

//!*******************************************************************************
//! Upwinding scheme, neumann bc for phi
//!*******************************************************************************


__global__ void upwdker(int nx,int ny,int nz,double* d_phi,double* d_phix,double* d_phiy,double* d_phiz,double* d_phi2,double* d_dphi,double* d_w1,double* d_w2, double *wxy,double* d_g,double dt)
{//                                     [ , ,  ]      [ , , ]   [[,,],[,,],[,,]]
    int iz=blockIdx.x+1;int ix=threadIdx.x+1;int iy=threadIdx.y+1;

    
//    !phi_x by central difference
//    do iz = 2, nz - 1
//        do iy = 2, ny - 1
//            do ix = 2, nx - 1
//                phix(ix, iy, iz) = w1(-1)*phi(ix - 1, iy, iz) + w1(1)*phi(ix + 1, iy, iz)
//            enddo
//       enddo
//    enddo

    
    d_phix[ix*nz*ny+iy*nz+iz] = d_w1[0]*d_phi[(ix-1)*nz*ny+iy*nz+iz] + d_w1[2]*d_phi[(ix+1)*nz*ny+iy*nz+iz];
    d_phiy[ix*nz*ny+iy*nz+iz] = d_w1[0]*d_phi[ix*nz*ny+(iy-1)*nz+iz] + d_w1[2]*d_phi[ix*nz*ny+(iy+1)*nz+iz];
    d_phiz[ix*nz*ny+iy*nz+iz] = d_w1[0] *d_phi[ix*nz*ny+iy*nz+iz-1] + d_w1[2] * d_phi[ix*nz*ny+iy*nz+ iz + 1];
/*    !phi_y by central difference
    do iz = 2, nz - 1
        do ix = 2, nx - 1
            do iy = 2, ny - 1
                phiy(ix, iy, iz) = w1(-1) * phi(ix, iy - 1, iz) + w1(1) * phi(ix, iy + 1, iz)
            enddo
        enddo
    enddo
    
    !phi_z by central difference
    do iy = 2, ny - 1
        do ix = 2, nx - 1
            do iz = 2, nz - 1
                phiz(ix, iy, iz) = w1(-1) * phi(ix, iy, iz - 1) + w1(1) * phi(ix, iy, iz + 1)
            enddo
        enddo
    enddo

    !phi_zz by central difference
    do iy = 2, ny - 1
        do ix = 2, nx - 1
            do iz = 2, nz - 1
                phi2(ix, iy, iz) = w2(-1)*phi(ix, iy, iz-1) + w2(0)*phi(ix, iy, iz) + w2(1)*phi(ix, iy, iz+1)
            enddo
        enddo
    enddo
*/
    d_phi2[ix*nz*ny+iy*nz+iz] = d_w2[0] *d_phi[ix*nz*ny+iy*nz+iz-1] + d_w2[1] * d_phi[ix*nz*ny+iy*nz+ iz ] + d_w2[2] * d_phi[ix*nz*ny+iy*nz+ iz + 1];

/*
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2, nx - 1
                dphi(ix, iy, iz) = (1.0d0 + phix(ix, iy, iz)**2 + phiy(ix, iy, iz)**2)*phi2(ix, iy, iz)
            enddo
        enddo
    enddo
*/
    __syncthreads();
    d_dphi[ix*nz*ny+iy*nz+iz]=(1+d_phix[ix*nz*ny+iy*nz+iz]*d_phix[ix*nz*ny+iy*nz+iz] 
        + d_phiy[ix*nz*ny+iy*nz+iz]*d_phiy[ix*nz*ny+iy*nz+iz])  *  d_phi2[ix*nz*ny+iy*nz+iz];
  

/*    !phi_yy by central difference
    do iz = 2, nz - 1
        do ix = 2, nx - 1
            do iy = 2, ny - 1
                phi2(ix, iy, iz) = w2(-1)*phi(ix, iy-1, iz) + w2(0)*phi(ix, iy, iz) + w2(1)*phi(ix, iy+1, iz)
            enddo
        enddo
    enddo
*/
    __syncthreads();
    d_phi2[ix*nz*ny+iy*nz+iz]= d_w2[0]*d_phi[ix*nz*ny+(iy-1)*nz+iz] + d_w2[1]*d_phi[ix*nz*ny+iy*nz+iz] + d_w2[2]*d_phi[ix*nz*ny+(iy+1)*nz+iz];

/*    
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2, nx - 1
                dphi(ix, iy, iz) = dphi(ix, iy, iz) + phi2(ix, iy, iz)*(1.0d0 + phix(ix, iy, iz)**2 + phiz(ix, iy, iz)**2)
            enddo
        enddo
    enddo
*/
    __syncthreads();
    d_dphi[ix*nz*ny+iy*nz+iz] = d_dphi[ix*nz*ny+iy*nz+iz] + d_phi2[ix*nz*ny+iy*nz+iz]*(1.0 + d_phix[ix*nz*ny+iy*nz+iz]*d_phix[ix*nz*ny+iy*nz+iz] + d_phiz[ix*nz*ny+iy*nz+iz]*d_phiz[ix*nz*ny+iy*nz+iz]);

/*
    !phi_xx by central difference
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2, nx - 1
                phi2(ix, iy, iz) = w2(-1)*phi(ix-1, iy, iz) + w2(0)*phi(ix, iy, iz) + w2(1)*phi(ix+1, iy, iz)
            enddo
        enddo
    enddo
*/
    __syncthreads();
    d_phi2[ix*nz*ny+iy*nz+iz]  = d_w2[0]*d_phi[(ix-1)*nz*ny+iy*nz+iz]+ d_w2[1]*d_phi[ix*nz*ny+iy*nz+iz] + d_w2[2]*d_phi[(ix+1)*nz*ny+iy*nz+iz];
/*    
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2, nx - 1
                dphi(ix, iy, iz) = dphi(ix, iy, iz) + phi2(ix, iy, iz)*(1.0d0 + phiy(ix, iy, iz)**2 + phiz(ix, iy, iz)**2)
            enddo
        enddo
    enddo
*/
    __syncthreads();
    d_dphi[ix*nz*ny+iy*nz+iz]= d_dphi[ix*nz*ny+iy*nz+iz]+ d_phi2[ix*nz*ny+iy*nz+iz]*(1.0 + d_phiy[ix*nz*ny+iy*nz+iz]*d_phiy[ix*nz*ny+iy*nz+iz]+ d_phiz[ix*nz*ny+iy*nz+iz]*d_phiz[ix*nz*ny+iy*nz+iz]);
    
/*
    !phi_xz by central difference
    do iy = 2, ny - 1
        do iz = 2, nz - 1
            do ix = 2, nx - 1
                phi2(ix, iy, iz) = wxy(-1, -1)*phi(ix-1, iy, iz-1) + wxy(-1, 1)*phi(ix-1, iy, iz+1)&
                           + wxy(1, -1)*phi(ix+1, iy, iz-1) + wxy(1, 1)*phi(ix+1, iy, iz+1)
            enddo
        enddo
    enddo
*/
    __syncthreads();
    d_phi2[ix*nz*ny+iy*nz+iz] = wxy[0]*d_phi[(ix-1)*nz*ny+iy*nz+iz-1]+wxy[2] *d_phi[(ix-1)*nz*ny+iy*nz+iz+1]+wxy[6]*d_phi[(ix+1)*nz*ny+iy*nz+iz-1]+wxy[8]*d_phi[(ix+1)*nz*ny+iy*nz+iz+1];
/*
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2, nx - 1
                dphi(ix, iy, iz) = dphi(ix, iy, iz) - 2.0d0*phi2(ix, iy, iz)*phix(ix, iy, iz)*phiz(ix, iy, iz)
            enddo
        enddo
    enddo
*/
    
    __syncthreads();
    d_phi[ix*nz*ny+iy*nz+iz] = d_phi[ix*nz*ny+iy*nz+iz] -2*d_phi2[ix*nz*ny+iy*nz+iz]*d_phix[ix*nz*ny+iy*nz+iz]*d_phiz[ix*nz*ny+iy*nz+iz];

/*
    !phi_xy by central difference
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2, nx - 1
                phi2(ix, iy, iz) = wxy(-1, -1)*phi(ix-1, iy-1, iz) + wxy(-1, 1)*phi(ix-1, iy+1, iz)&
                        + wxy(1, -1)*phi(ix+1, iy-1, iz) + wxy(1, 1)*phi(ix+1, iy+1, iz)
            enddo
        enddo
    enddo
*/
    __syncthreads();
    d_phi2[ix*nz*ny+iy*nz+iz] = wxy[0]*d_phi[(ix-1)*nz*ny+(iy-1)*nz+iz] +wxy[2]*d_phi[(ix-1)*nz*ny+(iy+1)*nz+iz]+wxy[6]*d_phi[(ix+1)*nz*ny+(iy-1)*nz+iz]+wxy[8]*d_phi[(ix+1)*nz*ny+(iy+1)*nz+iz];
/*
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2, nx - 1
                dphi(ix, iy, iz) = dphi(ix, iy, iz) - 2.0d0*phi2(ix, iy, iz)
*phix(ix, iy, iz)*            phiy(ix, iy, iz)
            enddo
        enddo
    enddo
  */
    __syncthreads();
    d_dphi[ix*nz*ny+iy*nz+iz] = d_dphi[(ix)*nz*ny+(iy)*nz+iz] -2*d_phi2[(ix)*nz*ny+(iy)*nz+iz]
*d_phix[(ix)*nz*ny+(iy)*nz+iz]*d_phiy[ix*nz*ny+iy*nz+iz];
/*
    !phi_yz by central difference
    do ix = 2, nx - 1
        do iz = 2, nz - 1
            do iy = 2, ny - 1
                phi2(ix, iy, iz) = wxy(-1, -1)*phi(ix, iy-1, iz-1) + wxy(-1, 1)*phi(ix, iy-1, iz+1)&
                          + wxy(1, -1)*phi(ix, iy+1, iz-1) + wxy(1, 1)*phi(ix, iy+1, iz+1)
            enddo
        enddo
    enddo
*/
// no dependency
    d_phi2[ix*nz*ny+iy*nz+iz] = wxy[0]*d_phi[(ix)*nz*ny+(iy-1)*nz+iz-1] +wxy[2]*d_phi[ix*nz*ny+(iy-1)*nz+iz+1]+wxy[6]*d_phi[(ix)*nz*ny+(iy+1)*nz+iz-1]+wxy[8]*d_phi[(ix)*nz*ny+(iy+1)*nz+iz+1];
/*
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2, nx - 1
                dphi(ix, iy, iz) = (   dphi(ix, iy, iz) - 
          2.0d0*phi2(ix, iy, iz) * phiy(ix,iy,iz)*phiz(ix,iy,iz)   )  *g(ix,iy,iz)&
                            /(1.0d0+phix(ix, iy, iz)**2 + phiy(ix, iy, iz)**2 + phiz(ix, iy, iz)**2)
            enddo
        enddo
    enddo
    
*/

    __syncthreads();
    d_dphi[ix*nz*ny+iy*nz+iz] = (   d_dphi[ix*nz*ny+iy*nz+iz]- 
        2*d_phi2[ix*nz*ny+iy*nz+iz]  *d_phiy[(ix)*nz*ny+(iy)*nz+iz]  )  *  d_g[ix*nz*ny+iy*nz+iz]
                         /(   1+d_phix[ix*nz*ny+iy*nz+iz]*d_phix[ix*nz*ny+iy*nz+iz]*+d_phiy[ix*nz*ny+iy*nz+iz]*d_phiy[ix*nz*ny+iy*nz+iz] +d_phiz[ix*nz*ny+iy*nz+iz]*d_phiz[ix*nz*ny+iy*nz+iz]   );
/*
    do iz = 2, nz - 1
        do iy = 2, ny - 1
            do ix = 2,nx - 1
                phi(ix, iy, iz) = phi(ix, iy, iz) + dt*dphi(ix, iy, iz)
            enddo
        enddo
    enddo
*/
    __syncthreads();
    d_phi[ix*nz*ny+iy*nz+iz] = d_phi[(ix)*nz*ny+(iy)*nz+iz] +dt*d_dphi[(ix)*nz*ny+(iy)*nz+iz];
    return;

}


extern "C" void  upwinding_(int* nx,int* ny,int* nz,double* dx,double* dt,double* g,double* gx,double* gy,double* gz,double* phi)
{
//    implicit real*8 (a-h, o-z)
//    dimension g(nx, ny, nz), gx(nx, ny, nz), gy(nx, ny, nz), gz(nx, ny, nz)
  //  dimension phi(nx, ny, nz), dphi(nx, ny, nz), phi2(nx, ny, nz)
 //   dimension phix(nx, ny, nz), phiy(nx, ny, nz), phiz(nx, ny, nz)
 //   dimension w1(-1:1), w2(-1:1), wxy(-1:1, -1:1), up(0:1), down(-1:0)
   
 //   w1(-1) = -5.d-1/dx; w1(0) = 0.0d0; w1(1) = 5.d-1/dx
    double w1[3]={-0.5/ *dx , 0 , 0.5/ *dx};
 //   w2(-1) = 1.0d0/dx/dx; w2(0) = -2.0d0/dx/dx; w2(1) = 1.0d0/dx/dx
    double tmp=1/ *dx/ *dx;
    double w2[3]={tmp,-2*tmp,tmp} ;
  //  wxy(-1, -1) = 2.5d-1/dx/dx; wxy(-1, 0) = 0.0d0; 	wxy(-1, 1) = -2.5d-1/dx/dx
//    wxy(0, -1) = 0.0d0; 	wxy(0, 0) = 0.0d0; 	wxy(0, 1) = 0.0d0
//    wxy(1, -1) = -2.5d-1/dx/dx; wxy(1, 0) = 0.0d0; 	wxy(1, 1) = 2.5d-1/dx/dx
//    		0,0 	0,1 	0,2
//		1,0	1,1	1,2
//  		2,0	2,1	2,2
    double wxy[9] ={0.25*tmp,   0, -0.25*tmp ,
                     0 ,        0, 0         ,
                    -0.25*tmp ,0, 0.25*tmp  };  
    
    double up[2] = {-1/ *dx , 1/ *dx};
    double down[2]={ 1.0/ *dx, -1.0/ *dx};
    double *phi2;
    double *d_phi,*d_phix,*d_phiy,*d_phiz,*d_phi2,*d_dphi,*d_w1,*d_w2,*d_g;
    int N=(*nx) * (*ny) * (*nz);
      /* Allocate complex array on device */
    cudaMalloc ((void **) &d_phi , sizeof(double)*N);
    cudaMalloc ((void **) &d_phix , sizeof(double)*N);
    cudaMalloc ((void **) &d_phiy , sizeof(double)*N);
    cudaMalloc ((void **) &d_phiz , sizeof(double)*N);
    cudaMalloc ((void **) &d_phi2 , sizeof(double)*N);
    cudaMalloc ((void **) &d_dphi , sizeof(double)*N);
    cudaMalloc ((void **) &d_w1 , sizeof(double)*N);
    cudaMalloc ((void **) &d_w2 , sizeof(double)*N);
    cudaMalloc ((void **) &d_g , sizeof(double)*N);
      /* Copy array from host memory to device memory */
    cudaMemcpy( d_phi, phi,  sizeof(double)*N,cudaMemcpyHostToDevice);

      /* Compute execution configuration */
    dim3 dimGrid(*nz);dim3 dimBlock(*nx,*ny);

    upwdker<<<dimGrid,dimBlock>>>( *nx, *ny, *nz, d_phi, d_phix, d_phiy, d_phiz, d_phi2, d_dphi, d_w1, d_w2,wxy,d_g,*dt);
//__global__ void upwdker(int nx,int ny,int nz,double* d_phi,double* d_phix,double* d_phiy,double* d_phiz,double* d_phi2,double* d_dphi,double* d_w1,double* d_w2, double *wxy,double* d_g,double dt)

    /* Copy the result back */
   cudaMemcpy(phi, d_phi, sizeof(double)*N,cudaMemcpyDeviceToHost);  

    /* Free memory on the device */
   cudaFree(d_phi );
   cudaFree(d_phix );
   cudaFree(d_phiy );
   cudaFree(d_phiz );
   cudaFree(d_phi2 );
   cudaFree(d_dphi );
   cudaFree(d_w1 );
   cudaFree(d_w2 );
   cudaFree(d_g );
    return;

}
