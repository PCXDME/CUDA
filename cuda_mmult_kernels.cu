#include "cuda_mmult_kernels.h"

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 4)
 *  -> Features: none (basic tiled version, using only global memory)
 */
__global__ void matrixMultKernel_global(float* Ad, float* Bd, float* Cd, int n)
{
   int i = blockIdx.x * TILE_SIZE + threadIdx.x;
   int k = blockIdx.y * TILE_SIZE + threadIdx.y;
   
   if(i < n && k < n) {
    float Celem = 0;
    
    for(int j=0; j<n; j++) {
        float Aelem = Ad[i*n+j];
        float Belem = Bd[j*n+k];
        Celem += Aelem*Belem;
    }
    
    Cd[i*n+k] += Celem;
   }
}

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 5)
 *  -> Features:
 *     - tiled matrix multiplication with use of shared memory
 */
__global__ void matrixMultKernel_tiled(float* Ad, float* Bd, float* Cd, int n)
{
   __shared__ float Ads[TILE_SIZE][TILE_SIZE];
   __shared__ float Bds[TILE_SIZE][TILE_SIZE];

   int tx = threadIdx.x;
   int ty = threadIdx.y;
   
   int i = blockIdx.x * TILE_SIZE + tx;
   int k = blockIdx.y * TILE_SIZE + ty;
   float Celem = 0;

   //Do for grid size
   for(int step = 0; step < gridDim.x; step++) {
        // Collaboratively copy Ad and Bd to Ads and Bds (shared memory)
        if((k) < n && (step * TILE_SIZE + tx) < n) {
            Ads[ty][tx] = Ad[ (k)                     * n + (step * TILE_SIZE + tx) ];
        }
        if((step * TILE_SIZE + ty) < n && (i) < n) {
            Bds[ty][tx] = Bd[ (step * TILE_SIZE + ty) * n + (i)                     ];
        }
        // Wait for every thread to finish copying
        __syncthreads();

        if(i < n && k < n) {
            for(int j = 0; j < TILE_SIZE; j++) {
                if((step * TILE_SIZE + j) < n) {
                    Celem += Ads[ty][j] * Bds[j][tx];
                }
            }
        }
        // Wait for every thread to finish computing
        __syncthreads();
    };

    if(i < n && k < n) {
        Cd[(k) * n + (i)] += Celem;
    }
}
