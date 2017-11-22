#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>

#define TILE_SIZE 512
#define WARP_SIZE 32


extern "C" void CSRmatvecmult(int* start, int* J, float* Val, int N, int nnz, float* x, float *y, bool bVectorized);
extern "C" void ELLmatvecmult(int N, int num_cols_per_row , int * indices, float * data , float * x , float * y);

/**
 * Custom CUDA error check wrapper.
 */
#define checkCUDAError() do {                           \
 cudaError_t error = cudaGetLastError();               \
 if (error != cudaSuccess) {                            \
   printf("(CUDA) %s", cudaGetErrorString(error)); \
   printf(" (" __FILE__ ":%d)\n", __LINE__);  \
  }\
} while (0)

/**
 * Cuda kernel for: CSR_s(A)x = y
 */
__global__ void k_csr_mat_vec_mm(int *start, int* j, float *a_content, int num_rows, float *x, float* y) {
    // TODO: implement the scalar crs kernel
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	float result = 0.0f;

	for(int i=start[row];i<start[row+1];i++) {
		int value = a_content[i];
		int column = j[i];

		result += x[column] * value;
	}

	y[row] += result;
}

/**
 * Cuda kernel for: CSR_v(A)x = y
 */
__global__ void k_csr2_mat_vec_mm(int *start, int* j, float *a_content, int num_rows, float *x, float* y) {
	//TODO: implement the vectorized csr kernel
}

/**
 * Cuda kernel for: ELL(A)x = y
 */
__global__ void k_ell_mat_vec_mm ( int N, int num_cols_per_row , int * indices,
									float * data , float * x , float * y ) {
	//NYI: ellpack kernel
}

/**
 * Perform: CSR(A)x = y
 */
void CSRmatvecmult(int* start, int* J, float* Val, int N, int nnz, float* x, float *y, bool bVectorized) {
	int *start_d, *J_d;
	float *Val_d, *x_d, *y_d;

	/************************/
	/* copy to device       */
	/************************/

	cudaMalloc((void **) &start_d, (N+1) * sizeof(int));
	checkCUDAError();
	cudaMemcpy(start_d, start, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) &J_d, nnz * sizeof(int));
	checkCUDAError();
	cudaMemcpy(J_d, J, nnz * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) &Val_d, nnz * sizeof(float));
	checkCUDAError();
	cudaMemcpy(Val_d, Val, nnz * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) &x_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) &y_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(y_d, y, N * sizeof(float) , cudaMemcpyHostToDevice);
	checkCUDAError();

	/************************/
	/* start kernel         */
	/************************/

	if (bVectorized) {
		//TODO: define grid and block size correctly
		dim3 grid(1, 1, N/TILE_SIZE);
		dim3 block(1, 1, TILE_SIZE);

		k_csr2_mat_vec_mm <<< grid, block >>> (start_d, J_d, Val_d, N, x_d, y_d);
	} else {
		dim3 grid((N - 1)/TILE_SIZE + 1, 1, 1);
		dim3 block(TILE_SIZE, 1, 1);

		k_csr_mat_vec_mm <<< grid, block >>> (start_d, J_d, Val_d, N, x_d, y_d);
	}

	checkCUDAError();

	/************************/
	/* copy back            */
	/************************/

	cudaMemcpy(y, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError();

	/************************/
	/* free memory          */
	/************************/
	cudaFree(start_d);
	cudaFree(J_d);
	cudaFree(Val_d);
	cudaFree(x_d);
	cudaFree(y_d);
}

/**
 * Perform: ELL(A)x = y
 */
void ELLmatvecmult(int N, int num_cols_per_row , int * indices,
		float * data , float * x , float * y) {
	int *indices_d;
	float *data_d, *x_d, *y_d;

	/************************/
	/* copy to device       */
	/************************/

	cudaMalloc((void **) &indices_d, N * num_cols_per_row * sizeof(int));
	checkCUDAError();
	cudaMemcpy(indices_d, indices, N * num_cols_per_row * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) &data_d, N * num_cols_per_row * sizeof(float));
	checkCUDAError();
	cudaMemcpy(data_d, data, N * num_cols_per_row * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) &x_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) &y_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	/************************/
	/* start kernel         */
	/************************/

	//NYI: define grid and block size
	//k_ell_mat_vec_mm <<< grid, block >>> (N, num_cols_per_row, indices_d, data_d , x_d, y_d);
	checkCUDAError();

	/************************/
	/* copy back            */
	/************************/

	cudaMemcpy(y, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError();

	/************************/
	/* free memory          */
	/************************/

	cudaFree(indices_d);
	cudaFree(data_d);
	cudaFree(x_d);
	cudaFree(y_d);
}

