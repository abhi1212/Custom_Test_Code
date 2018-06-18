
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C"
{
#include "sgemm.h"
#include "cuda.h"
#include "utils.h"
#include "gemm.h"
}


__global__ void sgemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{

	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	if( col < N && row < M) 
	{
		for(int i = 0; i < K; i++) 
		{		//n*k                     k*m
			sum += (ALPHA *A_gpu[row * K + i]) * B_gpu[i * N + col];
		}
		C_gpu[row * N + col]+= sum;
	}	

}


void sgemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    //printf("Cublas has started Successfully\n");
    //printf("Printing out the parameters\n");
      printf("Gpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);

   // printf("These are the calls to gemm gpu\n");
    const dim3 blocksize(32,32);
    const dim3 gridsize(N/blocksize.y +1,M/blocksize.x+1);
    sgemm<<<gridsize,blocksize>>>(TA, TB, M, N, K, ALPHA, 
        A_gpu, lda, 
        B_gpu, ldb,
        BETA,
        C_gpu, ldc);

    cudaThreadSynchronize();
    check_error(cudaPeekAtLastError());
  //printf("Cublas has ended Successfully\n");
}




















