/* My Code to check whether Cublas sgemm, Normal Sgemm, Shared Sgemm and Batched CUBLAS gives the same output
   All the matrices are without transpose
   Also we will be having a Matrix Transpose Kernel


Structure would be simple
1)Allocate Device and Host Data
2)Do a Basic Initialization from 0 to n
3)Call Cublas, normal sgemm, Shared sgemm, Batchjed Cublas Sgemm
4)After Cublas and normal gemm gives similar output schedule that for simulation.
5)Compare the Matrices.
6)Need to Check the Code For Different ALPHA and Beta Values
7)Consider Different Sizes
8)A big mistake was about to happen always multiply ALPHA and Beta and add previous C.

Steps to Consider--
1)All sizes of m,n and k
2)Values of alpha and beta 
3)Also check if transpose works with everything.
4)It works only with square matrices.


//Making it Rectangular




*/



/********************************* All the Header Files*******************************************/




/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <stdio.h>
#include "device_launch_parameters.h"


/* Matrix size */
#define N (500)
#define TILE_WIDTH 16
const int TILE_DIM = 32;
const int BLOCK_DIM = 16;


__global__ void custom_sgemm(int TA, int TB, int M, int n, int K, float ALPHA, 
float *A_gpu, int lda, 
float *B_gpu, int ldb,
float BETA,
float *C_gpu, int ldc);


/**************************************************************************************************/





/****************************************Global Memory Gemm Kernel*********************************/

/* THis version is used when both the Both the Matrices are Transposed*/




__global__ void custom_sgemm_tt(int TA, int TB, int M, int n, int K, float ALPHA, 
float *A_gpu, int lda, 
float *B_gpu, int ldb,
float BETA,
float *C_gpu, int ldc)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < K && row < M) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += A_gpu[row * n + i] * B_gpu[i * K + col];
        }
        C_gpu[row * K + col] = sum;
    }

}



/****************************************Global Memory Gemm Kernel*********************************/

/* THis version is used when both the Both the Matrices are Transposed*/




__global__ void custom_sgemm_nn(int TA, int TB, int M, int n, int K, float ALPHA, 
float *A_gpu, int lda, 
float *B_gpu, int ldb,
float BETA,
float *C_gpu, int ldc)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < K && row < M) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += A_gpu[row * n + i] * B_gpu[i * K + col];
        }
        C_gpu[row * K + col] = sum;		//Need to add plus here		
    }

}



/*******************************Shared Memory Matrix Multiplication***************************************/


__global__ void matrixMultiply(float * A, float * B, float * C,
	       int numARows, int numAColumns,
		       int numBRows, int numBColumns,
		       int numCRows, int numCColumns) {
	//@@ Insert code to implement matrix multiplication here
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x, by = blockIdx.y,
	tx = threadIdx.x, ty = threadIdx.y,
	Row = by * TILE_WIDTH + ty,
	Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
	if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
	  ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
	else
	  ds_M[ty][tx] = 0;
	if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
	  ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
	else
	  ds_N[ty][tx] = 0;

	__syncthreads();
	for (int k = 0; k < TILE_WIDTH; ++k)
	  Pvalue += ds_M[ty][k] * ds_N[k][tx];
	__syncthreads();
	}
	if (Row < numCRows && Col < numCColumns)
	C[Row*numCColumns+Col] = Pvalue;
	}



/******************************************************************TRanspose Kernel**************************************/



__global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}




/****************************************************  Main  **********************************************/


/* Main */
int main(int argc, char **argv)
{
    cublasStatus_t status;
    float *h_A;  		// Host Array A
    float *h_B;			//Host Array  B
    float *h_C;			//Host Array  C
    float *h_C_ref;		//Host Referrence Array
    float *d_A = 0;		//Device Array A
    float *d_B = 0;		//Device Array B
    float *d_C = 0;		//Device Array C
    float alpha = 1.0f;		
    float beta = 1.0f;

    int rows=8;
    int columns=3; 
    int n2 = rows * columns;		//Size of h_A. h_B, h_C



    int i;
    float error_norm;
    float ref_norm;
    float diff;
    cublasHandle_t handle;
    int dev = findCudaDevice(argc, (const char **) argv);	//If GPU exist


    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");

    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }





/****************Initialization Complete Now Allocate Memory******************************************/



    /* Allocate host memory for the matrices */
    h_A = (float *)malloc(n2 * sizeof(h_A[0]));


    if (h_A == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }

    h_B = (float *)malloc(n2 * sizeof(h_B[0]));

    if (h_B == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    h_C = (float *)malloc(n2 * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }


    
    /* Allocate device memory for the matrices */
    if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

   
    

    /************************************Memory Allocation Done, Now Initialization ****************************/ 



    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++)
    {
        h_A[i] = (float)i;			//rand() / (float)RAND_MAX;
        h_B[i] = (float)i;			//rand() / (float)RAND_MAX;
        h_C[i] = 0.0;			//rand() / (float)RAND_MAX;
    }

 

   



    /**************************************Cuda Memcopies********************************/

     /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");

        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }



    /********************************************Kernel Call to Cublas GEMM*************************/

    /* Performs operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    cudaDeviceSynchronize();


    /****************************************Kernel Call to Global Mem GEMM Transpose One ***************************/


    //Kernel Call to the Custom Function

  /*  const dim3 blocksize(32,16);
    const dim3 gridsize(N/blocksize.y +1,N/blocksize.x+1);
    custom_sgemm_tt<<<gridsize,blocksize>>>(0, 0, N, N, N, alpha, 
        d_A, N,d_B,N, 
        beta,
        d_C, N);
  
   cudaDeviceSynchronize();

*/


   /*************************************Kernel Call to Global Mem Non TRanspose One GEMM *******************************/

/*
 	custom_sgemm_nn<<<gridsize,blocksize>>>(0, 0, N, N, N, alpha, 
        d_A, N,d_B,N, 
        beta,
        d_C, N);
  
       cudaDeviceSynchronize();

*/


   /*************************************Kernel Call to Shared Memory GEMM *******************************/



  /*	dim3 dimGrid((N-1)/TILE_WIDTH+1, (N-1)/TILE_WIDTH+1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,
                                          N, N,
                                          N, N,
                                          N, N);

	cudaThreadSynchronize();
*/



  /**************************************KERNEL Call to TRanspose*******************************************/


	/*dim3 dimGrid(N/BLOCK_DIM+1, N/BLOCK_DIM+1, 1);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);

	transpose<<< dimGrid, dimBlock >>>(d_C, d_A,columns,rows);



	cudaThreadSynchronize();*/


    /*************************************************Getting The results back*************************/


    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);



    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }


    
    /***********************************************Printing the Matrices********************************************************/	



   int count1=0;
   int count2=0;
   int count=0;

   printf("The Input Matrix A is\n\n");
   for(i=0;i<(n2);i++)
	{
		if(count1==columns){
			printf("\n");
			count1=0;
			}
		count1=count1+1;	
		printf("%f ",h_A[i]);
	}
   printf("\n\n");

   printf("The Input Matrix B is\n");
   for(i=0;i<(n2);i++)
	{
		if(count2==columns){
			printf("\n");
			count2=0;
			}
		count2=count2+1;	
		printf("%f ",h_B[i]);
	}   

   printf("\n\n");





   printf("The output elements are\n");
    for(i=0;i<(n2);i++)
	{
		if(count==columns){
			printf("\n");
			count=0;
			}
		count=count+1;	
		printf("%f ",h_C[i]);
	}






    /* Check result against reference 
    error_norm = 0;
    ref_norm = 0;

    for (i = 0; i < n2; ++i)
    {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }*/

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
   //free(h_C_ref);

    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_B) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_C) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }

  /*  if (error_norm / ref_norm < 1e-6f)
    {
        printf("simpleCUBLAS test passed.\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        printf("simpleCUBLAS test failed.\n");
        exit(EXIT_FAILURE);
    }*/
}
