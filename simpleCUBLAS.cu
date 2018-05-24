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


//Making it Rectangular Done
//Trying without initializing lda,ldb,ldc

//Can be issues with Tranpose kernel Launches




*/



/********************************* All the Header Files*******************************************/




/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include <helper_cuda.h>
#include <stdio.h>
//#include "device_launch_parameters.h"


/* Matrix size */
#define N (500)
#define TILE_WIDTH 16
const int TILE_DIM = 32;
const int BLOCK_DIM = 16;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


__global__ void custom_sgemm(int TA, int TB, int M, int n, int K, float ALPHA, 
float *A_gpu, int lda, 
float *B_gpu, int ldb,
float BETA,
float *C_gpu, int ldc);


/**************************************************************************************************/

/****************************************Sgemm Kernel*********************************/

/* Consider the following scenario 
  
Matrix A   Matrix B   Matrix C
m*k	   k*n	      m*n
5*4	   4*3	      5*3

Matrix A   Matrix B   Matrix C
n*k	   k*m        n*m 
3*4        4*5        3*5

				*/


__global__ void blas_sgemm(int TA, int TB, int m, int k, int n, float ALPHA, 
float *A_gpu, int lda, 
float *B_gpu, int ldb,
float BETA,
float *C_gpu, int ldc)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y; //Row of output Matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x; //Column of Output Matrix
    int sum = 0;


    if( col < m && row < n)   //Now as my Matrix is n*m 
    {
        for(int i = 0; i < k; i++) 
        {
            sum += A_gpu[row * k + i] * B_gpu[i * m + col];
        }
        C_gpu[row * m + col] = sum;
    }

}



/****************************************Global Memory Gemm Kernel*********************************/

/* THis version is used when both the Both the Matrices are Transposed*/




__global__ void custom_sgemm_tt(int TA, int TB, int m, int k, int n, float ALPHA, 
float *A_gpu, int lda, 
float *B_gpu, int ldb,
float BETA,
float *C_gpu, int ldc)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < n && row < m) 
    {
        for(int i = 0; i < k; i++) 
        {
            sum += A_gpu[row * k + i] * B_gpu[i * n + col];
        }
        C_gpu[row * n + col] = sum;
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






/**********************************************************GEMM********************************************/

__global__ void gemm_kernel(float *a, float *b, float *c,int NI, int NK, int NJ,float ALPHA, float BETA)  
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int row=0;
	int column=0;
        int size_a=NI*NK;

/*	if(i==0 && j==0) 			//For this condition Test- for device array data ordering
	{
                int count=0;
		for(row=0;row<size_a;row++)
		{
			printf("%f ",a[row]);
			count+=1;
			if(count==NK)
			{
				printf("\n");
				count=0;
			}
		}	
		

	}*/

	if ((i < NI) && (j < NJ))
	{	
		c[i * NJ + j] = c[i * NJ + j]*BETA;

		int k;
		for(k=0; k < NK; k++)
		{
			c[i * NJ + j] += ALPHA * a[i * NI + k] * b[k * NK +j];
		}
	}
}



/****************************************************  Main  **********************************************/


/* Main */
int main(int argc, char **argv)
{
    cublasStatus_t status;
    float *h_A;  		// Host Array A
    float *h_A_T;		//Host Transpose Array
    float *h_B;			//Host Array  B
    float *h_B_T;		//Host Array  B Transpose
    float *h_C;			//Host Array  C
    float *h_C_MM;
    float *h_C_ref;		//Host Referrence Array
    float *d_A = 0;		//Device Array A
    float *d_A_T = 0;		//Transpose Device Array
    float *d_B =  0;		//Device Array B
    float *d_B_T= 0;		//Transpose Device array B
    float *d_C = 0;		//Device Array C
    float *d_C_MM=0;
    float alpha = 1.0f;		
    float beta = 1.0f;
    int j=0;

    int m=80;
    int k=40;
    int n=50;
 
    int size_a=m*k;
    int size_b=k*n;
    int size_c=(m*n);
    //int n2 = rows * columns;		//Size of h_A. h_B, h_C

    //Considering the case when it is non transpose

    int lda=m;
    int ldb=k;
    int ldc=m;

    int i;
    float error_norm;
    float ref_norm;
    float diff;
    cublasHandle_t handle;
   /* int dev = findCudaDevice(argc, (const char **) argv);	//If GPU exist


    if (dev == -1)
    {
        return EXIT_FAILURE;
    }*/

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
    h_A = (float *)malloc(size_a * sizeof(h_A[0]));
    h_A_T = (float *)malloc(size_a * sizeof(h_A[0]));

    if (h_A == 0 ||h_A_T == 0 )
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }

    h_B = (float *)malloc(size_b * sizeof(h_B[0]));
    h_B_T = (float *)malloc(size_b * sizeof(h_B[0]));

    if (h_B == 0 || h_B_T== 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    h_C = (float *)malloc(size_c * sizeof(h_C[0]));
    h_C_MM = (float *)malloc(size_c * sizeof(h_C[0]));
    if (h_C == 0 && h_C_MM == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }


    
    /* Allocate device memory for the matrices */
    if (cudaMalloc((void **)&d_A, size_a * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_A_T, size_a * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }


    if (cudaMalloc((void **)&d_B, size_b * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_B_T, size_b * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }




    if (cudaMalloc((void **)&d_C, size_c * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_C_MM, size_c * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    

    /************************************Memory Allocation Done, Now Initialization ****************************/ 



     /* Fill the matrices with test data */	//Allocation for A
    for (i = 0; i < m; i++)
    {
	 for(j = 0; j < k; j++)	
	 {
        	h_A[(i*k)+j] = (i*k)+j;	
		h_A_T[(i*k)+j] = 0;
	 }
    }
 

    /* Fill the matrices with test data */	//Allocation for A
     for (i = 0; i < k; i++)
    {
	 for(j = 0; j < n; j++)	
	 {	
        	h_B[(i*n)+j] = (i*n)+j;	
		h_B_T[(i*n)+j] = 0;		
	 }
    }
 

    /* Fill the matrices with test data */	//Allocation for A
    for (i = 0; i < m; i++)
    {
	 for(j = 0; j < n; j++)	
	 {
        	h_C[(i*n)+j] = 0;
		h_C_MM[(i*n)+j] = 0;		
	 }
    }


  

    /**************************************Cuda Memcopies********************************/

	//Try doing a normal memcopy, try with set matrix and get matrix 

     /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(size_a, sizeof(h_A[0]), h_A, 1, d_A, 1);
   // cublasSetVector(size_a, sizeof(h_A[0]), h_A_T, 1, d_A_T, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }


    status = cublasSetVector(size_b, sizeof(h_B[0]), h_B, 1, d_B, 1);
    //status = cublasSetVector(size_b, sizeof(h_B[0]), h_B_T, 1, d_B_T, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");

        return EXIT_FAILURE;
    }

    status = cublasSetVector(size_c, sizeof(h_C[0]), h_C, 1, d_C, 1);
    status = cublasSetVector(size_c, sizeof(h_C[0]), h_C_MM, 1, d_C_MM, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

/*
    
  /**************************************KERNEL Call to TRanspose*******************************************/


/*	dim3 dimGrid(m/BLOCK_DIM+1, n/BLOCK_DIM+1, 1);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);

	transpose<<< dimGrid, dimBlock >>>(d_A_T, d_A,k,m);

	cudaThreadSynchronize();

	dim3 dimGrid1(k/BLOCK_DIM+1, n/BLOCK_DIM+1, 1);
	dim3 dimBlock1(BLOCK_DIM, BLOCK_DIM, 1);

	transpose<<< dimGrid1, dimBlock1 >>>(d_B_T, d_B,n,k);

	cudaThreadSynchronize();

*/




    /********************************************Kernel Call to Cublas GEMM*************************/

    /* Performs operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    cudaDeviceSynchronize();


    /****************************************Kernel Call to Global Mem GEMM Transpose One ***************************/


    //Kernel Call to the Custom Function
    //The threads launched should be enough to size m*n we need threads equal to m*n.

   /* const dim3 blocksize(32,32);// Block of 32*32 threads
    const dim3 gridsize(n/blocksize.y +1,m/blocksize.x+1);//Number of blocks in each direction
    custom_sgemm_tt<<<gridsize,blocksize>>>(0, 0, m, k, n, alpha, 
        d_A, N,d_B,N, 
        beta,
        d_C_MM, N);
  
   cudaDeviceSynchronize();
    */


  /****************************************Kernel Call to Blas_sgemm ***************************/


    //The threads launched should be enough to size m*n we need threads equal to m*n.+

    const dim3 blocksize(32,32);// Block of 32*32 threads
    const dim3 gridsize(n/blocksize.y +1,m/blocksize.x+1);//Number of blocks in each direction. // Make sure what are these
    blas_sgemm<<<gridsize,blocksize>>>(0, 0, m, k, n, alpha, 
        d_B, N,d_A,N, 
        beta,
        d_C_MM, N);
  
   cudaDeviceSynchronize();




   /*************************************Kernel Call to Global Mem Non TRanspose One GEMM *******************************/



        
 	/*custom_sgemm_nn<<<gridsize,blocksize>>>(0, 0, N, N, N, alpha, 
        d_A, N,d_B,N, 
        beta,
        d_C, N);
  
       cudaDeviceSynchronize();*/




   /*************************************Kernel Call to Shared Memory GEMM *******************************/



  /*	dim3 dimGrid((N-1)/TILE_WIDTH+1, (N-1)/TILE_WIDTH+1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,
                                          N, N,
                                          N, N,
                                          N, N);

	cudaThreadSynchronize();
*/




   /***********************************GEMM Function Call***************************************************/

      /*  dim3 dimGrid(m/BLOCK_DIM+1, n/BLOCK_DIM+1, 1);
	dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);

	gemm_kernel<<< dimGrid, dimBlock >>>(d_A, d_B,d_C,m,k,n,alpha,beta);
        


	cudaThreadSynchronize();*/
    





    /*************************************************Getting The results back*************************/
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				

    /* Read the result back */

    // cublasGetVector(size_a, sizeof(h_A[0]), d_A_T, 1, h_A_T, 1);
    //cublasGetVector(size_b, sizeof(h_B[0]), d_B_T, 1, h_B_T, 1);

   

    status = cublasGetVector(size_c, sizeof(h_C[0]), d_C, 1, h_C, 1);
    status = cublasGetVector(size_c, sizeof(h_C[0]), d_C_MM, 1, h_C_MM, 1);



    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }







    
    /***********************************************Printing the Matrices********************************************************/	



   int count1=0;
   int count2=0;
   int count=0;
   int count3=0;

 /* printf("The Input Matrix A is\n\n");
   for(i=0;i<(size_a);i++)
	{
		if(count1==k){
			printf("\n");
			count1=0;
			}
		count1=count1+1;	
		printf("%f ",h_A[i]);
	}
   printf("\n\n");

   printf("The Input Matrix B is\n");
   for(i=0;i<(size_b);i++)
	{
		if(count2==n){
			printf("\n");
			count2=0;
			}
		count2=count2+1;	
		printf("%f ",h_B[i]);
	}   

   printf("\n\n");


  /* printf("The Transpose Matrix A is\n\n");

   for(i=0;i<(size_a);i++)
	{
		if(count1==k){
			printf("\n");
			count1=0;
			}
		count1=count1+1;	
		printf("%f ",h_A_T[i]);
	}
   printf("\n\n");
	

   printf("The Transpose Matrix B is\n");
   for(i=0;i<(size_b);i++)
	{
		if(count2==n){
			printf("\n");
			count2=0;
			}
		count2=count2+1;	
		printf("%f ",h_B_T[i]);
	}   */

   /*printf("\n\n");



   printf("The output elements after Cublas GEMM are\n");
    for(i=0;i<(size_c);i++)
	{
		if(count==m){
			printf("\n");
			count=0;
			}
		count=count+1;	
		printf("%f ",h_C[i]);
	}

   printf("\n\n");*/
   printf("The output element difference after GEMM is\n");
    for(i=0;i<(size_c);i++)
	{
		if(count3==n){
			printf("\n");
			count3=0;
			}
		count3=count3+1;	
		printf("%f ",h_C_MM[i]);
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
