#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
    
}



void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    //#pragma omp parallel for
    printf("I am here");
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
		//printf("%f \n",	C[i*ldc+j]);
            }	
        }
    }
	 
}


void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}




int main()
{
	float *h_A;  		// Host Array A
	float *h_A_T;		//Host Transpose Array
	float *h_B;			//Host Array  B
	float *h_B_T;		//Host Array  B Transpose
	float *h_C;			//Host Array  C
	float alpha = 1.0f;		
	float beta = 1.0f;
	int j=0;
	int m=32;
	int k=2700;
	int n=36;
	int i;

	int size_a=m*k;
	int size_b=k*n;
	int size_c=(m*n);

	int lda=m;
	int ldb=k;
	int ldc=m;
	
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
	if (h_C == 0)
	{
		fprintf(stderr, "!!!! host memory allocation error (C)\n");
		return EXIT_FAILURE;
	}


	for (i = 0; i < m; i++)
    {
	 for(j = 0; j < k; j++)	
	 {
        	h_A[(i*k)+j] = ((i*k)+j);	
		h_A_T[(i*k)+j] = 0;
	 }
    }
 

    /* Fill the matrices with test data */	//Allocation for A
     for (i = 0; i < k; i++)
    {
	 for(j = 0; j < n; j++)	
	 {	
        	h_B[(i*n)+j] = ((i*n)+j);	
		h_B_T[(i*n)+j] = 0;		
	 }
    }
 

    /* Fill the matrices with test data */	//Allocation for A
    for (i = 0; i < m; i++)
    {
	 for(j = 0; j < n; j++)	
	 {
        	h_C[(i*n)+j] = 1;
			
	 }
    }


	
	gemm(0,0,m,n,k,1,h_A,k,h_B,n,1,h_C,n);


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

   printf("\n\n");*/

    printf("The output elements after Cublas GEMM are\n");
    for(i=0;i<(size_c);i++)
	{
		if(count==n){
			printf("\n");
			count=0;
			}
		count=count+1;	
		printf("%f ",h_C[i]);
	}



}







