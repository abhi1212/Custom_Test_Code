//We are using C with 0 based Indexing



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

//N is the number of rows and M is the number of columns and NxM is our array.


int main (void){
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* devPtrA;
    float* a = 0;

    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }



    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {

            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);

	    printf("THe index in a is %d and the value at that index is %f \n",IDX2C(i,j,M), a[IDX2C(i,j,M)]);
        }

    }



    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
		//printf("%f ",a[(i*M)+j+1]);
	}
	printf("\n");
    }		


}
