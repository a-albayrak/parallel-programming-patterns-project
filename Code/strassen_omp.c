#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>              
#include <math.h>            
#define THRESHOLD 16          

/*-------------------------------------------------------------
 * Utility Functions for Matrices
 *-------------------------------------------------------------*/
double** allocate_matrix(int n) {
    double **matrix = (double **)malloc(n * sizeof(double *));
    if (!matrix) {
        fprintf(stderr, "Memory allocation failed (matrix rows).\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        matrix[i] = (double *)calloc(n, sizeof(double)); 
        if (!matrix[i]) {
            fprintf(stderr, "Memory allocation failed (matrix cols).\n");
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

void free_matrix(double **matrix, int n) {
    if (!matrix) return;
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


double matrix_add(double **A, double **B, double **C, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
            sum += C[i][j];  
        }
    }

    return sum; 
}

double matrix_sub(double **A, double **B, double **C, int n) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] - B[i][j];  
            sum += C[i][j];  
        }
    }

    return sum;
}

/*-------------------------------------------------------------
 * Strassen's Algorithm (Parallel)
 *-------------------------------------------------------------*/
void strassen_multiply_internal(double **A, double **B, double **C, int n) {
    if (n <= THRESHOLD) {
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                double sum = 0.0;
                for (int k = 0; k < n; k++){
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return;
    }
    
    // Base case
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }
    
    int k = n / 2;
    
    double **A11 = allocate_matrix(k);
    double **A12 = allocate_matrix(k);
    double **A21 = allocate_matrix(k);
    double **A22 = allocate_matrix(k);
    
    double **B11 = allocate_matrix(k);
    double **B12 = allocate_matrix(k);
    double **B21 = allocate_matrix(k);
    double **B22 = allocate_matrix(k);
    
    double **C11 = allocate_matrix(k);
    double **C12 = allocate_matrix(k);
    double **C21 = allocate_matrix(k);
    double **C22 = allocate_matrix(k);
    
    double **M1 = allocate_matrix(k);
    double **M2 = allocate_matrix(k);
    double **M3 = allocate_matrix(k);
    double **M4 = allocate_matrix(k);
    double **M5 = allocate_matrix(k);
    double **M6 = allocate_matrix(k);
    double **M7 = allocate_matrix(k);
    
    double **T1 = allocate_matrix(k);
    double **T2 = allocate_matrix(k);
    
    // Fill submatrices
    #pragma omp parallel
{
    #pragma omp sections
    {
        #pragma omp section
        {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    A11[i][j] = A[i][j];
                    B11[i][j] = B[i][j];
                }
            }
        }

        #pragma omp section
        {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    A12[i][j] = A[i][j + k];
                    B12[i][j] = B[i][j + k];
                }
            }
        }

        #pragma omp section
        {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    A21[i][j] = A[i + k][j];
                    B21[i][j] = B[i + k][j];
                }
            }
        }

        #pragma omp section
        {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    A22[i][j] = A[i + k][j + k];
                    B22[i][j] = B[i + k][j + k];
                }
            }
        }
    }
}


    #pragma omp task shared(M1)
    {
        double **T1_local = allocate_matrix(k);
        double **T2_local = allocate_matrix(k);

        matrix_add(A11, A22, T1_local, k);  // T1 = A11 + A22
        matrix_add(B11, B22, T2_local, k);  // T2 = B11 + B22
        strassen_multiply_internal(T1_local, T2_local, M1, k);

        free_matrix(T1_local, k);
        free_matrix(T2_local, k);
    }

    #pragma omp task shared(M2)
    {
        double **T1_local = allocate_matrix(k);

        matrix_add(A21, A22, T1_local, k);  // T1 = A21 + A22
        strassen_multiply_internal(T1_local, B11, M2, k);

        free_matrix(T1_local, k);
    }

    #pragma omp task shared(M3)
    {
        double **T2_local = allocate_matrix(k);

        matrix_sub(B12, B22, T2_local, k);  // T2 = B12 - B22
        strassen_multiply_internal(A11, T2_local, M3, k);

        free_matrix(T2_local, k);
    }

    #pragma omp task shared(M4)
    {
        double **T2_local = allocate_matrix(k);

        matrix_sub(B21, B11, T2_local, k);  // T2 = B21 - B11
        strassen_multiply_internal(A22, T2_local, M4, k);

        free_matrix(T2_local, k);
    }

    #pragma omp task shared(M5)
    {
        double **T1_local = allocate_matrix(k);

        matrix_add(A11, A12, T1_local, k);  // T1 = A11 + A12
        strassen_multiply_internal(T1_local, B22, M5, k);

        free_matrix(T1_local, k);
    }

    #pragma omp task shared(M6)
    {
        double **T1_local = allocate_matrix(k);
        double **T2_local = allocate_matrix(k);

        matrix_sub(A21, A11, T1_local, k);  // T1 = A21 - A11
        matrix_add(B11, B12, T2_local, k);  // T2 = B11 + B12
        strassen_multiply_internal(T1_local, T2_local, M6, k);

        free_matrix(T1_local, k);
        free_matrix(T2_local, k);
    }

    #pragma omp task shared(M7)
    {
        double **T1_local = allocate_matrix(k);
        double **T2_local = allocate_matrix(k);

        matrix_sub(A12, A22, T1_local, k);  // T1 = A12 - A22
        matrix_add(B21, B22, T2_local, k);  // T2 = B21 + B22
        strassen_multiply_internal(T1_local, T2_local, M7, k);

        free_matrix(T1_local, k);
        free_matrix(T2_local, k);
    }

    // Wait for all tasks to complete
    #pragma omp taskwait

    matrix_add(M1, M4, T1, k);    // T1 = M1 + M4
    matrix_sub(T1, M5, T2, k);    // T2 = T1 - M5
    matrix_add(T2, M7, C11, k);   // C11 = T2 + M7
    
    matrix_add(M3, M5, C12, k);   // C12 = M3 + M5
    matrix_add(M2, M4, C21, k);   // C21 = M2 + M4
    
    matrix_sub(M1, M2, T1, k);    // T1 = M1 - M2
    matrix_add(T1, M3, T2, k);    // T2 = T1 + M3
    matrix_add(T2, M6, C22, k);   // C22 = T2 + M6
    
    // Copy sub-blocks into C
    for (int i = 0; i < k; i++){
        for (int j = 0; j < k; j++){
            C[i][j]         = C11[i][j];
            C[i][j + k]     = C12[i][j];
            C[i + k][j]     = C21[i][j];
            C[i + k][j + k] = C22[i][j];
        }
    }
    
    free_matrix(A11, k); free_matrix(A12, k);
    free_matrix(A21, k); free_matrix(A22, k);
    free_matrix(B11, k); free_matrix(B12, k);
    free_matrix(B21, k); free_matrix(B22, k);
    free_matrix(C11, k); free_matrix(C12, k);
    free_matrix(C21, k); free_matrix(C22, k);
    free_matrix(M1, k);  free_matrix(M2, k);
    free_matrix(M3, k);  free_matrix(M4, k);
    free_matrix(M5, k);  free_matrix(M6, k);
    free_matrix(M7, k);
    free_matrix(T1, k);  free_matrix(T2, k);
}

void strassen_multiply_omp(double **A, double **B, double **C, int n) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            strassen_multiply_internal(A, B, C, n);
        }
    }
}

/*-------------------------------------------------------------
 * Main
 *-------------------------------------------------------------*/
int main(void) {
    int n;
    printf("Enter matrix dimension (power of 2 recommended): ");
    scanf("%d", &n);

    double **A = allocate_matrix(n);
    double **B = allocate_matrix(n);
    double **C = allocate_matrix(n);

    // Randomly initialize A, B
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i][j] = rand() % 10;  
            B[i][j] = rand() % 10;  
        }
    }

    // Time the parallel Strassen
    double start = omp_get_wtime();
    strassen_multiply_omp(A, B, C, n);
    double end = omp_get_wtime();
    printf("Time for Parallel Strassen multiply (n=%d): %.6f s\n", n, end - start);

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}
