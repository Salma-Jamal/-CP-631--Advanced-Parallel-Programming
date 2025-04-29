#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int random_matrix_generator(int N) {
    double matrix[N][N];
    int i, j;

    srand((unsigned)time(NULL));

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX * 10.0; 
        }
    }

    FILE *fp = fopen("matrix.txt", "w");
    if (fp == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            fprintf(fp, "%.2f ", matrix[i][j]); 
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    printf("Matrix (%dx%d) successfully written to matrix.txt\n", N, N);
    return 0;
}


int main(int argc, char *argv[]) {
    if (argc != 2) {  
        return 1;
    }

    int N = atoi(argv[1]); 
    if (N <= 0) {
        printf("Matrix size should be a positive integer.\n");
        return 1;
    }

    return random_matrix_generator(N);
}