#include "stdio.h"
#include "omp.h"

#define MAT_DIM 10 // Matrix dimension
#define sizeOfBlock 100 // Block size for block based function

int N = MAT_DIM;
int A[MAT_DIM][MAT_DIM];
int Asum[MAT_DIM][MAT_DIM];       // To stor output of Serial function
int Asumtask[MAT_DIM][MAT_DIM];   // To store output of OpenMP tasks function
int Asumtaskblock[MAT_DIM][MAT_DIM]; //  To store output of block based function

// Srial 
int compute_serial() {
    int i, j;

    // Compute corner
    Asum[0][0] = A[0][0];

    // Compute top row
    i = 0;
    for (j = 1; j < N; j++) {
        Asum[i][j] = A[i][j] + Asum[i][j - 1];
    }

    // Compute left column
    j = 0;
    for (i = 1; i < N; i++) {
        Asum[i][j] = A[i][j] + Asum[i - 1][j];
    }

    // Compute interior
    for (i = 1; i < N; i++) {
        for (j = 1; j < N; j++) {
            Asum[i][j] = A[i][j] + Asum[i - 1][j] + Asum[i][j - 1] - Asum[i - 1][j - 1];
        }
    }

    return 0;
}

// Queston 2 
// use OpenMP task and dependencies to make sure the correct execution order.
int compute_tasks() {
    #pragma omp parallel // creates team of threds
    {
        #pragma omp single // only one thread executes the block (generates tasks)
        {
            // iterate over matrix elements
            int row, col; 
            for (row = 0; row < N; row++) {
                for (col = 0; col < N; col++) {
                    // top-left corner (no dependencies), produce Asum[0][0]
                    if (row == 0 && col == 0) {
                    #pragma omp task firstprivate(row, col) depend(out: Asumtask[row][col])
                    {
                        Asumtask[row][col] = A[row][col];
                    }
                    } else if (row == 0) { // top row only left dependency, 
                    #pragma omp task firstprivate(row, col) depend(in: Asumtask[row][col - 1]) depend(out: Asumtask[row][col])
                    {
                        // Sum the current A with thhe left neighbor's Asum
                        Asumtask[row][col] = A[row][col] + Asumtask[row][col - 1];
                    }
                    // left column top dependency
                    } else if (col == 0) {
                    #pragma omp task firstprivate(row, col) depend(in: Asumtask[row - 1][col]) depend(out: Asumtask[row][col])
                    {
                        Asumtask[row][col] = A[row][col] + Asumtask[row - 1][col];
                    }
                    // the interior elements (all dependencies)
                    } else {
                    #pragma omp task firstprivate(row, col) depend(in: Asumtask[row - 1][col], Asumtask[row][col - 1], Asumtask[row - 1][col - 1]) depend(out: Asumtask[row][col])
                    {
                        Asumtask[row][col] = A[row][col] + Asumtask[row - 1][col] + Asumtask[row][col - 1] - Asumtask[row - 1][col - 1];
                    }
                    }} }
        }
    }
    return 0;}

// Question 3

// a help method to compute block of the matrix, compute a block at positon (bi, bj)
void compute_block(int bi_index, int bj_index, int block_S, int N) {
    int start_pos = bi_index * block_S;
    int end_pos;
        if (start_pos + block_S < N) {
            end_pos = start_pos + block_S;
        } else {
            end_pos = N;
        }
    int start_pos_at_j = bj_index * block_S;
    int end_pos_at_j;      
            if (start_pos_at_j + block_S < N) {
            end_pos_at_j = start_pos_at_j + block_S;
        } else {
            end_pos_at_j = N;
        }

    // Top left of block cornr
    if (bi_index == 0 && bj_index == 0) {
        Asumtaskblock[0][0] = A[0][0];
    }

    // the top row 
    if (bi_index == 0) {
        int j_index_start = (bj_index == 0 ? 1 : start_pos_at_j);
        for (int j = j_index_start; j < end_pos_at_j; j++) {
            Asumtaskblock[0][j] = A[0][j] + Asumtaskblock[0][j - 1];
        }
    }

    // the left col of block
    if (bj_index == 0) {
        int i_index_start = (bi_index == 0 ? 1 : start_pos);
        for (int i = i_index_start; i < end_pos; i++) {
            Asumtaskblock[i][0] = A[i][0] + Asumtaskblock[i - 1][0];
        }
    }

    // the Interior (inside the block)
    int i_index_start = (bi_index == 0 ? 1 : start_pos);
    int j_index_start = (bj_index == 0 ? 1 : start_pos_at_j);
    for (int i = i_index_start; i < end_pos; i++) {
        for (int j = j_index_start; j < end_pos_at_j; j++) {
            Asumtaskblock[i][j] = A[i][j] + Asumtaskblock[i - 1][j] +
                                  Asumtaskblock[i][j - 1] - Asumtaskblock[i - 1][j - 1];}  }
}


void compute_tasks_blocks() {
    int N = MAT_DIM;
    int S = sizeOfBlock;
    int B = (N + S - 1) / S;
    int dummy[B * B];

    #pragma omp parallel
    #pragma omp single
    {
        for (int bi = 0; bi < B; bi++) {
            for (int bj = 0; bj < B; bj++) {
                if (bi == 0 && bj == 0) {
                    #pragma omp task depend(out: dummy[0])
                    {
                        compute_block(0, 0, S, N);
                        dummy[0] = 1;
                    }
                }
                else if (bi == 0) {
                    #pragma omp task depend(in: dummy[0 * B + (bj - 1)]) depend(out: dummy[0 * B + bj])
                    {
                        compute_block(0, bj, S, N);
                        dummy[0 * B + bj] = 1;
                    }
                }
                else if (bj == 0) {
                    #pragma omp task depend(in: dummy[(bi - 1) * B + 0]) depend(out: dummy[bi * B + 0])
                    {
                        compute_block(bi, 0, S, N);
                        dummy[bi * B + 0] = 1;
                    }
                }
                else {
                    #pragma omp task depend(in: dummy[(bi - 1) * B + bj], dummy[bi * B + (bj - 1)], dummy[(bi - 1) * B + (bj - 1)]) depend(out: dummy[bi * B + bj])
                    {
                        compute_block(bi, bj, S, N);
                        dummy[bi * B + bj] = 1;
                    }
                }
            }
        }
    }
}

// Question 4
// Compare the matrices to check for equality, return 1 if equal
int check_matrices_equality(int Mat_1[MAT_DIM][MAT_DIM], int Mat_2[MAT_DIM][MAT_DIM]) {
    int equal = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (Mat_1[i][j] != Mat_2[i][j]) 
                {
                    equal = 0;
                    printf("Found a mismatch at position (%d,%d) of matrices: value in Mat_1=%d, value in Mat_2=%d\n", i, j, Mat_1[i][j], Mat_2[i][j]);               
                    break;
                }
        }
        if(!equal) return 0;
    }
    return 1;
}

int main() {
    int i, j;
    double start, end, start_s, end_s, start_ts, end_ts, start_bl, end_bl;

    // arrays Initialization
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = i * N + j;
            Asum[i][j] = -99999;
            Asumtask[i][j] = -99999;
            Asumtaskblock[i][j] = -99999;
        }
    }

    // run Serial function and time it
    start_s = omp_get_wtime();
    compute_serial();
    end_s = omp_get_wtime();
    

    // run the fuction with Tasks openmp and time it
    start_ts = omp_get_wtime();
    compute_tasks();
    end_ts = omp_get_wtime();
   

    // run the fuction with blocks and time it
    start_bl = omp_get_wtime();
    compute_tasks_blocks();
    end_bl = omp_get_wtime();
    
    // print the time taken for each function
    printf("Time Taken Serial in seconds: %f\n", end_s - start_s);
    printf("Time Taken Tasks in seconds: %f\n", end_ts - start_ts);
    printf("Time Taken Blocked in seconds: %f\n", end_bl - start_bl);


    // Check for equality of the matrices using the function above
    if (check_matrices_equality(Asum, Asumtask) && check_matrices_equality(Asum, Asumtaskblock)) {
        printf("The matrices match.\n");
    } else {
        printf("The matrices do not match.\n");
    }

    return 0;
}

