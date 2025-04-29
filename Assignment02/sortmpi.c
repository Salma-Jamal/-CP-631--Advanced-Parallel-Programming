#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define N 5


//determine number of items to send to each process
int number_communicated_values(double* arr, int low, int high, double target) { 
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] < target) low = mid + 1;
        else high = mid;
    }
    return low;
}
//use qsort to get numbers in ascending order.
int values_comparison(const void* a, const void* b) {
    double first_value = *(const double*)a;
    double second_value = *(const double*)b;
    return (first_value > second_value) - (first_value < second_value);
}

//exchange the counts of the numbers to be sent and received
void find_counts_exchange(int* send_counts, int* recv_counts, int p) {
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
}

//exchange the data between processes
void exchange_data_between_processes(double* send_buffer, int* items_counts_send, int* send_displac,
                   double* recv_buffer, int* items_counts_recv, int* revcv_displac, int p) {
    MPI_Alltoallv(send_buffer, items_counts_send, send_displac, MPI_DOUBLE,
                  recv_buffer, items_counts_recv, revcv_displac, MPI_DOUBLE, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double p = size;

   
    MPI_Barrier(MPI_COMM_WORLD); // synchronize processes
    double start_time = MPI_Wtime();

    //get random numbers n between 0 and 1
    srand(time(NULL) + rank); //use time and rank as seed
    double* numbers = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        numbers[i] = (double)rand() / RAND_MAX;
    }

    qsort(numbers, N, sizeof(double), values_comparison); //sort the numbers

    //split points computation
    int* splits = (int*)malloc((size - 1) * sizeof(int));
    for (int k = 1; k < size; k++) {
        double boundary = (double)k / p;
        splits[k - 1] = number_communicated_values(numbers, 0, N, boundary);
    }

    //compute send counts
    int* send_counts = (int*)malloc(size * sizeof(int));
    if (size > 1) {
        send_counts[0] = splits[0];
        for (int k = 1; k < size - 1; k++) {
            send_counts[k] = splits[k] - splits[k - 1];
        }
        send_counts[size - 1] = N - splits[size - 2];
    } else {
        send_counts[0] = N; //single process keeps all numbers
    }

    //exchange send counts
    int* recv_counts = (int*)malloc(size * sizeof(int));
    find_counts_exchange(send_counts, recv_counts, size);

    //compute total number of elements to receive
    int total_recv = 0;
    for (int j = 0; j < size; j++) {
        total_recv += recv_counts[j];
    }

    //init receive buffer
    double* recv_buffer = (double*)malloc(total_recv * sizeof(double));

    //calcuate send displacements
    int* sdispls = (int*)malloc(size * sizeof(int));
    if (size > 1) {
        sdispls[0] = 0;
        for (int k = 1; k < size; k++) {
            sdispls[k] = splits[k - 1];
        }
    } else {
        sdispls[0] = 0;
    }

    //calcuate receive displacements
    int* rdispls = (int*)malloc(size * sizeof(int));
    rdispls[0] = 0;
    for (int j = 1; j < size; j++) {
        rdispls[j] = rdispls[j - 1] + recv_counts[j - 1];
    }

    //exchange the data
    exchange_data_between_processes(numbers, send_counts, sdispls, recv_buffer, recv_counts, rdispls, size);

    //sort received data
    qsort(recv_buffer, total_recv, sizeof(double), values_comparison);


        
    MPI_Barrier(MPI_COMM_WORLD); // synchronize before stopping the timer
    double end_time = MPI_Wtime();

    printf("process %d has %d numbers: ", rank, total_recv); //get sample output 
    for (int i = 0; i < total_recv; i++) {
        printf("%.2f ", recv_buffer[i]);
    }
    printf("\n");

    // Print  time on process 0
    if (rank == 0) {
        printf("Time Taken: %f seconds\n\n", end_time - start_time);
    }    
    //remove the variables and finish the mpi 
    free(numbers);

    free(splits);

    free(send_counts);

    free(recv_counts);

    free(sdispls);
    free(rdispls);

    free(recv_buffer);

    MPI_Finalize();
    return 0;
}