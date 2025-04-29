#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void write_matrix_to_file(int n, double *local_matrix, MPI_Comm comm);
void read_matrix_from_file(int n, double *local_matrix, MPI_Comm comm);


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    // check if the number of arguments is correct
    if (argc != 2) {  
        return 1;
    }
    // convert the first argument to an integer
    int n = atoi(argv[1]); 
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if(n % p != 0) 
    {
        if(rank == 0)
            // error message when wrong number of processes
            fprintf(stderr, "n (%d) must be divisible by # processes %d\n", n, p);
             MPI_Abort(MPI_COMM_WORLD, 1);
    }
    //  number of rows per process
    int n_p = n / p;  

     // allocate memory for the matrix
    double *process_matrix = malloc(n * n_p * sizeof(double));
    if (!process_matrix) {
        fprintf(stderr, "Error: malloc failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    double start_time = MPI_Wtime();

    /// First Read From file matrix.txt
    if(rank == 0)
        printf("Reading matrix.txt\n");
        read_matrix_from_file(n, process_matrix, MPI_COMM_WORLD);
    
    /// break between them to finish
    MPI_Barrier(MPI_COMM_WORLD);

    /// Write matrix to file matrix2.txt
    if(rank == 0)
        printf("Writing the matrix to matrix2.txt\n");
        write_matrix_to_file(n, process_matrix, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);  
    double end_time = MPI_Wtime();

    double runtime = end_time - start_time;


    double max_runtime;
    MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Runtime: %f seconds\n", max_runtime);
    }


    free(process_matrix);
    MPI_Finalize();
    return 0;
}


void write_matrix_to_file(int n, double *local_matrix, MPI_Comm comm) {
    int rank, p, n_p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    n_p = n / p; 
    // create a new MPI datatype
    MPI_Datatype send_matrix_part;
    MPI_Type_vector(n_p, n_p,n,MPI_DOUBLE,&send_matrix_part);
    MPI_Type_commit(&send_matrix_part);

    if (rank == 0) {
        FILE *write_file = fopen("matrix2.txt", "w");

        // memory for n_p 
        double *buffer_mem = malloc(n_p * n * sizeof(double));
        if (!buffer_mem) {
            fprintf(stderr, "Error: malloc failed.\n");
            MPI_Abort(comm, 1);
        }

       
        for (int part = 0; part < p; part++) {
            
            // write process 0 part of the matrix
            for (int i = 0; i < n_p; i++) {   
                for (int j = 0; j < n_p; j++) { 
                    int global_row = part * n_p + i;
                    buffer_mem[i * n + j] = local_matrix[j * n + global_row];
                }
            }
            // write the rest of the processes
            for (int pc = 1; pc < p; pc++) {
                double *temp_mem = malloc(n_p * n_p * sizeof(double));

                MPI_Recv(temp_mem, n_p * n_p, MPI_DOUBLE, pc, 0, comm, MPI_STATUS_IGNORE);
    
                for (int j = 0; j < n_p; j++) {
                    for (int i = 0; i < n_p; i++) {
                        int global_col =  pc * n_p + j;
                        buffer_mem[i * n + global_col] = temp_mem[i * n_p + j]; 
                    }
                }
                free(temp_mem);
            }
            
            // write in file
            for (int i = 0; i < n_p; i++) {
                for (int j = 0; j < n; j++) {
                    fprintf(write_file, "%.2f ", buffer_mem[i * n + j]);
                }
                fprintf(write_file, "\n");
            }
        }
        free(buffer_mem);
        fclose(write_file);
    } else {

        for (int block = 0; block < p; block++) {
            MPI_Send(local_matrix + block * n_p, 1, send_matrix_part, 0, 0, comm);
        }
    }
    MPI_Type_free(&send_matrix_part);
}



void read_matrix_from_file(int n, double *local_matrix, MPI_Comm comm) {
    int rank, p,n_p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    n_p = n / p;
 
    // create a new MPI datatype
    MPI_Datatype matrix_part_type;
    MPI_Type_vector(n_p,n_p,n, MPI_DOUBLE, &matrix_part_type);
    MPI_Type_commit(&matrix_part_type);

    if (rank == 0) {
        FILE *read_file = fopen("matrix.txt", "r");
        // memory 
        double *row_block = malloc(n_p * n * sizeof(double));


        // read the matrix from the file
        for (int block = 0; block < p; block++) {
          
            for (int i = 0; i < n_p * n; i++) {
                if (fscanf(read_file, "%lf", &row_block[i]) != 1) {
                    fprintf(stderr, "Error");
                    MPI_Abort(comm, 1);
                }
            }
            // process the matrix data
            for (int proc = 0; proc < p; proc++) {
                if (proc == 0) {
                    // process zero copy
                    for (int i = 0; i < n_p; i++) {
                        for (int j = 0; j < n_p; j++) {
                            local_matrix[j * n + block * n_p + i] =
                                row_block[i * n + proc * n_p + j];
                        }
                    }
                } else {
                    // send for other processess
                    MPI_Send(row_block + proc * n_p, 1, matrix_part_type, proc, 0, comm);
                }
            }
        }
        free(row_block);
        fclose(read_file);
    } 
    else {
        // receive the matrix data
        MPI_Datatype recv_type;
        MPI_Type_vector(n_p, n_p, n, MPI_DOUBLE, &recv_type);
        MPI_Type_commit(&recv_type);
        for (int part = 0; part < p; part++) {
            MPI_Recv(local_matrix + part * n_p, 1, recv_type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
        MPI_Type_free(&recv_type);
    }
    MPI_Type_free(&matrix_part_type);
}

