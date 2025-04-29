#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <crypt.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);                          
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);            
    MPI_Comm_size(MPI_COMM_WORLD, &size);            

    char hash[256] = {0}; 

    // read hash and broadcast to processes
    if (rank == 0) { 
        FILE *fp = fopen("/home/sahmed147/Assigment1/password_hash.txt", "r");
        if (fp == NULL || fgets(hash, sizeof(hash), fp) == NULL) {
            perror("Failed while Reading File");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(fp);
        hash[strcspn(hash, "\n")] = '\0';
    }
    // brodcast hash to all processes
    MPI_Bcast(hash, sizeof(hash), MPI_CHAR, 0, MPI_COMM_WORLD);

    int num_pass_chars = 64;  // number of characters allowed in password
    int pass_len = 4;         // length of password
    char passwd[pass_len + 1];
    const char *const passchars =
      "./0123456789ABCDEFGHIJKLMNOPQRST"
      "UVWXYZabcdefghijklmnopqrstuvwxyz";
    
    long long int number_of_possible_passwords = 1;
    int j;
    char found_password[pass_len + 1];
    found_password[0] = '\0'; // initate it to empty, to use it as a flag to break from loop.
    long long int ilong, itest;
    char *result;
    int ok;

    double end_time,elapsed_time,start_time;

    int local_found = 0;
    int local_found_rank = size;  // to help choose the root to broadcast msg.

    for (j = 0; j < pass_len; j++) {
        number_of_possible_passwords *= num_pass_chars;
    }

    start_time = MPI_Wtime();
    // loop to search for password
    for (ilong = rank; ilong < number_of_possible_passwords; ilong += size) {
       // search for password
        itest = ilong;
        for (j = 0; j < pass_len; j++) {
            passwd[j] = passchars[itest % num_pass_chars];
            itest /= num_pass_chars;
        }
        // null terminate the password
        passwd[pass_len] = '\0';
        
        result = crypt(passwd, hash);
        ok = (strcmp(result, hash) == 0);

        if (ok) {
            strcpy(found_password, passwd);
            // set local_found to 1 to indicate that the password is found.
            local_found = 1;
            local_found_rank = rank;

            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;

            printf("'%s' is the password\n", found_password);
            printf("Found by process %d\n", rank);
            printf("Elapsed time: %f seconds\n", elapsed_time);

        }
        // brodcast the found password to all processes
        int global_found = 0;
        MPI_Allreduce(&local_found, &global_found, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (global_found) {
           
            // Maybe more than one process found the password and set local_found_rank to its rank. 
            // So, we take the lowest rank to be root of the brodcast msg. 
            int global_found_rank;
            MPI_Allreduce(&local_found_rank, &global_found_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Bcast(found_password, pass_len + 1, MPI_CHAR, global_found_rank, MPI_COMM_WORLD);
            break;
        }
    }
    
    MPI_Finalize();  
    return 0;
}

