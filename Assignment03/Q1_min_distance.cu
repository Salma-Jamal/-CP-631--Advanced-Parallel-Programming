#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>
#include <sys/time.h>


// kernel config cuda
#define BLOCK_SIZE 256


// Funtion to get the current time in seconds.
double current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


// Implementation of CPU Serial (Part 1)
// The method find the minimum Euclidean distance between 
// unique pairs of particles in 2D space using a serial implementation on cpu
float cpu_serial_min_distance(const float *x, const float *y, int n) {
    float dist_minimum = FLT_MAX;
    // we iterates through unique particle pairs , j > i only
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) { 
            // Euclidean distance part of formula (x2 - x1)
            float dx = x[i] - x[j];
            // Euclidean distance part of formula (y2 - y1)
            float dy = y[i] - y[j];
            // Final part of Euclidean distance sqrt(dx**2 + dy**2)
            float pair_dist = sqrtf(dx * dx + dy * dy);

            // Update the minimum distance
            if (pair_dist < dist_minimum) {
                dist_minimum = pair_dist;
            }
        }
    }
    return dist_minimum;
}


// Implementation of GPU One Thread per Particle (Part 2)
// In the funton each thread is assigned particle and computes the minimum distance.
// So each thread calculate the Euclidean distance between its particle and all other particle (not itself)
// And it stores the minimum Euclidean distance it finds in local array.
__global__
void min_distance_perParticle_gpu(const float *x, const float *y, int n, float *temp_local_min) {
    // compute thread global index 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

     // Make sure the thread is within bounds
    if (i < n) {
        float x_index = x[i];
        float y_index = y[i];

        // Initialization to largest possible float value
        float minimum_val = FLT_MAX;

        for (int j = 0; j < n; j++) {
            // Don't calculate the distance with itself
            if (j == i) continue;
            // calculate the euclidean distance.
            float d_x = x_index - x[j];
            float d_y = y_index - y[j];
            float local_dist = sqrtf(d_x * d_x + d_y * d_y);
            // update is smaller
            if (local_dist < minimum_val)
                minimum_val = local_dist;
        }
        temp_local_min[i] = minimum_val;
    }
}


// Implementation of GPU One Thread per Pair (Part 3)
// In this part in each thread it calculate the distance for one unique pair of particles.
// unique pairs total number ~= N*(N-1)/2.
__global__
void gpu_min_distance_per_pair(const float *x, const float *y, int n, float *pair_dist, long long total_Pairs) {
    // get the thread index (thread of unique pair)
    long long th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // break if index of thread > unique pairs number
    if (th_idx >= total_Pairs) return;


    // This mapping walks through a conceptual upper-triangular matrix

    // Map the index to a unique pair (i, j) such that i < j.
    // in the mapping it walks thruugh an upper-triangular matrix to invert the index.
    int rowi = 0;
    long long tempopary = th_idx;


    // Find row (i) by doing subtraction of lengths of each row in the uper triangle until tempopary fits in
    while (tempopary >= (n - 1 - rowi)) {
        tempopary -= (n - 1 - rowi);
        rowi++;
    }

    // now find j, which is the offset in row i 
    int j = rowi + 1 + tempopary;

    float d_x = x[rowi] - x[j];
    float d_y = y[rowi] - y[j];
    float temppdist = sqrtf(d_x * d_x + d_y * d_y);
    pair_dist[th_idx] = temppdist;
}


// this part is kernel to do binary reduction and compute the minimum value in the input array.
// will use it in part 2 (GPU) and part 3 (GPU) for reduction.
__global__
void reduce_min_kernel(float *g_data, int n) {
    // define a memory shared to store minimum distances values (intermediate) in a block.
    extern __shared__ float shared_data[];
    // the thread index
    unsigned int thread_id = threadIdx.x;
    // index for the first  two elements in the thread
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    // initiate to largest value
    float thread_Min = FLT_MAX;
     // put the first element into to thread min if in bounds.
    if (i < n) {
        thread_Min = g_data[i];
        // update thread_Min if smaller 
        if (i + blockDim.x < n) {
            float temp = g_data[i + blockDim.x];
            if (temp < thread_Min)
                thread_Min = temp;
        }
    }
    // save minimum value into shared memory to later reduction
    shared_data[thread_id] = thread_Min;

    // Synchronize the threads to make sure all at the same stage (all data is written)
    __syncthreads();

    // binary reduction on the shared memory
    for (unsigned int strid = blockDim.x / 2; strid > 0; strid >>= 1) {
        if (thread_id < strid) {
            float a = shared_data[thread_id];
            float b = shared_data[thread_id + strid];

            if (b < a)
            {
                shared_data[thread_id] = b;
            }
            else
            {
               shared_data[thread_id] = a; 
            }
           // shared_data[thread_id] = (b < a) ? b : a;
        }
        // Synchronize
        __syncthreads();
    }
    // save block result to global memory
    if (thread_id == 0) {
        g_data[blockIdx.x] = shared_data[0];
    }
}



int main(int argc, char **argv) {

    int N = 4096; 
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (N <= 1) {
        fprintf(stderr, "(N): Must be greater than 1.\n");
        return 1;
    }

    // Use a fixed seed 
    srand(0);

    // Memory allocation for particle coordinates (cpu)
    float *memory_x = (float *)malloc(N * sizeof(float));
    float *memory_y = (float *)malloc(N * sizeof(float));

    // Random Number generation in range [0, 1].
    for (int ii = 0; ii < N; ii++) {
        memory_x[ii] = (float)rand() / (float)RAND_MAX;
        memory_y[ii] = (float)rand() / (float)RAND_MAX;
    }

    
    // Serial computation (Part 1)
    double start_time_epu = current_time();
    float min_euc_dis_cpu = cpu_serial_min_distance(memory_x, memory_y, N);
    double end_time_cpu = current_time();
    printf("Part 1:\nMinimum Euclidean Distance = %e, Time = %f seconds\n\n", min_euc_dis_cpu, end_time_cpu - start_time_epu);

    
    // Copy to GPU
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, memory_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, memory_y, N * sizeof(float), cudaMemcpyHostToDevice);

   

    // GPU one thread per particle (Part 2)
    // memory allocation for local minimum distances.
    float *min_dist_local;
    cudaMalloc((void**)&min_dist_local, N * sizeof(float));

    int threadInBlock = BLOCK_SIZE;
    int PerGrid_blocks = (N + threadInBlock - 1) / threadInBlock;

    double start_time_gpu_p2 = current_time();
    min_distance_perParticle_gpu<<<PerGrid_blocks, threadInBlock>>>(d_x, d_y, N, min_dist_local);
    cudaDeviceSynchronize();

    // reduce the local min_dist to a global.
    int Size_Cur = N;
    while (Size_Cur > 1) {
        int thrds = BLOCK_SIZE;
        int blocks = (Size_Cur + thrds * 2 - 1) / (thrds * 2);
        reduce_min_kernel<<<blocks, thrds, thrds * sizeof(float)>>>(min_dist_local, Size_Cur);
        cudaDeviceSynchronize();
        Size_Cur = blocks;
    }
    float min_dist_gpu_p2;
    cudaMemcpy(&min_dist_gpu_p2, min_dist_local, sizeof(float), cudaMemcpyDeviceToHost);
    double end_time_gpu_p2 = current_time();

    printf("Part 2:\nMinimum Euclidean Distance = %e, Time = %f seconds\n\n", min_dist_gpu_p2, end_time_gpu_p2 - start_time_gpu_p2);

    // GPU with one thread per pair (Part 3)
    // unique pairs total
    long long total_Pairs = ((long long)N * (N - 1)) / 2;
    float *d_pair_dist;
    cudaMalloc((void**)&d_pair_dist, total_Pairs * sizeof(float));

    int threadInBlockPair = BLOCK_SIZE;
    int blocksPairPerGrid = (total_Pairs + threadInBlockPair - 1) / threadInBlockPair;
    double start_time_gpu2 = current_time();
    gpu_min_distance_per_pair<<<blocksPairPerGrid, threadInBlockPair>>>(d_x, d_y, N, d_pair_dist, total_Pairs);
    cudaDeviceSynchronize();

    // Reduce pair distances to get global minimum
    Size_Cur = total_Pairs;
    while (Size_Cur > 1) {
        int thrds = BLOCK_SIZE;
        int blocks = (Size_Cur + thrds * 2 - 1) / (thrds * 2);
        reduce_min_kernel<<<blocks, thrds, thrds * sizeof(float)>>>(d_pair_dist, Size_Cur);
        cudaDeviceSynchronize();
        Size_Cur = blocks;
    }
    float min_euc_dist_gpu2;
    cudaMemcpy(&min_euc_dist_gpu2, d_pair_dist, sizeof(float), cudaMemcpyDeviceToHost);
    double end_time_gpu2 = current_time();
    printf("Part 3:\nMinimum Euclidean Distance = %e, Time = %f seconds\n\n", min_euc_dist_gpu2, end_time_gpu2 - start_time_gpu2);

    // Clean 
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(min_dist_local);
    cudaFree(d_pair_dist);
    free(memory_x);
    free(memory_y);

    return 0;
}
