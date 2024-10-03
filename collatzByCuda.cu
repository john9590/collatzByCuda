#include <stdio.h>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

long long N = 1 << 30;
long long threads = 256;
long long blocks = (N + threads * 300 - 1) / (threads * 300);

__global__ void collatz(bool* x, long long N) {
    long long tid = blockDim.x * blockIdx.x + threadIdx.x;
    tid *= 300;
    if (tid + 299 < N && !x[tid+299]) {
        if (x[tid]) {
            bool b = x[tid];
            bool c = true;
            for (int k = tid + 1; k < tid + 300; k++) {
                bool a = x[k];
                x[k] = a ^ b ^ c;
                c = ((a ^ b) & c) | (a & b);
                b = a;
            }
            x[tid] = false;
        }
        else {
            for (int k = tid + 1; k < tid + 300; k++) {
                x[k - 1] = x[k];
            }
            x[tid + 299] = false;
        }
    }
}

void prepare_input_data(bool A[], int n) {
    std::default_random_engine gen(20240312);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for (int k = 0; k < n; k++) {
        A[k] = (k % 300 <= 111) ? (fran(gen) >= 0.5) : false;
    }
}

int main(int argc, char* argv[]) {
    fprintf(stdout, "blocks : %lld\nthreads : %lld\n\n", blocks, threads);
    bool* h_A = new bool[N];
    prepare_input_data(h_A, N);
    bool* d_A;
    cudaMalloc((void**)&d_A, N * sizeof(bool));
    cudaMemcpy(d_A, h_A, N * sizeof(bool), cudaMemcpyHostToDevice);

    for (int i = 0; i < 2000; i++) {
        collatz << <blocks, threads >> > (d_A, N);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(h_A, d_A, N * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N / 300; i++) {
        for (int k = i * 300 + 299; k >= i * 300 + 64; k--) {
            if (h_A[k]) {
                for (int l = i * 300 + 299; l >= i * 300; l--) {
                    printf("%d", h_A[l] ? 1 : 0);
                }
                printf("\n");
                break;
            }
        }
    }
    cudaFree(d_A);
    delete[] h_A;
    return 0;
}
