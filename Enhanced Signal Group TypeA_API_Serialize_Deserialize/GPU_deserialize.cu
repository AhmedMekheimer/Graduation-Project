#include "GPU_headers.h"

using namespace std;

__global__ void ComXf_Inv_Com_ComSignalGroupA_kernel(uint8* buffer, uint32 bufferLength, SignalGroup_A_Type* dataElement, int* offsets1) {

    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint8* Buffer_ptr = buffer;
    SignalGroup_A_Type* Data_ptr = dataElement;

    if (x < NUM_BYTES) {
        uint32 idx = x;

        if (idx >= offsets1[0] && idx < offsets1[1]) {
            *((uint8*)(&(Data_ptr->signal1)) + (idx * 2 - offsets1[0])) = *(Buffer_ptr + idx * 2);
            *((uint8*)(&(Data_ptr->signal1)) + (idx * 2 + 1  - offsets1[0])) = *(Buffer_ptr + idx * 2 + 1);
        }
        if (idx >= offsets1[2]/2 && idx < offsets1[3]/2) {
            *((uint8*)(&(Data_ptr->signal2)) + (idx * 2 - offsets1[2])) = *(Buffer_ptr + idx * 2);
            *((uint8*)(&(Data_ptr->signal2)) + (idx * 2 + 1 - offsets1[2])) = *(Buffer_ptr + idx * 2 + 1);
        }
        if (idx >= offsets1[4]/2 && idx < offsets1[5]/2) {
            *((uint8*)(&(Data_ptr->signal3)) + (idx * 2  - offsets1[4])) = *(Buffer_ptr + idx * 2);
            *((uint8*)(&(Data_ptr->signal3)) + (idx * 2 + 1 - offsets1[4])) = *(Buffer_ptr + idx * 2 + 1);
        }
    }
}

uint8 ComXf_Inv_Com_ComSignalGroupA(uint8* buffer, uint32 bufferLength, SignalGroup_A_Type* dataElement) {

    uint32 block_size = 1024;
    uint32 grid_size = (NUM_BYTES + block_size - 1) / block_size;
    uint32 bufferlength = bufferLength;

    SignalGroup_A_Type* ptr_s = dataElement;
    SignalGroup_A_Type* dev_ptr_s = 0;


    cudaError_t cudaStatus = cudaMalloc((void**)&dev_ptr_s, sizeof(SignalGroup_A_Type));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed at device struct in Deserialization: %s\n", cudaGetErrorString(cudaStatus));
    }

    const int offsets_size_k1 = 6;
    int offsets_k1[offsets_size_k1];
    int* d_offsets_k1;

    int start = 0, end = sizeof(dev_ptr_s->signal1);
    offsets_k1[0] = start;
    offsets_k1[1] = end;

    start = end;
    end = start + sizeof(dev_ptr_s->signal2);
    offsets_k1[2] = start;
    offsets_k1[3] = end;

    start = end;
    end = start + sizeof(dev_ptr_s->signal3);
    offsets_k1[4] = start;
    offsets_k1[5] = end;


    cudaMalloc((void**)&d_offsets_k1, offsets_size_k1 * sizeof(int));
    cudaMemcpy(d_offsets_k1, offsets_k1, offsets_size_k1 * sizeof(int), cudaMemcpyHostToDevice);


    //Deserialization Sandwich
    //I'm here

    ComXf_Inv_Com_ComSignalGroupA_kernel << <grid_size, block_size >> > ((uint8*)buffer, bufferlength, dev_ptr_s, d_offsets_k1);

    cudaDeviceSynchronize();

    //Deserialization Sandwich


    // Transfer data from device to host
    auto start_time_d4 = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMemcpy(dataElement, dev_ptr_s, bufferlength, cudaMemcpyDeviceToHost);;
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed at device to host in Deserialization: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_ptr_s);  // Free the allocated device memory
    }
    auto finish_time_d4 = std::chrono::high_resolution_clock::now();
    auto duration_ns_d4 = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time_d4 - start_time_d4);  //Time in NANOSEC
    double duration_s_d4 = duration_ns_d4.count();
    printf("\nTime of device to host of struct in Deserialization In Micro Seconds: %f\n", duration_s_d4 * 1e-3);  //Time in MICROSEC

    cudaFree(dev_ptr_s);
    cudaFree(d_offsets_k1);

    return E_OK;
}
