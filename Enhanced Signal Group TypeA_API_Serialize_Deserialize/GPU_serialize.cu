#include "GPU_headers.h"

using namespace std;

__global__ void ComXf_Com_ComSignalGroupA_kernel(uint8* buffer, uint32* bufferLength, SignalGroup_A_Type* dataElement, int* offsets1) {

    uint8* buff = buffer;
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    SignalGroup_A_Type* dataElement_ptr = dataElement;

    if (x < NUM_BYTES) {
        uint32 idx = x;
        if (idx >= offsets1[0] && idx < (offsets1[1]/2)) {
            buff[idx * 2] = *(((uint8*)&dataElement_ptr->signal1) + ((idx * 2) - offsets1[0]));
            buff[idx * 2 + 1] = *(((uint8*)&dataElement_ptr->signal1) + ((idx * 2 + 1) - offsets1[0]));
        }
        if (idx >= (offsets1[2]/2) && idx < (offsets1[3]/2)) {
            buff[idx * 2] = *(((uint8*)&dataElement_ptr->signal2) + ((idx * 2) - offsets1[2]));
            buff[idx * 2 + 1] = *(((uint8*)&dataElement_ptr->signal2) + ((idx * 2 + 1) - offsets1[2]));
        }
        if (idx >= (offsets1[4] / 2) && idx < (offsets1[5] / 2)) {
            buff[idx * 2] = *(((uint8*)&dataElement_ptr->signal3) + ((idx * 2) - offsets1[4]));
            buff[idx * 2 + 1] = *(((uint8*)&dataElement_ptr->signal3) + ((idx * 2 + 1) - offsets1[4]));
        }
    }
}

void ComXf_Com_ComSignalGroupA(uint8* buffer, uint32* bufferLength, SignalGroup_A_Type dataElement) {

    uint32 block_size = 1024;
    uint32 grid_size = (NUM_BYTES + block_size - 1) / block_size;

    uint32* bufferlength = bufferLength;

    SignalGroup_A_Type* ptr_s = &dataElement;
    SignalGroup_A_Type* dev_ptr_s = 0;


    auto start_time_d1 = std::chrono::high_resolution_clock::now();
    // Allocate device memory
    cudaError_t cudaStatus = cudaMalloc((void**)&dev_ptr_s, sizeof(SignalGroup_A_Type));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed at device struct in Serialization: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Transfer data from host to device
    cudaStatus = cudaMemcpy(dev_ptr_s, ptr_s, sizeof(SignalGroup_A_Type), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed at device struct in Serialization: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_ptr_s);  // Free the allocated device memory
    }
    auto finish_time_d1 = std::chrono::high_resolution_clock::now();
    auto duration_ns_d1 = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time_d1 - start_time_d1);  //Time in NANOSEC
    double duration_s_d1 = duration_ns_d1.count();
    printf("\nTime of Host to Device of Struct In Micro Seconds: %f\n", duration_s_d1 * 1e-3);  //Time in MICROSEC


    //Start & End offsets of Signals 1 - 3
    const int offsets_size_k1 = 6;
    int offsets_k1[offsets_size_k1];
    int* d_offsets_k1;

    int start = 0, end = sizeof(dev_ptr_s->signal1);
    offsets_k1[0] = start;
    offsets_k1[1] = end;
    printf("\nSignal 1 End: %d\n", end);

    start = end;
    end = start + sizeof(dev_ptr_s->signal2);
    offsets_k1[2] = start;
    offsets_k1[3] = end;
    printf("\nSignal 2 Start: %d\n", start);
    printf("\nSignal 2 End: %d\n", end);

    start = end;
    end = start + sizeof(dev_ptr_s->signal3);
    offsets_k1[4] = start;
    offsets_k1[5] = end;
    printf("\nSignal 3 Start: %d\n", start);
    printf("\nSignal 3 End: %d\n", end);

    cudaMalloc((void**)&d_offsets_k1, offsets_size_k1 * sizeof(int));
    cudaMemcpy(d_offsets_k1, offsets_k1, offsets_size_k1 * sizeof(int), cudaMemcpyHostToDevice);



    //Serialization Sandwich
    //I'm here

    ComXf_Com_ComSignalGroupA_kernel << <grid_size, block_size >> > ((uint8*)buffer, bufferlength, dev_ptr_s, d_offsets_k1);

    cudaDeviceSynchronize();


    //Serialization Sandwich

    printf("\nGRID SIZE......... : %d \n", grid_size);
    printf("\nBLOCK SIZE........ : %d \n", block_size);
    //for (int i = 0; i < NUM_BYTES; i++) {
    //    printf("%x \n", buffer[i]);
    //    if ((i % 6000) == 0)
    //    {
    //        printf("\n ...........NEW SIGNAL.......... \n");
    //        printf("\n ...........NEW SIGNAL.......... \n");
    //        printf("\n ...........NEW SIGNAL.......... \n");
    //    }
    //    if ((i % 6000) == 0)
    //    {
    //        printf("Press Enter to continue...\n");
    //        getchar();  // Wait for user input before printing the next chunk
    //    }
    //}

    cudaFree(dev_ptr_s);
    cudaFree(d_offsets_k1);

}


