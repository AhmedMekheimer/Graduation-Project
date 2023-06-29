#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "chrono"
#include <stdio.h>
#include "stdint.h"
#include <stdlib.h>
#include <time.h>
#include "Platform_Types.h"
#include "stdio.h"
#include "stdlib.h"
#include "iostream"
#include <cuda_runtime.h>

#define um
#define NUM_BYTES 1500000
#define E_OK 0
#define signal1_size 500000
#define signal2_size 500000
#define signal3_size 500000
#define buffer_length NUM_BYTES


typedef struct {
    uint8 signal1[signal1_size];
    uint8 signal2[signal2_size];
    uint8 signal3[signal3_size];

}SignalGroup_A_Type;

__global__ void ComXf_Com_ComSignalGroupA_kernel(uint8* buffer, uint32* bufferLength, SignalGroup_A_Type* dataElement, int* offsets1);
void ComXf_Com_ComSignalGroupA(uint8* buffer, uint32* bufferLength, SignalGroup_A_Type dataElement);

__global__ void ComXf_Inv_Com_ComSignalGroupA_kernel(uint8* buffer, uint32 bufferLength, SignalGroup_A_Type* dataElement, int* offsets1);
uint8 ComXf_Inv_Com_ComSignalGroupA(uint8* buffer, uint32 bufferLength, SignalGroup_A_Type* dataElement);


