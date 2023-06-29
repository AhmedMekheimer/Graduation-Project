#include "GPU_headers.h"
bool compareSignalGroups(const SignalGroup_A_Type* group1, const SignalGroup_A_Type* group2) {
    // Compare the values in signal1 array
    for (int i = 0; i < signal1_size; i++) {
        if (group1->signal1[i] != group2->signal1[i]) {
            return false;
        }
    }

    // Compare the values in signal2 array
    for (int i = 0; i < signal2_size; i++) {
        if (group1->signal2[i] != group2->signal2[i]) {
            return false;
        }
    }

    // Compare the values in signal3 array
    for (int i = 0; i < signal3_size; i++) {
        if (group1->signal3[i] != group2->signal3[i]) {
            return false;
        }
    }

    // All values are equal
    return true;
}

int main()
{
    //Struct is Allocated Using CUDAMemCpy
    SignalGroup_A_Type s;
    SignalGroup_A_Type d;
    SignalGroup_A_Type* d_ptr = &d;
    SignalGroup_A_Type* ptr_s = &s;


    for (int i = 0; i < signal1_size; i++) {
        ptr_s->signal1[i] = i % 11;
    }
    for (int i = 0; i < signal2_size; i++) {
        ptr_s->signal2[i] = i % 11;
    }
    for (int i = 0; i < signal3_size; i++) {
        ptr_s->signal3[i] = i % 11;
    }


    //Buffer is Allocated in Unified Memory

    uint32 buffl = buffer_length;
    uint32* buffll = &buffl;
    uint8_t* buffer = nullptr;

    cudaError_t cudaStatus;
    cudaFree(0);
    cudaStatus = cudaMallocManaged(&buffer, *(buffll));
    if (buffer == nullptr) {
        printf("Failed to allocate memory for buffer.\n");
    }
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Serailization API CALL

    auto start_time = std::chrono::high_resolution_clock::now();
    ComXf_Com_ComSignalGroupA(buffer, buffll, s);
    auto finish_time = std::chrono::high_resolution_clock::now();

    // Serailization API CALL

    //Time Calculation
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time);  //Time in NANOSEC
    double duration_s = duration_ns.count();
    printf("\nTime of Serailization In Micro Seconds: %f\n", duration_s * 1e-3);  //Time in MICROSEC

    //2nd Empty SignalGroup for Deserialization 
    for (int i = 0; i < signal1_size; i++) {
        d_ptr->signal1[i] = 0;
    }
    for (int i = 0; i < signal2_size; i++) {
        d_ptr->signal2[i] = 0;
    }
    for (int i = 0; i < signal3_size; i++) {
        d_ptr->signal3[i] = 0;
    }

    // Deserailization API CALL

    auto start_time_d = std::chrono::high_resolution_clock::now();
    ComXf_Inv_Com_ComSignalGroupA(buffer, buffl, d_ptr);
    auto finish_time_d = std::chrono::high_resolution_clock::now();

    // Deserailization API CALL

    //Time Calculation
    auto duration_ns_d = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time_d - start_time_d);  //Time in NANOSEC
    double duration_s_d = duration_ns_d.count();
    printf("\nTime of Deserailization In Micro Seconds: %f\n", duration_s_d * 1e-3);  //Time in MICROSEC


    //Checking if Deserialized Data in New Struct is same as Old Struct
    if (compareSignalGroups(&s, &d) == true)
        printf("\nDeserialization Successfull\n");
    else
        printf("\nDeserialization Failed\n");
    //Checking if Deserialized Data in New Struct is same as Old Struct



                                                    ///////////             STOP HERE           ///////////
    //printf("\n Deserialized struct:\n");
    //for (int i = 0; i < 163200; i++) {
    //    //printf("%x \n", *((uint8*)(d_ptr)+i));
    //    printf("%x \n", *((uint8*)(& (d_ptr->signal1))+i));
    //    if ((i + 1) % 8767 == 0)
    //    {
    //        printf("Press Enter to continue...\n");
    //        printf("Press Enter to continue...\n");
    //        printf("Press Enter to continue...\n");
    //        printf("Press Enter to continue...\n");
    //        printf("Press Enter to continue...\n");
    //        printf("Press Enter to continue...\n");
    //        getchar();  // Wait for user input before printing the next chunk
    //    }
    //}
    FILE* serialize;
    char serialize_filename[] = "Time of serialization.txt";
    char serialize_text[] = "Time of serialization In Micro Seconds: ";

    serialize = fopen(serialize_filename, "w");


    fprintf(serialize, "%s", serialize_text);
    fprintf(serialize, "%f", duration_s * 1e-3);


    FILE* deserialize;
    char deserialize_filename[] = "Time of deserialization.txt";
    char deserialize_text[] = "Time of deserialization In Micro Seconds: ";

    deserialize = fopen(deserialize_filename, "w");


    fprintf(deserialize, "%s", deserialize_text);
    fprintf(deserialize, "%f", duration_s_d * 1e-3);


    FILE* serialize_data;
    char serialize_data_filename[] = "Buffer after serialization";
    //fprintf(serialize_data, "%s", "Buffer data after serialization:\n");
    serialize_data = fopen(serialize_data_filename, "w");
    for (int i = 0; i < NUM_BYTES; i++) {

        fprintf(serialize_data, "%u", buffer[i]);
        fprintf(serialize_data, "%s", "\n");

    }



    FILE* deserialize_data;
    char deserialize_data_filename[] = "Struct after deserialization";
    deserialize_data = fopen(deserialize_data_filename, "w");
    for (int i = 0; i < signal1_size; i++) {

        fprintf(deserialize_data, "%u", *((uint8*)(&(d_ptr->signal1)) + i));
        fprintf(deserialize_data, "%s", "\n");

    }
    fprintf(deserialize_data, "%s", "End of signal 1--------------------------------------\n");
    for (int i = 0; i < signal2_size; i++) {

        fprintf(deserialize_data, "%u", *((uint8*)(&(d_ptr->signal2)) + i));
        fprintf(deserialize_data, "%s", "\n");

    }
    fprintf(deserialize_data, "%s", "End of signal 2--------------------------------------\n");
    for (int i = 0; i < signal3_size; i++) {

        fprintf(deserialize_data, "%u", *((uint8*)(&(d_ptr->signal3)) + i));
        fprintf(deserialize_data, "%s", "\n");

    }
    fprintf(deserialize_data, "%s", "End of signal 3--------------------------------------\n");


    cudaStatus = cudaFree(buffer);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    return(0);
}