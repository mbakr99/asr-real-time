#include<iostream>
#include<immintrin.h>
#include<cstdlib>
#include <chrono>

float* allocateAlignedMemory(size_t size, size_t alignment){

    void* ptr = nullptr;
    if (posix_memalign(&ptr,alignment,size*sizeof(float))!=0){
        throw std::bad_alloc();
    }
    return static_cast<float*>(ptr);
}



float* vectorMultiply(float* a, float* b, float* result, size_t size){
    constexpr size_t CHUNK_SIZE = 8; //only 8 floats can be moved at a time (265 bits) 
    //move the aligned vectors to the YYM register 
    size_t i = 0;
    for (; i+CHUNK_SIZE <= size; i+=CHUNK_SIZE){
        __m256 vecA = _mm256_load_ps(&a[i]);
        __m256 vecB = _mm256_load_ps(&b[i]);
        __m256 vecR = _mm256_mul_ps (vecA,vecB);
        _mm256_store_ps(&result[i],vecR);
    }

    for (; i < size; i++ ){
        result[i] = a[i]*b[i];
    }
    
    return result;
}


float* serialMultiply(float* a, float* b, float* result, size_t size){

    for (size_t i = 0; i < size; i++){
        result[i] = a[i] * b[i];
    }
    return result;
}


int main(){

    constexpr size_t num_elements = 100000;
    constexpr size_t ALIGN = 32;

    //Allocate some aligned memory 
    float* vector_a = allocateAlignedMemory(num_elements, ALIGN);
    float* vector_b = allocateAlignedMemory(num_elements, ALIGN);
    float* result = allocateAlignedMemory(num_elements, ALIGN);

    //Initialize the vectors with some values 
    for (int i = 0; i<num_elements; i++){
        vector_a[i] = i;
        vector_b[i] = i*3;
    }


    //Setting timers to caculate computational cost
    std::chrono::system_clock clock;
    std::chrono::time_point<std::chrono::system_clock> t_i,t_f;
    
    //Perform multiplication
    t_i = clock.now(); //Start timer
    result = vectorMultiply(vector_a, vector_b, result, num_elements);
    t_f = clock.now();
    std::cout << "Duration of vectorized operation: " <<
                std::chrono::duration_cast<std::chrono::nanoseconds>(t_f-t_i).count() <<
                std::endl;
    
    //Perform serial multiplication
    t_i = clock.now();
    result = serialMultiply(vector_a,vector_a,result,num_elements);
    t_f = clock.now();
    std::cout << "Duration of serial operation: " <<
                std::chrono::duration_cast<std::chrono::nanoseconds>(t_f-t_i).count() <<
                std::endl;


    free(vector_a);
    free(vector_b);
    free(result);
}