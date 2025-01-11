#include<iostream>
#include<immintrin.h>
#include<stdlib.h>


int* allocAligned(size_t num_elements, size_t allignment){

    void* ptr = nullptr;
    ptr = std::aligned_alloc(allignment, num_elements*sizeof(int));
    if (!ptr)
        throw std::bad_alloc();
    
    return static_cast<int*>(ptr);
}


void vectorSum(int* a, int* b, int* result, size_t num_elements){
    constexpr size_t CHUNK_SIZE = 256/sizeof(int);
    size_t i = 0;
    for (; i + CHUNK_SIZE <= num_elements; i+=CHUNK_SIZE){
        __m256i vec_A = _mm256_load_si256(reinterpret_cast<__m256i*>(&a[i]));
        __m256i vec_B = _mm256_load_si256(reinterpret_cast<__m256i*>(&b[i]));
        __m256i vec_Result = _mm256_add_epi32 (vec_A, vec_B);
        _mm256_store_si256 (reinterpret_cast<__m256i*>(&result[i]),vec_Result);
    }

    for (; i < num_elements; i++)
        result[i] = a[i] + b[i];

}


int main(){
    
    //Define the vectors 
    size_t num_elements = 577;
    size_t ALLIGN = 32;
    int* vec_A = allocAligned(num_elements, ALLIGN);
    int* vec_B = allocAligned(num_elements, ALLIGN);
    int* result = nullptr;

    //Fill vectors 
    for (int i = 0; i < num_elements; i++){
        vec_A[i] = 0;
        vec_B[i] = i;
    }
    
    //Perform Addition 
    vectorSum(vec_A, vec_B, result, num_elements);

    for (int i = 0; i < 10; i++)
        std::cout<< result[i] << std::endl;   




}