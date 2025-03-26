#include <vector>
#include <mutex>
#include <memory>
#include "glog/logging.h"
#include <thread>
#include <immintrin.h>
#include <utils/ring_buffer.hpp>


const size_t buffer_size = 1000;
std::unique_ptr<ringBuffer<int>> buff_ptr = std::make_unique<ringBuffer<int>>(buffer_size); 
bool shutdown = false;

void produce(ringBuffer<int>* buff_ptr){
    static int l = 0;
    static int u = 1;
    const static int buff_once  = 256;
    while (!shutdown){
        std::cout << "[Producer]: Filling global buffer ..." << std::endl;
        for (int i = l * buff_once; i < u * buff_once; i++){
            buff_ptr->insert(i);
        }

        l++;
        u++;

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}


void consume(ringBuffer<int>* buff_ptr){ 
    const static int buff_once  = 256;
    std::vector<int> disp_buff(buff_once);

    while (!shutdown){
        std::cout << "[Consumer]: Copying data form global buffer to display on console: " << std::endl;
        for (int i = 0; i < buff_once; i++){
            disp_buff[i] = (buff_ptr->pop());
        }

        // Print the data
        for (int i = 0; i < buff_once; i++){
            std::cout<< disp_buff[i] << ",";
        }
        std::cout<< std::endl;
        disp_buff.clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));  
    }  
}

void hold(){
    std::cout<< "[Wait]: Waiting for user input to terminate.." << std::endl;
    std::cin.get();
    shutdown = true;
}

int main(){

    std::thread producer_thread(produce,buff_ptr.get());
    std::thread consumer_thread(consume,buff_ptr.get());
    std::thread wait_thread(hold);


    while(!shutdown){

        std::cout<< "[Main]: " << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }



    producer_thread.join();
    consumer_thread.join();
    wait_thread.join();


}