#include <iostream>
#include <vector>
#include <torch/script.h>
#include <string>
#include <fstream>
#include <greedy_decoder.hpp>



int main(int argc, char* argv[]){

    if (argc < 2){
        std::cout<< "[Parser/Error]: Must pass a path to the token file " << std::endl;
    }
    torch::Tensor emissions = torch::rand({100000, 29});

 



    greedyDecoder decoder;
    bool flag = decoder.init_vocab(argv[1]);

    if (flag){
        std::cout << "[main]: init_vocab sucess" << decoder.tokens << std::endl; 
    }



    std::vector<char> result = decoder.decode_chars(emissions);
    std::string output(result.begin(),result.end());
    std::cout<< output << std::endl;;

}




