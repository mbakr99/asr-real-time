#include "decoders/beam.hpp"
#include <iostream>

int main(){

    // beam::ctcBeam test_beam;

    // std::string test_string = "||I|Love|Safiya|Al-Yaze-edi|Very|Much|";
    // std::string test_string_compl = "And|I|Plan|To|Mar-ry|Her";
    // std::string tmp = "";


    // beam::ctcBeam copy_beam;

    // for (const auto& letter : test_string){
    //     tmp += letter;
    //     std::cout << "current seq: " << tmp << std::endl;
    //     test_beam.update_beam(letter, 1);
    //     std::cout << "[main] last word is: " << test_beam.get_last_word() << std::endl;
    // }

    // copy_beam = test_beam;
    // tmp = test_string;
    // for (const auto& letter : test_string_compl){
    //     tmp += letter;
    //     std::cout << "current set: " << tmp << std::endl;
    //     copy_beam.update_beam(letter, 1);
    //     std::cout << "[main] last word is: " << copy_beam.get_last_word() << std::endl;
    // }<

    std::string str1 = "Hello";
    std::string str2 = "Hello";
    
    std::cout << "equal? " << (str1 == str2) << std::endl;


}

