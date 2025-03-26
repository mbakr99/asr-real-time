#include "decoders/beam.hpp"
#include <vector>
#include <ctime>
#include <random>
#include <iostream>


int main(){

    std::cout << "Hi\n";
    // create a normal distributio for score 
    std::mt19937 generator(std::time(0));
    std::uniform_int_distribution<int> distribution(1,5);

    
    std::string dum_sequence = "I|love|eating|bannana|and|playing|football";
    std::cout << "num of chars is: " << dum_sequence.size() << std::endl;
    std::vector<int> score;
    for (char a : dum_sequence){
        score.push_back(distribution(generator));
    }


    // 
    beam::ctcBeam beamA;
    int num_steps = 30;
    if (num_steps > dum_sequence.size()) num_steps = dum_sequence.size();
    for (int i = 0; i < num_steps; ++i){
        beamA.update_beam(dum_sequence[i], score[i]);
        std::cout << "seuqence: " << beamA.get_sequence() << std::endl;
        std::cout << "last word:" << beamA.get_last_word() << std::endl; 
    }

   


}