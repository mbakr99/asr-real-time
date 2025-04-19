#include "utils/my_utils.hpp"
#include "decoders/lexicon_fst.hpp"
#include <vector>
#include <iostream>
#include "decoders/beam.hpp"


// using namespace asr;

template <typename T>
void fill_ones(std::vector<T>& vec, int size){
    for (int i = 0; i < size; ++i) vec.push_back(static_cast<T>(1));
}

fst::SymbolTable beam::ctcBeam::input_symbol_table_;
std::unique_ptr<beam::FSTDICT>  beam::ctcBeam::dictionary_ptr_;
std::unique_ptr<beam::FSTMATCH> beam::ctcBeam::matcher_ptr_;

int main(int argc, char** argv){
  
    if (argc < 2){ 
        std::cerr << "pass the verbosity level" << std::endl;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    fs::path log_dir = "/workspace/asr-real-time/logs";
    FLAGS_log_dir = log_dir;
    FLAGS_v = std::stoi(argv[1]);

    // set fst 
    fs::path fst_path = "/workspace/asr-real-time/data/lexicon/lexicon_fst.fst"; 
    // load an fst 
    LexiconFst lex_fst;
    if (!lex_fst.load_fst(fst_path)){
        std::cerr << "[main]: failed to load the fst" << std::endl;
        return -1;
    }

    auto dict_fst  = lex_fst.get_lexicon_fst();
    std::cout << "ptr: "<< dict_fst << std::endl;
    auto symbol_table = lex_fst.get_input_symbol_table();
    beam::ctcBeam::set_fst(dict_fst, symbol_table); // symbol_table 



    // print symbol table
    for (auto it = symbol_table.begin(); it != symbol_table.end(); ++it){
        std::cout << it->Symbol() << " : " << it->Label() << std::endl;
    }


    // set dum data 
    std::string sequence = "i|play|footbal-l|";
    std::vector<float> prob;
    fill_ones(prob, sequence.size());


    // set beam
    std::string init_seq = "";
    float inti_score = 1.0;
    beam::ctcBeam t_beam{init_seq, inti_score};
    

    
    for (int i = 0; i < sequence.size(); ++i){
        auto symbol_i = sequence[i];
        auto prob_i   = prob[i];
        t_beam.update_beam(symbol_i, prob_i);
        
    
    }
    
    std::cout << "[main]: sequence: " << t_beam.get_sequence() << ", score: " << t_beam.get_score() << NL;
}


