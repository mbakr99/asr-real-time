#include <fstream>
#include <torch/nn/functional.h>
#include "decoders/ctc_decoder.hpp"
#include "utils/my_utils.hpp"
#include "utils/fst_glog_safe_log.hpp"

// convenience log warning at certain verbosity level 
#define VLOG_WARNING(verboselevel) if (VLOG_IS_ON(verboselevel)) LOG(WARNING)

ctcDecoder::ctcDecoder(const std::string& path_to_tokens) : _max_num_beams(10){
    DLOG(INFO) << "[ctcDecoder/constructor]: defualt instance created with "
               << _max_num_beams << " beams";
    read_tokens_file(path_to_tokens);
    init_beams();
}

ctcDecoder::ctcDecoder(const std::string& path_to_tokens, int num_beams) : _max_num_beams(num_beams){
    DLOG(INFO) << "[ctcDecoder/constructor]: instance created";
    read_tokens_file(path_to_tokens);
    init_beams();
}


ctcDecoder::~ctcDecoder(){
    DLOG(INFO) << "[ctcDecoder/destructor]: instance created";
}


void ctcDecoder::set_fst(LexiconFst& fst){_lex_fst = fst;}

void ctcDecoder::init_beams(){
    _top_beams.push_back(beam::ctcBeam());
}

void ctcDecoder::read_tokens_file(const std::string& path_to_tokens){
    std::ifstream tokens_file(path_to_tokens);
    if (!tokens_file.is_open()){
        DLOG(WARNING) << "[ctcDecoder/read_tokens_file]: failed to open the token file at "
                      << path_to_tokens
                      << "\n"
                      << "Throwing exception.";
        throw std::runtime_error("failed to read tokens file");
    }

    std::string token;
    while(std::getline(tokens_file, token)){ // TODO: this decoder assumes single char tokens for now 
        VLOG(6) << "[ctcDecoder/read_tokens_file]: adding token: " << token
                << " to the decoder tokens.";
        if (token.size() > _decoding_info.max_chars_per_token){
            DLOG(WARNING) << "[ctcDecoder/read_tokens_file]: tokens file had invalid layout. " 
                          << "Each line should contain "
                          << _decoding_info.max_chars_per_token
                          << "at max." << "\n" << "Throwing an exception";
            throw std::runtime_error("invalid tokens file layout. Violation of max chars per token.");
        }
        _decoding_info.idx_2_token.push_back(token[0]);
        _decoding_info.num_tokens++;
    
    }
}


void ctcDecoder::expand_beam(const beam::ctcBeam& beam, const torch::Tensor& emission){
    // I am assuming that emission is (num_tokens x 1) or (num_tokens) 
    auto emmision_size = emission.sizes();
    if (emmision_size[0] != _decoding_info.idx_2_token.size()){
        DLOG(WARNING) << "[ctcDecoder/expand_beam]: emission has incompaitble shape: " << emmision_size[-1]
                      << ". Number of tokens is: " << _decoding_info.idx_2_token.size() << "\n"
                      << "Throwing exception";
        throw std::runtime_error("incompatible emission size");
    }

    beam::ctcBeam new_beam;
    for (int i = 0; i < _decoding_info.idx_2_token.size(); ++i){
        new_beam = beam;
        VLOG(5) << "[ctcDecoder/expand_beam]: token: " << _decoding_info.idx_2_token[i]
                << ", score: " << emission[i].item<float>();
        new_beam.update_beam(_decoding_info.idx_2_token[i], emission[i].item<float>());
        VLOG(5) << "[ctcDecoder/expand_beam]: updated sequence: " << new_beam.get_sequence();
        if (new_beam.is_full_word_fromed()){
            auto last_word = new_beam.get_last_word();
            VLOG(5) << "[ctcDecoder/expand_beam]: word is formed: " << last_word;
            if (!is_word_valid(last_word)){ 
                // reject/penalize this beam TODO: This might need improvement 
                VLOG(6) << "[ctcDecoder/expand_beam]: " << last_word 
                        << " is not valid. Penalizing this beam.";
                penalize_beam(new_beam); 

            }
            else{ // I think this is stupid but I will leave it for now
                VLOG(6) << "[ctcDecoder/expand_beam]: " << last_word << " is valid";
            }
        }
        update_beams_map(new_beam);   
    }
    return ;
}


bool ctcDecoder::is_word_valid(const std::string& word){ 
    if (word.size()==0){
        DLOG(WARNING) << "[ctcDecoder/is_word_valid]: recieved an empty word.";
        return false;
    }
    VLOG(6) << "[ctcDecoder/is_word_valid]: checking " << word << " for validity";  
    return _lex_fst.is_sequence_valid_fst(word);
}


void ctcDecoder::penalize_beam(beam::ctcBeam& beam){
    beam.discount(_decoding_info.invalid_word_penalty); //_decoding_info.invalid_word_penalty
}


void ctcDecoder::update_beams_map(const beam::ctcBeam& beam){
    VLOG(5) << "[ctcDecoder/update_beams_map]: adding " << beam.get_sequence<std::string>()
            << " with score: " << beam.get_score() << " to the map.";
    _beams_map[beam.get_sequence<std::string>()] += beam.get_score() ;
}


void ctcDecoder::clear_beams_map(){
    VLOG(6) << "[ctcDecoder/clear_beams_map]: clearing beams map";
    if (!_beams_map.empty()){
        _beams_map.clear();
    }
}



void ctcDecoder::convert_map_to_min_heap(){
    VLOG(6) << "[ctcDecoder/convert_map_to_min_heap]: converting map to min heap for ranking.";
    if (!_beams_max_heap.empty()){
        VLOG(6) << "[ctcDecoder/conve<rt_map_to_min_heap] this is to check if clearing the"
                << " the heap ever gets called.";
        clearHeap(_beams_max_heap);
    }

    // push element to min heap
    for (const auto& [sequence, score] : _beams_map){
        VLOG(6) << "[ctcDecoder/convert_map_to_min_heap]: adding " << sequence
                << " to heap, score: " << score;
        _beams_max_heap.emplace(sequence,score);
        if (_beams_max_heap.size() > _max_num_beams){
            _beams_max_heap.pop();
        }
    }           
}


void ctcDecoder::update_top_beams(){
    convert_map_to_min_heap(); // to order the beams 
    clear_top_beams(); // clear existing top beams
    VLOG(6) << "[ctcDecoder/update_top_beams]: cleared the top beams for the update."
            << " _top_beams size (should be zero): " << _top_beams.size(); 
    while (!_beams_max_heap.empty()){ // update top beams
        _top_beams.push_back(_beams_max_heap.top());
        _beams_max_heap.pop();
    }
    std::reverse(_top_beams.begin(),_top_beams.end());
}   


void ctcDecoder::clear_top_beams(){
    VLOG(6) << "[ctcDecoder/clear_top_beams]: clearing top beams";
    if (!_top_beams.empty()){
        _top_beams.clear();
    }
}


std::vector<beam::ctcBeam> ctcDecoder::get_top_beams(){
    return _top_beams;
} 


void ctcDecoder::decode_step(const torch::Tensor& emission){
    /*
        1: loop over the beams 
        2: expand the beams with tokens 
        3: update the beam map
    */
    if (_top_beams.empty()){
        DLOG(WARNING) << "[ctcDecoder/decode_step]: top beams are empty.";
        VLOG_WARNING(6) << "[ctcDecoder/decode_step]: top beams are empty. "
                        << "This could be the result of not initializing _top_beams"; 
    }

    static size_t step_counter = 1;
    VLOG(4) << "[ctcDecoder/decode_step]: Step " << step_counter << "\n" << "-----------------------------------------";
    for (const auto& beam : _top_beams){
        VLOG(4) <<  "[ctcDecoder/decode_step]: beam in consideration: " << beam.get_sequence()
                << ", score: " << beam.get_score() 
                << ", last_word: " << beam.last_word_window.word_begin << beam.last_word_window.word_end;
        expand_beam(beam, emission);
    }
    update_top_beams();
    ++step_counter;

    // print updated top beams
    VLOG(5) << "[ctcDecoder/decode_step]: top beams are: \n";
    
    for (size_t i = 0; i < _top_beams.size(); ++i){
        VLOG(5) << i << ": " << _top_beams[i].get_sequence(); 
    }
}


std::vector<beam::ctcBeam> ctcDecoder::decode_sequence(torch::Tensor& emissions){
    // ensure compaitble shape. I expect [time, features] input.
    auto emissions_squeezed = torch::squeeze(emissions); // remove redundant axis
    if (emissions_squeezed.sizes()[1] != _decoding_info.num_tokens){
        DLOG(WARNING) << "[ctcDecoder/decode_chars]: input tensor has incompatible shape: "
                      << emissions_squeezed.sizes() << " while the num of tokens is: " 
                      << _decoding_info.num_tokens;
    }

    // convert raw emissions to to probability 
    auto emissions_score    = torch::nn::functional::softmax(emissions,
                                torch::nn::functional::SoftmaxFuncOptions(1));

    // loop over time
    auto emissions_shape = emissions_score.sizes();
    size_t num_time_steps = emissions_shape[0];
    for (size_t t = 0; t < num_time_steps; ++t){
        auto score_at_t =  emissions_score[t];
        decode_step(score_at_t);
    }

    return _top_beams;
}

        