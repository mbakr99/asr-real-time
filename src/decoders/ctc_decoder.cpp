#include <fstream>
#include <torch/nn/functional.h>
#include "decoders/ctc_decoder.hpp"
#include "utils/my_utils.hpp"
#include "utils/fst_glog_safe_log.hpp"

// convenience log warning at certain verbosity level 
#define VLOG_WARNING(verboselevel) if (VLOG_IS_ON(verboselevel)) LOG(WARNING)


// construcotrs 
ctcDecoder::ctcDecoder(const std::string& path_to_tokens, int num_beams = 10) : _max_num_beams(num_beams), _beams_map{_max_num_beams}{
    DLOG(INFO) << "[ctcDecoder/constructor]: instance created";
    read_tokens_file(path_to_tokens);
    _beams_map.set_beams_width(_max_num_beams);
    init_beams();
}


ctcDecoder::ctcDecoder(const std::string& path_to_tokens, int num_beams, 
    LexiconFst& fst, const fs::path& path_to_lm_model) : ctcDecoder::ctcDecoder(path_to_tokens, num_beams){
        set_fst(fst);
        if (path_to_lm_model.string() != "none"){ // if model is provided 
            LOG(INFO) << "[ctcDecoder/constructor]: language modle was provided."; 
            set_lm(path_to_lm_model);
        }
        else{
            // _use_lm_model_flag is false by default (for symmetry)
            LOG(WARNING) << "[ctcDecoder/constructor]: no language modle was provided."
                         << "This will decrease decoding accuracy."; 
        }
    }

ctcDecoder::ctcDecoder(const std::string& path_to_tokens, int num_beams, LexiconFst& fst) : 
    ctcDecoder::ctcDecoder(path_to_tokens, num_beams){
        set_fst(fst);
    }


ctcDecoder::~ctcDecoder(){
    DLOG(INFO) << "[ctcDecoder/destructor]: instance created";
}



// intialization and settings related 
void ctcDecoder::set_fst(LexiconFst& fst){_lex_fst = fst;}


bool ctcDecoder::set_lm(const fs::path& path_to_lm_model){
    LOG(INFO) << "[ctcDecoder/set_lm]: setting lm model at " << path_to_lm_model;
    bool loading_sucess  = get_lm_model().setup_model_from(path_to_lm_model);
    if (!loading_sucess){
        LOG(WARNING) << "[ctcDecoder/set_lm]: failed to load the lm model from " << path_to_lm_model;
        return false;
    }
    _use_lm_model_flag = true;
    return true;
}   

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


// internal steps 
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
                // reject/penalize this beam 
                VLOG(6) << "[ctcDecoder/expand_beam]: " << last_word 
                        << " is not valid. Penalizing this beam.";
                penalize_beam(new_beam); 

            }
            else{ // word is formed get the lm score 
                VLOG(6) << "[ctcDecoder/expand_beam]: " << last_word << " is valid";
                float lm_score = new_beam.get_score(); // defaul value if no lm model is used
                if (_use_lm_model_flag){ // if lm exist 
                    auto sentence = new_beam.get_sequence();
                    to_capital(sentence); // convert to upper case for compatibaility with lm model TODO:
                    // this has to be controlled by the decoding or scoring information // FUTURE:
                    lm_score = compute_lm_score(sentence);
                    VLOG(5) << "[ctcDecoder/expand_beam]: sentence: " << sentence
                            << ", lm score: "  << lm_score
                            << ", tot score: " << get_weighted_score(new_beam.get_score(), lm_score); 
                }
                float weighted_score = get_weighted_score(new_beam.get_score(), lm_score); 
                new_beam.set_score(weighted_score); // update beam with the empty token (will not change ) 
            }
        }
        update_beams_map(new_beam);   
    }
    return ;
}

float ctcDecoder::compute_lm_score(const std::vector<std::string>& sentence){
    std::cout << "[ctcDecoder/compute_lm_score]: lm recieved this sentence: " << std::endl;
    for (const auto& word :  sentence) std::cout << word << ", ";
    std::cout << std::endl;
    return (get_lm_model().score_sentence(sentence)); // use the score type set by the scoringConfig 
}

float ctcDecoder::compute_lm_score(const std::string& sentence){
    return compute_lm_score(stringmanip::break_to_words(sentence, _decoding_info.word_delimiter)); 
}

inline float ctcDecoder::get_weighted_score(const float& ctc_score, const float& lm_score){
    return (ctc_score + _decoding_info.alpha * lm_score);
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
    beam.zero_out_score(); // discount(_decoding_info.invalid_word_penalty);
}


void ctcDecoder::update_beams_map(const beam::ctcBeam& beam){
    // this methods serves as a proxy to the update_beams_map of class BeamsMap
    _beams_map.update_beams_map(beam); 
}


void ctcDecoder::clear_beams_map(){
    _beams_map.clear_beams_map();
}

void ctcDecoder::scale_scores(const int& lower_range, const int& upper_range){
    _beams_map.scale_beams_score(lower_range, upper_range);
}

void ctcDecoder::convert_map_to_min_heap(){
    VLOG(6) << "[ctcDecoder/convert_map_to_min_heap]: converting map to min heap for ranking.";
    if (!_beams_max_heap.empty()){
        VLOG(6) << "[ctcDecoder/conve<rt_map_to_min_heap] this is to check if clearing the"
                << " the heap ever gets called.";
        clearHeap(_beams_max_heap);
    }

    // push element to min heap
    for (auto& [sequence, beam] : _beams_map.get_map()){
        VLOG(6) << "[ctcDecoder/convert_map_to_min_heap]: adding " << sequence
                << " to heap, score: " << beam.get_score();
        
        _beams_max_heap.push(beam);
        if (_beams_max_heap.size() > _max_num_beams){
            _beams_max_heap.pop();
        }
    }       

}


void ctcDecoder::init_beams_map_with_top_beams(){
    if (_top_beams.empty()){
        LOG(WARNING) << "[ctcDecoder/init_beams_map_with_top_beams]: " 
                     << "no top beams found. Make sure update_top_beams is called first";
        return;
    }
    else{
        if (!_beams_map.is_empty()){
            clear_beams_map();
        }
        for (const auto& beam : get_top_beams()){
            update_beams_map(beam);
        }
        return;
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



// decoding interface 
void ctcDecoder::decode_step(const torch::Tensor& emission){
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
    static auto limits_tuple = _decoding_info.get_ctc_score_limits(); // FIXME: What happens if I change the limits?
    scale_scores(std::get<0>(limits_tuple), std::get<1>(limits_tuple));
    update_top_beams();
    clear_beams_map();
    init_beams_map_with_top_beams();
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



// language model related 