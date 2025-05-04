#include <fstream>
#include <torch/nn/functional.h>
#include "decoders/ctc_decoder.hpp"
#include "utils/my_utils.hpp"
#include "utils/fst_glog_safe_log.hpp"
#include <sstream>
#include <algorithm>

// convenience log warning at certain verbosity level 
#define VLOG_WARNING(verboselevel) if (VLOG_IS_ON(verboselevel)) LOG(WARNING)

float OOV_PENALTY = 1000;

using namespace asr;

#define pngram(ngram) for (const auto& word : ngram) std::cout << word << "-"
#define vlogram(ngram) for (const auto& word : ngram) VLOG(5) << word;

// construcotrs 
ctcDecoder::ctcDecoder(const std::string& path_to_tokens,
        size_t num_beams = 10) : _max_num_beams(num_beams){
    DLOG(INFO) << "[ctcDecoder/constructor]: instance created";
    read_tokens_file(path_to_tokens);
    init_beams();

}


ctcDecoder::ctcDecoder(const std::string& path_to_tokens, 
                       size_t num_beams, 
                       const fs::path& path_to_fst,
                       const fs::path& path_to_lm_model) : 
                            ctcDecoder::ctcDecoder(path_to_tokens, num_beams){         
        if (!set_beams_dictionary(path_to_fst)){
            std::ostringstream oss;
            oss << "failed to set the beams shared dictironary variables";
            throw std::runtime_error(oss.str());
        }
        if (path_to_lm_model.string() != "none"){ // if model is provided 
            LOG(INFO) << "[ctcDecoder/constructor]: language model was provided."; 
            set_lm(path_to_lm_model);
        }
        else{
            // _use_lm_model_flag is false by default (for symmetry)
            LOG(WARNING) << "[ctcDecoder/constructor]: no language modle was provided."
                         << "This will decrease decoding accuracy."; 
        }
}

ctcDecoder::~ctcDecoder(){
    DLOG(INFO) << "[ctcDecoder/destructor]: instance created";
}



// intialization and settings related 
bool ctcDecoder::set_beams_dictionary(const fs::path& path_to_fst){
    /*
    path_to_fst: this is path to the fst representing the dictionary 
                the implementation relies on this fst being sorted (FUTURE: can I add a check on this condition)
    */
    // load the dictionary fst
    auto dictionary_ptr = fst::StdVectorFst::Read(path_to_fst.string());
    if (!dictionary_ptr){
        LOG(WARNING) << "[LexiconFst/load_fst]: loaded fst is empty";
        return false;
    }

    // load the input symbol table 
    fs::path parent_directory = path_to_fst.parent_path();
    auto [input_symbol_table_ptr, _] = myfst::load_symbol_tables(parent_directory);
    if (!input_symbol_table_ptr){
        return false;
    }

    // set the shared dictionary and symbol table  
    beam::ctcBeam::set_fst(dictionary_ptr, input_symbol_table_ptr);
    return true;
}




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
    auto initial_beam = new beam::ctcBeam();
    initial_beam->score = initial_beam->prob_b_prev = 0;
    _top_beams.push_back(initial_beam); // sequence is empty be default 
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
void ctcDecoder::expand_beam(beam::ctcBeam* beam, 
        torch::Tensor& emission){
    // I am assuming that emission is (num_tokens x 1) or (num_tokens) 
    auto emmision_size = emission.sizes();
    if (emmision_size[0] != _decoding_info.idx_2_token.size()){
        DLOG(WARNING) << "[ctcDecoder/expand_beam]: emission has incompaitble shape: " << emmision_size[-1]
                      << ". Number of tokens is: " << _decoding_info.idx_2_token.size() << "\n"
                      << "Throwing exception";
        throw std::runtime_error("incompatible emission size");
    }

    // convert tensor to std::vector 
    emission = emission.contiguous();
    auto num_elements = emission.numel();
    std::vector<float> emission_vec(emission.data_ptr<float>(), emission.data_ptr<float>() + num_elements);

    // get the parent sequence prob 
    auto [prob_b_parent, prob_nb_parent] = beam->get_parent_probs(); 
    auto score_parent = beam->get_score(); // this is p_b + p_nb (used for handling the initial empty beam ) 
    auto parent_sequence = beam->get_sequence();

    // loop over pruned prob
    auto pruned_tokens_prob = myutils::get_pruned_log_probs(emission_vec, 0.95, 5, 1);


    for (const auto& [i, prob_i] : pruned_tokens_prob){
        
        double prob_b_child = -INF_DOUBLE;
        double prob_nb_child = -INF_DOUBLE;
        
        
        VLOG(5) << "[ctcDecoder/expand_beam]: token: " << _decoding_info.idx_2_token[i]
        << ", score: " << prob_i;

        // get current char
        char current_char =  _decoding_info.idx_2_token[i];
        
        // if blank (update current prefix)
        if (current_char == _decoding_info.blank_token){    
            /*
            the child sequence is the same as parent sequence,
            since adding a blank does not chane the sequence
            */

            auto existing_beam = _beams_map.find_beam(parent_sequence);
            if (!existing_beam){
                prob_b_child = myutils::log_sum_exp(prob_b_child, 
                    2 * prob_i + score_parent); 
            }
            else{
                auto existing_probs = existing_beam->get_current_probs();
                prob_b_child = existing_probs.first; // prob_nb_child = existing_probs.second;

                VLOG(5) << "[ctcDecoder/expand_beam]: "  
                        << "similar sequence exist. using p_b = " << prob_b_child;

                prob_b_child = myutils::log_sum_exp(prob_b_child, 
                    2 * prob_i + score_parent);  
            }

            beam->prob_b_cur = prob_b_child;
            _beams_map.add_beam(beam);

            // resret prob_b_child
            prob_b_child = -INF_DOUBLE;
            continue;
        }




        // if repeated char (update current prefix)
        char end_char = parent_sequence.back();
        if (current_char == end_char){
            
            auto existing_beam = _beams_map.find_beam(parent_sequence); // to take into consideration identical beams
            if (!existing_beam){
                prob_nb_child = myutils::log_sum_exp(prob_b_child, 
                2 * prob_i + prob_nb_parent);
            }
            else{
                auto existing_probs  = existing_beam->get_current_probs();
                prob_nb_child = existing_probs.second; // prob_b_child = existing_probs.first;

                VLOG(5) << "[ctcDecoder/expand_beam]: " 
                        << "similar sequence exist. using p_nb = " << prob_nb_child;
                prob_nb_child = myutils::log_sum_exp(prob_nb_child,
                    2 * prob_i + prob_nb_parent); 
            }

            beam->prob_nb_cur = prob_nb_child; 
            VLOG(5) << "updating p_nb of " << parent_sequence << " to " << prob_nb_child;
            _beams_map.add_beam(beam); 
            prob_nb_child = -INF_DOUBLE;
        }


        
        // check if new prefix exists in the map
        std::string child_prefix = parent_sequence + current_char;
        auto existing_beam = _beams_map.find_beam(child_prefix);
        beam::ctcBeam* child_beam = nullptr;
        bool child_is_new_beam = false;



        if (existing_beam){ // prefix already exist
            auto existing_probs = existing_beam->get_current_probs();
            prob_b_child = existing_probs.first; prob_nb_child = existing_probs.second;
            
            child_beam = existing_beam;
        }
        else{ 
            child_beam = beam->get_new_beam(_decoding_info.idx_2_token[i]);
            child_is_new_beam = true;
        }


        
        
        if (!child_beam){ // only skip if no existing beam and sequence not valid
            /*
            sequence is not valid 
            */
            continue;
        }


        VLOG(5) << "[ctcDecoder/expand_beam]: the child beam is: " << child_beam->sequence;
        // update new prefix score 
        double log_p = -INF_DOUBLE; 
        if (current_char == end_char && 
            prob_b_parent > -INF_DOUBLE){ // n_p_nb[new_prefix] = p(p_b[prefix])
            log_p = 2 * prob_i + prob_b_parent;
        }
        else if (current_char != end_char) {
            log_p = 2 * prob_i + score_parent; 
        }

        if (child_beam->is_full_word_fromed() &&
            _use_lm_model_flag){ 
            /*
            Note:decode_sequence
            With the new modifications to beam, adding a dictionary to the beam, it is not 
            required to check the validity of a word when it is formed. This is doen 
            by the beam now. If the fst is not set in the ctc decoder as it is own memebr
            i.e. not part of the beam, I will get a segmenation fault as I will be trying to
            access something that does not exist.
            
            For now (deugging), I will leave this functionality as it is. I will 
            have to set the the fst explicilty first. 
            */
            auto last_word = child_beam->get_last_word();
            VLOG(5) << "[ctcDecoder/expand_beam]: word is formed: " << last_word
                    << ", last word window: " << std::get<0>(child_beam->last_word_window.get_window())
                    << ", " << std::get<1>(child_beam->last_word_window.get_window());
            auto sentence = child_beam->get_sequence();                  
            // convert to upper case for compatibaility with lm model in FUTURE:
            // this has to be controlled by the decoding or scoring information 
            to_capital(sentence); 

            // get ngram and score 
            auto [_, sentence_end] = child_beam->last_word_window.get_window();
            auto ngram = child_beam->generate_ngrams(
                sentence, 
                _decoding_info.lm_order,
                _decoding_info.word_delimiter,
                _decoding_info.sentence_start_token, 
                sentence_end - 1 
            ); 


            auto lm_score = compute_lm_score(ngram);   
            VLOG(5) << "[ctcDecoder/expand_beam]: sentence: " << sentence
                    << ", lm score: "  << lm_score;

            // update log_p 
            log_p += lm_score * _decoding_info.alpha;
            /*
            in Awni's paper, the term |W|^beta is used, which in log space
            would be beta * log(|W|). For now, I will follow parlance implementation 
            */
            log_p += _decoding_info.beta;   
        }



        prob_nb_child = myutils::log_sum_exp(prob_nb_child, 
            log_p);
        
        child_beam->prob_nb_cur = prob_nb_child;
      

        // add to beams vector if it is new
        if (child_is_new_beam) {
            _beams_map.add_beam(child_beam);
        }
    }

    // add the parent beam to the beams map
    _beams_map.add_beam(beam);
    return ;
}
          
    


float ctcDecoder::compute_lm_score(const std::vector<std::string>& ngram){
    return (get_lm_model().score_sentence(ngram, ngrams::LOGITS)); // use logits instead of probability  
}


inline float ctcDecoder::get_weighted_score(const float& ctc_score, const float& lm_score){
    return (ctc_score + _decoding_info.alpha * lm_score);
}



void ctcDecoder::clear_top_beams(){
    if (!_top_beams.empty()){
        _top_beams.clear();
    }
}



void ctcDecoder::update_top_beams(){

    std::vector<beam::ctcBeam*> new_beams;
    _beams_map.clean_garbage();

    // convert beams map to vector 
    for (auto& [prefix, beam] : _beams_map) {
        beam->update_score();
        /*
        a more efficient implementation would use a max heap here
        */

        new_beams.push_back(beam); 
    }

    // clear the beams_map
    _beams_map.clear();

   
    // sort based on score 
    size_t num_beams = std::min(new_beams.size(), _max_num_beams); 
    std::sort( // not very efficient 
        new_beams.begin(), 
        new_beams.end(),
         myutils::prefix_compare
        );
    
    
    VLOG(5) << "[ctcDecoder/update_top_beams]: the sorted beams are: ";
    for (size_t i = 0; i < num_beams; ++i){
        VLOG(5) << "prefix: " << new_beams[i]->get_sequence()
                << ", score: " << new_beams[i]->score;
    }
    
    
    // delete beams outside of beam width 
    for (size_t i = num_beams; i < new_beams.size(); ++i){
        VLOG(5) << "deleting " << i << ", memory:" << new_beams[i] << ", sequence: " <<  new_beams[i]->get_sequence(); 
        delete new_beams[i];
    }

    // update the top beams vector 
    clear_top_beams();
    for (size_t i = 0; i < num_beams; ++i){
        _top_beams.push_back(new_beams[i]);
    }

    // no need to clear the new_beams vector as it will
    // go out of scope
}

std::vector<beam::ctcBeam*> ctcDecoder::get_top_beams(){
    return _top_beams;
} 



// decoding interface 
void ctcDecoder::decode_step(torch::Tensor& emission){
    if (_top_beams.empty()){
        VLOG_WARNING(5) << "[ctcDecoder/decode_step]: top beams are empty. "
                        << "This could be the result of not initializing _top_beams"; 
    }

    std::vector<beam::ctcBeam*> new_beams;
    static size_t step_counter = 1;
    VLOG(4) << "[ctcDecoder/decode_step]: Step " << step_counter << "\n" << "-----------------------------------------";
    for (beam::ctcBeam* beam : _top_beams){
        VLOG(4) <<  "[ctcDecoder/decode_step]: beam in consideration: " << beam->get_sequence()
                << ", address: " << beam
                << ", score: " << beam->get_score() 
                << ", sequence size: " << beam->sequence.size()
                << ", last_word: " << beam->last_word_window.word_begin << beam->last_word_window.word_end;
        expand_beam(beam, emission);
    }

    update_top_beams(); 
    ++step_counter;

    // print updated top beams
    VLOG(5) << "[ctcDecoder/decode_step]: top beams are: \n";
    for (size_t i = 0; i < _top_beams.size(); ++i){
        VLOG(5) << i << ": " << _top_beams[i]->get_sequence() << ", score: " << _top_beams[i]->get_score()
                << ", size of sequence: " << _top_beams[i]->sequence.size()
                << ", o_p_b: " << _top_beams[i]->prob_b_prev
                << ", o_p_nb: " << _top_beams[i]->prob_nb_prev; 
    }
}



std::vector<beam::ctcBeam*> ctcDecoder::decode_sequence(torch::Tensor& emissions){
    // ensure compaitble shape. I expect [time, features] input.
    auto emissions_squeezed = torch::squeeze(emissions); // remove redundant axis
    if (emissions_squeezed.sizes()[1] != _decoding_info.num_tokens){
        DLOG(WARNING) << "[ctcDecoder/decode_chars]: input tensor has incompatible shape: "
                      << emissions_squeezed.sizes() << " while the num of tokens is: " 
                      << _decoding_info.num_tokens;
    }

    // convert raw emissions to to log prob 
    auto emissions_score    = torch::nn::functional::log_softmax(emissions_squeezed,
                                torch::nn::functional::LogSoftmaxFuncOptions(1));

    // loop over time
    auto emissions_shape = emissions_score.sizes(); 
    std::cout << emissions_shape << std::endl;
    size_t num_time_steps = emissions_shape[0];
    for (size_t t = 0; t < num_time_steps; ++t){
        auto score_at_t =  emissions_score[t];
        decode_step(score_at_t);
    }

    return _top_beams;
}



