#include <vector>
#include <string>
#include <torch/script.h>
#include <unordered_map>
#include <tuple>
#include <queue>
#include "beam.hpp"
#include "decoders/beams_map.hpp"
#include "decoders/lexicon_fst.hpp"
#include "models/ngrams_model.hpp"
#include "utils/my_utils.hpp"

using namespace asr;

struct ctcScoreSettings{
    int lower_val = 0;
    int upper_val = 5;
};

struct DecodingInfo{
    std::vector<char> idx_2_token;
    int max_chars_per_token = 1;
    int num_tokens = 0;
    int invalid_word_penalty = 1000; 
    ctcScoreSettings ctc_score_info{};
    float alpha = 10; // score = ctc_score + alpha x lm_score
    char word_delimiter = '|';  

    // getters 
    std::tuple<int, int> get_ctc_score_limits(){
        return std::make_tuple(ctc_score_info.lower_val, ctc_score_info.upper_val);
    }

    // setters
    void set_alpha(float new_alpha){alpha = new_alpha;}
    void set_word_delimiter(char new_word_delimiter){word_delimiter = new_word_delimiter;};
};  


class ctcDecoder{

private:
    beam::BeamsMapWrapper _beams_map;
    std::priority_queue<
        beam::ctcBeam, 
        std::vector<beam::ctcBeam>,
        beam::beamGreaterThan> _beams_max_heap;
    int _max_num_beams;
    std::vector<beam::ctcBeam> _top_beams;
    LexiconFst _lex_fst; 
    DecodingInfo _decoding_info;
    ngrams::nGramsModelWrapper _ngrams_model; // this need the path to the model to be set
    bool _use_lm_model_flag = false; // FUTURE: I don't like the idea of having to repeatedly check a use
                                // condition that is static throughout the application lifetime
                                // I might use a strategy patten or a warpper function 
    
    

public:
    /*
    I don't want to burden the code reader or user with having to set tpo many arguments during intialization.
    So, I'll divide the paramters into a cruical set and functional set. The cruical set can not have defualt values 
    and are crucial for running the decoder will include:
        - the tokens path 
        - the lexicon fst path (I will ship this with the app in the future) TODO: 
         
    The other set will include:
        - the beam width 
    */
    ctcDecoder(const std::string& path_to_tokens, 
               int num_beams, LexiconFst& fst,
               const fs::path& path_to_lm_model);
    ctcDecoder(const std::string& path_to_tokens, int num_beams, LexiconFst& fst);
    ctcDecoder(const std::string& path_to_tokens, std::string path_to_fst);
    ctcDecoder(const std::string& path_to_tokens, int num_beams);
    ctcDecoder(const std::string& path_to_tokens, LexiconFst& fst); //FIXME: Am I usign this?
    
    ~ctcDecoder();

    // top level 
    void decode_step(const torch::Tensor& emmission);
    std::vector<beam::ctcBeam> decode_sequence(torch::Tensor& emmissions);

    // main steps
    void expand_beam(const beam::ctcBeam& beam, const torch::Tensor& emmission);
    void reject_beam(beam::ctcBeam& beam);
    void update_top_beams();
    std::vector<beam::ctcBeam> get_top_beams();
    void clear_top_beams();
    bool is_word_valid(const std::string& word);
    
    // control decocding settings
    void set_lm_weight(float new_alpha){_decoding_info.alpha = new_alpha;}

    // internal 
private:
    // settings related 
    void read_tokens_file(const std::string& path_to_tokens);
    void set_fst(LexiconFst& fst);
    bool set_lm(const fs::path& path_to_ngram_model);
    // functional (beams)
    void penalize_beam(beam::ctcBeam& beam);
    void init_beams();
    void update_beams_map(const beam::ctcBeam& beam); 
    void clear_beams_map();
    void scale_scores(const int& lower_range, const int& upper_range);
    void convert_map_to_min_heap();
    void clear_beams_min_heap();
    void init_beams_map_with_top_beams();
    // functional (lm)
    float compute_lm_score(const std::string& sentence);
    float compute_lm_score(const std::vector<std::string>& sentence);
    inline float get_weighted_score(const float& ctc_score, const float& lm_score);
    inline void to_capital(std::string& sequence){stringmanip::upper_case(sequence);}
    // getters 
    ngrams::nGramsModelWrapper& get_lm_model(){return _ngrams_model;}  //FUTURE: I should create a base lm class 



};
