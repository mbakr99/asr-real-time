#include <vector>
#include <string>
#include <torch/script.h>
#include <unordered_map>
#include <tuple>
#include <queue>
#include "beam.hpp"
#include "decoders/lexicon_fst.hpp"



struct decodingInfo{
    std::vector<char> idx_2_token;
    int max_chars_per_token = 1;
    int num_tokens = 0;
    int invalid_word_penalty = 1000; 
};


class ctcDecoder{

private:
    std::unordered_map<std::string, float> _beams_map; //FUTURE: I want to use a set instead, and modify the defualt behavior to accmulate score of identical beams
    std::priority_queue<
        beam::ctcBeam, 
        std::vector<beam::ctcBeam>,
        beam::beamGreaterThan> _beams_max_heap;
    int _max_num_beams;
    std::vector<beam::ctcBeam> _top_beams;
    LexiconFst _lex_fst; 
    decodingInfo _decoding_info;

    

public:
    ctcDecoder(const std::string& path_to_tokens);
    ctcDecoder(const std::string& path_to_tokens, std::string path_to_fst);
    ctcDecoder(const std::string& path_to_tokens, int num_beams);
    ctcDecoder(const std::string& path_to_tokens, LexiconFst fst);
    ctcDecoder(const std::string& path_to_tokens, int num_beams, LexiconFst fst);
    ~ctcDecoder();

    // top level 
    void set_fst(LexiconFst& fst);
    void decode_step(const torch::Tensor& emmission);
    std::vector<beam::ctcBeam> decode_sequence(torch::Tensor& emmissions);

    // main steps
    void expand_beam(const beam::ctcBeam& beam, const torch::Tensor& emmission);
    void reject_beam(beam::ctcBeam& beam);
    void update_top_beams();
    std::vector<beam::ctcBeam> get_top_beams();
    void clear_top_beams();
    bool is_word_valid(const std::string& word);
    
    
    // internal 
private:
    void read_tokens_file(const std::string& path_to_tokens);
    void penalize_beam(beam::ctcBeam& beam);
    void init_beams();
    void update_beams_map(const beam::ctcBeam& beam); 
    void clear_beams_map();
    void convert_map_to_min_heap();
    void clear_beams_min_heap();
    

};
