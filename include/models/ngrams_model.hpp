#include <kenlm/lm/model.hh>
#include <filesystem>
#include <optional>
#include <cmath>
#include <limits>
/*base::
What is the beahvior I am trying to achieve?
- When given a sentence, I want to get the score as a- as it is b- probability c- probabilisty mapped to a range
    - A method shoudl exist for computign the score for a single word 
- Since the score represents on an internal state (check the lm::ngrams::Model source code) I want to 
be able to reset the internal state
- 
*/


// Numerical Limits
constexpr float NF_NINF_SCORE = std::numeric_limits<float>::lowest(); // based on using float for score


namespace fs = std::filesystem;

namespace asr{
    namespace ngrams{



enum scoreType {
    LOGITS = 1,
    PROB   = 2,
    MAPPED = 3
};


struct lmScoringConfig{
    // mapping to a range 
    struct scoreMap{
        float lower_map_val = 0;
        float upper_map_val = 5;

        // for now only since I know that the output represent probability  FIXME: 
        float min_val = 0;
        float max_val = 1;

        void set_map_range(float& low, float& high){
        lower_map_val = low;
        upper_map_val = high;
        }

        float map_val(float val, const float& val_min, const float& val_max) const {
            return (val - val_min) / (val_max - val_min) * 
                (upper_map_val - lower_map_val) + lower_map_val; 
        }

        float map_val(float val) const {
            return map_val(val, min_val, max_val);
        }

    } score_map;

    // control scoring beahvior   
    float length_penalty = 0.6; // to account for decreaing length 
    scoreType score_type = MAPPED; // FIXME:

    // external 
};
    

typedef lm::ngram::State State;
typedef lm::WordIndex WordIndex;


class nGramsModelWrapper{
private:    
    typedef std::unique_ptr<lm::ngram::Model> ModelPtr;


public:
    nGramsModelWrapper(){};
    nGramsModelWrapper(fs::path path_to_model);
    nGramsModelWrapper(fs::path path_to_model, lmScoringConfig scoring_config); // FUTURE: 
    ~nGramsModelWrapper();
    // getters 
    fs::path get_model_path() const {return path_to_ngrams_model_;}
    // 
    bool setup_model_from(fs::path path_to_model);
    // functionality
    float score_word(const std::string& word);
    float score_sentence(std::vector<std::string> sentence);
    float score_sentence(std::vector<std::string> sentence, scoreType score_type); 
    void start_new_sentence(); 

private:
    bool load_model(fs::path path_to_ngrams_model);
    void init_vocab();
    void set_internal_state(const lm::ngram::State& state);
    void set_model_path(const fs::path& path_to_model){path_to_ngrams_model_ = path_to_model;}
    void reset_internal_state();
    inline float map_score(float val) const; 
    WordIndex get_word_index(const std::string& word) const;
    void initialize_model();
    State get_internal_state(){return internal_state_;}
    float convert_to_prob(float score) const {return exp(score);};


private:
    ModelPtr ngram_model_ptr_;
    fs::path path_to_ngrams_model_;
    State internal_state_;
    const lm::ngram::ProbingVocabulary* vocab_ptr_;
    lmScoringConfig scoring_config;
    float OOV_PENALTY_ = 1000;

};


    } // namespcae ngrams
} // namespcae asr 