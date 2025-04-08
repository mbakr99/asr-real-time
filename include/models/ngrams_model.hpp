#include <kenlm/lm/model.hh>
#include <glog/logging.h>
#include <filesystem>
#include <optional>
#include <cmath>

/*base::
What is the beahvior I am trying to achieve?
- When given a sentence, I want to get the score as a- as it is b- probability c- probabilisty mapped to a range
    - A method shoudl exist for computign the score for a single word 
- Since the score represents on an internal state (check the lm::ngrams::Model source code) I want to 
be able to reset the internal state
- 
*/

namespace fs = std::filesystem;

namespace asr{
    namespace ngrams{

typedef lm::ngram::State State;
typedef lm::WordIndex WordIndex;

enum scoreType {
    LOGITS = 1,
    PROB   = 2,
    MAPPED = 3
};

class nGramsModelWrapper{

private:    
    typedef std::unique_ptr<lm::ngram::Model> ModelPtr;


public:
    nGramsModelWrapper(){};
    nGramsModelWrapper(fs::path path_to_model);
    ~nGramsModelWrapper();

    // getters 
    fs::path get_model_path() const {return path_to_ngrams_model_;}
    
    // 
    bool setup_model_from(fs::path path_to_model);
    
    // functionality
    float score_word(const std::string& word);
    float score_sentence(std::vector<std::string> sentence, scoreType score_type);
    void start_new_sentence(); 



private:
    bool load_model(fs::path path_to_ngrams_model);
    void init_vocab();
    void set_internal_state(const lm::ngram::State& state);
    void set_model_path(const fs::path& path_to_model){path_to_ngrams_model_ = path_to_model;}
    void reset_internal_state();
     
    float map_score(float& lower_range_val, float& upper_range_val) const; 
    WordIndex get_word_index(const std::string& word) const;
    void initialize_model();
    State get_internal_state(){return internal_state_;}
    float conver_to_prob(float score){return exp(score);};
    
    

private:
    ModelPtr ngram_model_ptr_;
    fs::path path_to_ngrams_model_;
    State internal_state_;
    const lm::ngram::ProbingVocabulary* vocab_ptr_;

};


    } // namespcae ngrams
} // namespcae asr 