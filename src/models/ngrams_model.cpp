#include "models/ngrams_model.hpp"
#include <glog/logging.h>


namespace asr{
    namespace ngrams{

nGramsModelWrapper::nGramsModelWrapper(fs::path path_to_ngrams_model) :
 path_to_ngrams_model_(path_to_ngrams_model){
    if (!load_model(get_model_path())){
        LOG(WARNING) << "[nGramsModelWrapper/constructor]: was not able to load the ngrams model at " << get_model_path(); 
        throw std::runtime_error("was not able to load the ngrams model");   
    };
    LOG(INFO) << "[nGramsModelWrapper/constructor]: instance created. loaded model from " << get_model_path();
    initialize_model();

}


nGramsModelWrapper::~nGramsModelWrapper() noexcept {
    LOG(INFO) << "[nGramsModelWrapper/destructor]: instance is out of scope";
}


bool nGramsModelWrapper::load_model(fs::path path_to_ngrams_model){
    if (!fs::exists(path_to_ngrams_model)){
        LOG(WARNING) << "nGramsModelWrapper/load_model: path " << path_to_ngrams_model << " does not exist";
        return false;
    }
    set_model_path(path_to_ngrams_model);
    ngram_model_ptr_ = std::make_unique<lm::ngram::Model>(get_model_path().string().c_str());
    LOG(INFO) << "[nGramsModelWrapper/load_model]: loaded model from " << get_model_path().string().c_str();
    return true;
}


void nGramsModelWrapper::set_internal_state(const lm::ngram::State& state){
    internal_state_ = state;
}


void nGramsModelWrapper::reset_internal_state(){
    if (!ngram_model_ptr_.get()){
        LOG(WARNING) << "[nGramsModelWrapper/set_init_intenral_state]: no model is loaded";
        return;
    }
    set_internal_state(ngram_model_ptr_->BeginSentenceState());
}


void nGramsModelWrapper::init_vocab(){
    if (!ngram_model_ptr_.get()){
        LOG(WARNING) << "[nGramsModelWrapper/set_vocab]: no model is loaded";
        return;
    }
    vocab_ptr_ = &(ngram_model_ptr_->GetVocabulary()); // FIXME: what happens of the ngrams_model_ptr_ get out of scope
    return;
}


void nGramsModelWrapper::initialize_model(){
    reset_internal_state();
    init_vocab();
}

 
WordIndex nGramsModelWrapper::get_word_index(const std::string& word) const {
    if (!ngram_model_ptr_.get()){
        LOG(WARNING) << "[nGramsModelWrapper/get_word_index]: no model is loaded";
        throw std::runtime_error("no model is loaded and hence no vocab exists");
    }
    // here
    return vocab_ptr_->Index(word);
}


float nGramsModelWrapper::score_word(const std::string& word){
    static State out_state;
    float score;
    WordIndex word_index = get_word_index(word);
    if (word_index == 0) { // word not found. FIXME: I remeber reading that 0 indicates word not found in the source code (I have to check again)
        LOG(WARNING) << "[nGramsModelWrapper/score_word]: word: " << word << " was not found in the vocab";
        return NF_NINF_SCORE; // lowest possible vlaue (probability of 0)
    }
    score = ngram_model_ptr_->Score(get_internal_state(), word_index, out_state);
    set_internal_state(out_state); // update the internal state 
    return score;
}


bool nGramsModelWrapper::setup_model_from(fs::path path_to_ngrams_model){
    if(!load_model(path_to_ngrams_model)){
        LOG(WARNING) << "[nGramsModelWrapper/load_model_from]: failed to load model form " << path_to_ngrams_model;
        return false;
    };

    initialize_model();
    return true;
} 


inline float nGramsModelWrapper:: map_score(float val) const {
    return scoring_config.score_map.map_val(convert_to_prob(val));
}; 



float nGramsModelWrapper::score_sentence(std::vector<std::string> sentence, scoreType score_type){

    double cond_prob;
    lm::ngram::State state, tmp_state, out_state;
    // avoid to inserting <s> in begin
    ngram_model_ptr_->NullContextWrite(&state);
    for (size_t i = 0; i < sentence.size(); ++i) {
      lm::WordIndex word_index = ngram_model_ptr_->BaseVocabulary().Index(sentence[i]);
      // encounter OOV
      if (word_index == 0) {
        return -OOV_PENALTY_;
      }
      cond_prob = ngram_model_ptr_->BaseScore(&state, word_index, &out_state);
      /*
      I don't get why saving the outstate 
      */
      tmp_state = state;
      state = out_state;
      out_state = tmp_state;
    }
    // return  loge prob
    auto log_prob = cond_prob / 0.4342944819; // Question: this number was found in paralnce implementation (whaty does it represent)


    switch (score_type){
        case LOGITS:
            return log_prob;
        case PROB:
            return convert_to_prob(log_prob);
        case MAPPED:
            return map_score(log_prob); // TODO: I still have to add a useer friendly way of setting the score to a range
        default:
            return log_prob;

    }

}


float nGramsModelWrapper::score_sentence(std::vector<std::string> sentence){
    return score_sentence(sentence, scoring_config.score_type);
} 


    } // namespace ngrams
} // namespace asr
