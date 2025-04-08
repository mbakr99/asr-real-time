#include "models/ngrams_model.hpp"


namespace asr{
    namespace ngrams{

nGramsModelWrapper::nGramsModelWrapper(fs::path path_to_ngrams_model) :
 path_to_ngrams_model_(path_to_ngrams_model){
    if (!load_model(get_model_path())){
        LOG(WARNING) << "[nGramsModelWrapper/constructor]: was not able to load the ngrams model at " << get_model_path();    
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
    vocab_ptr_ = &ngram_model_ptr_->GetVocabulary(); // FIXME: what happens of the ngrams_model_ptr_ get out of scope
    return;
}


void nGramsModelWrapper::initialize_model(){
    reset_internal_state();
    init_vocab();
}


void nGramsModelWrapper::start_new_sentence(){
    reset_internal_state();
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
    }
    score = ngram_model_ptr_->Score(get_internal_state(), word_index, out_state);
    set_internal_state(out_state); // update the internal state 
    return score;
}


float nGramsModelWrapper::score_sentence(std::vector<std::string> sentence, scoreType score_type = LOGITS){
    float tot_score;
    for (const auto word : sentence){
        tot_score = score_word(word);
    }

    switch (score_type){
        case LOGITS:
            return tot_score;
        case PROB:
            return conver_to_prob(tot_score);
        case MAPPED:
            return tot_score; // TODO: I still have to add a useer friendly way of setting the score to a range
        default:
            return tot_score;

    }

}


bool nGramsModelWrapper::setup_model_from(fs::path path_to_ngrams_model){
    if(!load_model(path_to_ngrams_model)){
        LOG(WARNING) << "[nGramsModelWrapper/load_model_from]: failed to load model form " << path_to_ngrams_model;
        return false;
    };

    initialize_model();
    return true;
} 


    } // namespace ngrams
} // namespace asr