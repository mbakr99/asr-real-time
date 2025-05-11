#include "models/ngrams_model.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace asr::ngrams;

#define MYLOG(level) std::cout << #level << ": " 

std::vector<std::string> get_words_from_sentence(std::string sentence){
    std::string word;
    std::vector<std::string> result;
    for (int i = 0; i < sentence.size(); ++i){
        word += sentence[i];
        if (sentence[i] == ' ' ){
            word.pop_back();
            result.push_back(word);
            word.clear();
        }
    }
    result.push_back(word);
    return result;
}


class nGramsModelTest : public testing::Test{
protected:
    nGramsModelTest(){};
    nGramsModelWrapper ngrams_model{}; 


};


TEST_F(nGramsModelTest, handles_invalid_model_file){
    char* project_root_ptr = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root_ptr) << "Project root need to be set";
    fs::path invalid_model_path = fs::path(project_root_ptr) / "data" / "models" / "fake_model.apra";

    EXPECT_FALSE(ngrams_model.setup_model_from(invalid_model_path));
}

TEST_F(nGramsModelTest, loads_model_correclty){
    char* project_root = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root) << "project root variable must be set";
    fs::path model_path = fs::path(project_root) / "data" / "models" / "3-gram.pruned.1e-7.arpa";
    

    MYLOG(INFO) << model_path.string().c_str();
    EXPECT_TRUE(ngrams_model.setup_model_from(model_path)); 
}

TEST_F(nGramsModelTest, scoring_makes_sense){
    /* sentences that are coherewnt shoudl produce higher score */
    std::string sentence_familiar = "I LOVE YOU";
    std::string sentence_rare     = "I BLUE YOU";
    float score_familiar_sentence, score_rare_sentence;

    // model path 
    char* project_root = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root) << "project root variable must be set";
    fs::path model_path = fs::path(project_root) / "data" / "models" / "3-gram.pruned.1e-7.arpa";

    ASSERT_TRUE(ngrams_model.setup_model_from(model_path)) << "failed to load the model";

    // get the score of the new sentences 
    score_familiar_sentence = ngrams_model.score_sentence(get_words_from_sentence(sentence_familiar), PROB);
    ngrams_model.start_new_sentence();
    score_rare_sentence = ngrams_model.score_sentence(get_words_from_sentence(sentence_rare), PROB);

    MYLOG(INFO) << "sentence: " << sentence_familiar << ", score: " << score_familiar_sentence << "\n"
                << "sentence: " << sentence_rare << ", score: " << score_rare_sentence;

    EXPECT_GT(score_familiar_sentence, score_rare_sentence) << "somethign is off, the rare sensore had high probalbility";

}

