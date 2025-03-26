#include <gtest/gtest.h>
#include "decoders/greedy_decoder.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

class greedyDecoderTest : public testing::Test{
protected:
    greedyDecoderTest(){};
    greedyDecoder greedy_decoder{};
};



TEST_F(greedyDecoderTest, handles_invalid_token_file_path){
    char* project_root_ptr = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root_ptr) << "PROJECT_ROOT environement varibale not set.";
    
    // set a dummy tokens path      
    fs::path project_root   = fs::path(project_root_ptr); 
    fs::path dictionary_dir = project_root   / "data" / "dictionary";
    fs::path dummy_path     = dictionary_dir / "dummy.txt";

    // the function shoudl return a flase flag
    bool init_vocab_success = greedy_decoder.init_vocab(dummy_path);
    EXPECT_FALSE(init_vocab_success);
}


TEST_F(greedyDecoderTest, reads_tokens_file){
    char* project_root_ptr = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root_ptr) << "PROJECT_ROOT environement varibale not set.";
    
    // set the actual tokens path      
    fs::path project_root   = fs::path(project_root_ptr); 
    fs::path dictionary_dir = project_root   / "data" / "dictionary";
    fs::path tokens_path    = dictionary_dir / "tokens.txt";

    // the function should return a ture flag
    bool init_vocab_success = greedy_decoder.init_vocab(tokens_path);
    ASSERT_TRUE(init_vocab_success) << "was not able to read tokens from "
                                    << tokens_path;

    // print the tokens 
    auto tokens_result = greedy_decoder.get_tokens();
    ASSERT_TRUE(tokens_result.has_value()) << "not able to read tokens";

    auto tokens = tokens_result.value();
    for (size_t i = 0; i < tokens.size(); ++i)
        std::cout << tokens[i] << " ";
    std::cout << std::endl;
    
}


TEST_F(greedyDecoderTest, hanldes_tokens_request_when_empty){
    char* project_root_ptr = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root_ptr) << "PROJECT_ROOT environement varibale not set.";
    
    // request tokens
    auto token_result = greedy_decoder.get_tokens();
    EXPECT_FALSE(token_result.has_value()); 
}


TEST_F(greedyDecoderTest, handles_invlaide_tensor_shape){
    char* project_root_ptr = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root_ptr) << "PROJECT_ROOT environement varibale not set.";
    
   // set the actual tokens path      
   fs::path project_root   = fs::path(project_root_ptr); 
   fs::path dictionary_dir = project_root   / "data" / "dictionary";
   fs::path tokens_path    = dictionary_dir / "tokens.txt";

    // the function should return a ture flag
    bool init_vocab_success = greedy_decoder.init_vocab(tokens_path);
    ASSERT_TRUE(init_vocab_success) << "was not able to read tokens from "
                                    << tokens_path;

    // reuqest tokens 
    auto token_reult = greedy_decoder.get_tokens();
    EXPECT_TRUE(token_reult.has_value());

    // create a dummy tensor
    int num_steps = 256;
    int features  = 20; // the actual number of tokens "tokens.txt" is 28  
    auto dummy_emmission_tensor = torch::rand({num_steps, features});
    try{
        greedy_decoder.decode_chars(dummy_emmission_tensor);
        EXPECT_TRUE(false) << "the greedy decoder (decode_chars) did not throw an expetion "
                           << "when presented with a tensor of incompatibel shape";
    }
    catch(std::runtime_error& e){
        EXPECT_TRUE(true) <<  e.what();
    }
}