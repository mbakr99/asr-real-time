#include "models/torch_script_model.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

class torchScriptModelTest : public testing::Test{
protected:
    torchScriptModelTest(){};
    torchScriptModel script_model{}; 


};


TEST_F(torchScriptModelTest, handles_invalid_model_file){
    fs::path current_dir = fs::current_path();
    fs::path invalid_model_path = current_dir / "fake.pt";
    EXPECT_FALSE(script_model.load_model(invalid_model_path));
}

TEST_F(torchScriptModelTest, loads_model_correclty){
    char* project_root = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root) << "project root variable must be set";
    fs::path models_dir = fs::path(project_root) / "data" / "models";
    fs::path model_path = models_dir / "model.pt";
    EXPECT_TRUE(script_model.load_model(model_path)); 
}

TEST_F(torchScriptModelTest, handles_empty_input){
    char* project_root = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root) << "project root variable must be set";
    fs::path models_dir = fs::path(project_root) / "data" / "models";
    fs::path model_path = models_dir / "model.pt";
    bool is_model_loaded = script_model.load_model(model_path);
    ASSERT_TRUE(is_model_loaded); 

    std::vector<float> dummy_input;
    auto model_result = script_model.pass_forward(dummy_input);
    ASSERT_EQ(model_result, std::nullopt);
}

TEST_F(torchScriptModelTest, handles_regular_input){
    char* project_root = std::getenv("PROJECT_ROOT");
    ASSERT_TRUE(project_root) << "project root variable must be set";
    fs::path models_dir = fs::path(project_root) / "data" / "models";
    fs::path model_path = models_dir / "model.pt";
    bool is_model_loaded = script_model.load_model(model_path);
    ASSERT_TRUE(is_model_loaded); 

    std::vector<float> dummy_input;
    for (size_t i = 0; i < 1000; ++i)
        dummy_input.push_back(i);

    auto model_result = script_model.pass_forward(dummy_input);
    ASSERT_TRUE(model_result.has_value()); 
    std::cout << model_result.value();
}
