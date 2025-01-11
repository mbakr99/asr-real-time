#include <iostream>
#include <vector>
#include <torch/script.h>

int main(int argc, char* argv[]){

    char* model_path = argv[1];
    std::string path_str = std::string(model_path);
    try{
            // Deserialize the ScriptModule from a file using torch::jit::load().
            torch::jit::Module module = torch::jit::load(path_str);
        }
        catch (const c10::Error& e){
            std::cerr << "[torchScriptoModel/load_model]: Error loading the model\n";
            return false;
        }
        std::cout<< "[torchScriptoModel/load_model]: Model has been loaded sucessfully" << std::endl;
        return true;
}

