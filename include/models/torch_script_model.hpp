#ifndef TORCH_SCRIPT_MODEL
#define TORCH_SCRIPT_MODEL

#include <torch/script.h>
#include <glog/logging.h>
#include <optional>

class torchScriptModel{
private:
    torch::TensorOptions _tensor_options; 
    torch::jit::script::Module _model;
    std::string _model_path;
    

public:
    torchScriptModel();
    ~torchScriptModel();
    bool load_model(const std::string& _file_path);
    std::optional<torch::Tensor> pass_forward(std::vector<float>& audio_data);

};


#endif