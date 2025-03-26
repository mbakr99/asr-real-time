#include "models/torch_script_model.hpp"
#include <glog/logging.h>




torchScriptModel::torchScriptModel(){
    _tensor_options = torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(torch::kCPU)
                                .requires_grad(false);
    DLOG(INFO) << "[torchScriptModel/constructor]: testing something";
}


torchScriptModel::~torchScriptModel(){
    DLOG(INFO) << "[torchScriptModel/destructor]: instance destroyed";
}


bool torchScriptModel::load_model(const std::string& _file_path){
    try{
        _model = torch::jit::load(_file_path);
        _model_path = _file_path;
    }
    catch (const c10::Error& e){ 
        DLOG(WARNING) << "[torchScriptoModel/load_model]: Error loading the model from "
                      << _file_path;
        return false;
    }
    DLOG(INFO) << "[torchScriptoModel/load_model]: Model has been loaded sucessfully";
    return true;
}


std::optional<torch::Tensor> torchScriptModel::pass_forward(std::vector<float>& audio_data){
    
    // chech input data validity 
    if (audio_data.empty()){
        return std::nullopt;
    }

    // convert the data into a tensor 
    void* data_ptr = static_cast<void*>(audio_data.data());
    auto audio_tensor = torch::from_blob(data_ptr,
                                        {1,static_cast<int64_t>(audio_data.size())},
                                        _tensor_options
                                        );

    // wrap the tensor in torcch::jit::IValue
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(audio_tensor);  

    // pass the data to the model  
    auto output = _model.forward(inputs);
    if (output.isTuple()){
        VLOG(3) << "[torchScriptModel/pass_forward]:" 
                << "output is a tuple, extracting the first element.";
        return output.toTuple()->elements()[0].toTensor().squeeze();
    }
    else{
        VLOG(3) << "[torchScriptModel/pass_forward]:" 
                << "output is a tensor, returning the tesnor." << std::endl;
        return output.toTensor().squeeze();
    }
}


