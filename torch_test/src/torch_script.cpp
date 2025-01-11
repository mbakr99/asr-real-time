#include <iostream>
#include <torch/script.h>
#include <memory>

int main(int argc, char* argv[]){

    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try{
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }

    catch (const c10::Error& e){
        std::cerr << "error loading the model\n";
        return -1;
    }

  std::cout << "ok\n";

  // Created a dummy input 
  at::Tensor dummy_audio = at::randn({1, 1000});

  std::vector<torch::jit::IValue> input({dummy_audio});

  auto result = module.forward(input);

  auto output_tuple = result.toTuple();

  auto& emissions = output_tuple->elements()[0].toTensor();

  std::cout<< emissions.sizes() << std::endl;
  std::cout<< emissions << std::endl;

//   std::vector<torch::jit::IValue> inputs;
//   inputs.push_back(torch::rand({3,4}));

//   //
//   at::Tensor output = module.forward(inputs).toTuple()->elements()[0].toTensor();
//   std::cout<< output <<std::endl;


}