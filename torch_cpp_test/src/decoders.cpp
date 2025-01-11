#include <greedy_decoder.hpp>




greedyDecoder::greedyDecoder(){std::cout<< "greedyDecoder instance created" << std::endl;};
greedyDecoder::~greedyDecoder(){};

bool greedyDecoder::init_vocab(const std::string& tokens_path){
    // Store the tokens into a lcoal buffer 
    std::ifstream tokens_file(tokens_path);
    std::string str;

    // Check if the file was opened sucessfully 
    if (!tokens_file.is_open()){
        std::cout<< "[init_vocab]: Tokens file was not open. Check that the file exist." << std::endl;
        return false;
    }

    // Read the file if open and store its tokens
    while (std::getline(tokens_file,str)){  
        if (str.size() > 1){
            std::cout<< "[init_vocab]: The file should contain just ons char per line" << std::endl;
            return false;
        }
        else
            tokens.push_back(str[0]);
    }

    // Set the number of tokens 
    num_tokens = static_cast<size_t>(tokens.size());
    std::cout<< "[init_vocab]: The number of tokens is: " << num_tokens << std::endl; 
    return true;
}


std::vector<char> greedyDecoder::decode_chars(torch::Tensor& emissions){
    // Ensure size compatibility 
    if (emissions.sizes()[1] != num_tokens){
        std::cout<< "[decode_chars]: The input tensor does not match the number of tokens" << std::endl;
    }

    // 1- Find the max probability value across the tokens dim
    auto max_info = torch::max(emissions,1);
    torch::Tensor max_vals = std::get<0>(max_info);
    torch::Tensor max_index = std::get<1>(max_info);


    // 2- Get the index
    auto tmp = max_index.data_ptr<int64_t>(); // TODO: I need to understand this casting better 
    std::vector<char> result;
    for (int i = 0; i < max_index.sizes()[0]; i++){
            result.push_back(tokens[tmp[i]]);
    }
    return result;
}


std::vector<char> greedyDecoder::get_tokens(){
        if (!tokens.empty())
            return tokens;
        else
            throw std::runtime_error("The decoder has no tokens. Call the init_vocab method.");
    }
